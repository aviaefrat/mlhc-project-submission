import os
from pathlib import Path

import pandas as pd
import duckdb

from . import utils

labs_logger = utils.make_logger("labs", "labs.log")


def create_all_labs_for_filtered_cohort_0_48h(cohort_path: Path = None, db_path: Path = None) -> None:
    """
    Pull labs from LABEVENTS restricted to t in [admittime, admittime+48h),
    for admissions in cohort_path and ITEMIDs in data/labs.csv.
    Writes directly to parquet at utils.FIRST_48H_LABS_PATH.
    """
    labs_map = pd.read_csv(utils.LABS_CSV)
    itemids = labs_map["itemid"].astype("int64").unique().tolist()

    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA threads=%d" % os.cpu_count())
        con.execute("PRAGMA memory_limit='28GB'")

        out_path = utils.FIRST_48H_LABS_PATH.as_posix()

        labs_logger.info(f"Running 0–48h LABEVENTS extract -> {utils.FIRST_48H_LABS_PATH}")

        sql = f"""
        CREATE OR REPLACE TEMP TABLE filtered_cohort AS
        SELECT
            CAST(subject_id AS INTEGER)   AS subject_id,
            CAST(admission_id AS INTEGER) AS admission_id,
            CAST(admittime AS TIMESTAMP)  AS admittime
        FROM read_parquet('{cohort_path.as_posix()}');

        COPY (
            SELECT
                le.subject_id,
                le.admission_id,
                le.itemid,
                le.charttime,
                le.valuenum,
                le.value,
                le.valueuom
            FROM (
                SELECT
                    CAST(subject_id AS INTEGER)   AS subject_id,
                    CAST(hadm_id    AS INTEGER)   AS admission_id,
                    CAST(itemid     AS INTEGER)   AS itemid,
                    CAST(charttime  AS TIMESTAMP) AS charttime,
                    CAST(valuenum   AS DOUBLE)    AS valuenum,
                    CAST(value      AS VARCHAR)   AS value,
                    CAST(valueuom   AS VARCHAR)   AS valueuom
                FROM labevents
                WHERE CAST(itemid AS INTEGER) IN ({",".join(str(x) for x in itemids)})
            ) AS le
            JOIN filtered_cohort AS m
              ON le.admission_id = m.admission_id
            WHERE le.charttime >= m.admittime
              AND le.charttime <  m.admittime + INTERVAL 48 HOUR
        ) TO '{out_path}' (FORMAT PARQUET);
        """
        con.execute(sql)
        labs_logger.info("Done writing labs parquet.")
    finally:
        con.close()


def remove_labs_missing_value_and_valuenum(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    value_na = df["value"].isna()
    valuenum_na = df["valuenum"].isna()
    both_na_mask = value_na & valuenum_na

    n_both = int(both_na_mask.sum())
    if n_both:
        labs_logger.info(f"Dropping {n_both:,} / {n_before:,} rows with BOTH value & valuenum NA.")
        per_item = df.loc[both_na_mask, "itemid"].value_counts().sort_values(ascending=False)
        for itemid, n in per_item.items():
            labs_logger.info(f"  itemid {int(itemid)}: dropped {int(n):,}")
    else:
        labs_logger.info("No rows with BOTH value & valuenum NA.")

    kept = df.loc[~both_na_mask].copy()
    labs_logger.info(f"Kept {len(kept):,} rows after missing-value filter (from {n_before:,}).")
    return kept


def remove_labs_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # get bounds
    labs_map = pd.read_csv(utils.LABS_CSV)
    bounds = (
        labs_map[["itemid", "min", "max"]]
        .astype({"itemid": "int64"})
        .rename(columns={"min": "min_bound", "max": "max_bound"})
    )
    work = df.copy()
    work["itemid"] = work["itemid"].astype("int64")
    work = work.merge(bounds, on="itemid", how="left")

    # Coerce valuenum -> numeric; drop non-numeric
    valuenum_num = pd.to_numeric(work["valuenum"], errors="coerce")
    non_numeric_mask = valuenum_num.isna()
    n_non_numeric = int(non_numeric_mask.sum())
    if n_non_numeric:
        labs_logger.info(
            f"remove_labs_out_of_bounds: dropping {n_non_numeric:,} row(s) with non-numeric `valuenum`."
        )
        per_item_non_numeric = (
            work.loc[non_numeric_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        for itemid, cnt in per_item_non_numeric.items():
            labs_logger.info(f"  itemid {int(itemid)}: non-numeric valuenum rows dropped = {int(cnt):,}")
    work["valuenum"] = valuenum_num

    # Out-of-bounds (only where bounds exist)
    have_bounds = work["min_bound"].notna() & work["max_bound"].notna()
    oob_mask = have_bounds & work["valuenum"].notna() & (
        (work["valuenum"] < work["min_bound"]) | (work["valuenum"] > work["max_bound"])
    )
    n_oob = int(oob_mask.sum())
    if n_oob:
        labs_logger.info(
            f"remove_labs_out_of_bounds: dropping {n_oob:,} row(s) out of bounds (exclusive of equals)."
        )
        per_item_oob = (
            work.loc[oob_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        for itemid, cnt in per_item_oob.items():
            vmin = work.loc[work["itemid"] == itemid, "min_bound"].iloc[0]
            vmax = work.loc[work["itemid"] == itemid, "max_bound"].iloc[0]
            labs_logger.info(
                f"  itemid {int(itemid)}: out-of-bounds dropped = {int(cnt):,} (allowed [{vmin}, {vmax}])"
            )

    # log if any rows lacked bounds (Avia: this isn't supposed to happen anymore, I'm keeping it here for sanity)
    no_bounds_mask = ~have_bounds
    n_no_bounds = int(no_bounds_mask.sum())
    if n_no_bounds:
        n_itemids_missing = work.loc[no_bounds_mask, "itemid"].nunique()
        labs_logger.info(
            f"remove_labs_out_of_bounds: {n_no_bounds:,} row(s) across {n_itemids_missing:,} itemid(s) "
            f"had no bounds in {utils.LABS_CSV}. Rows kept."
        )

    # Keep rows that pass numeric + bounds checks
    keep_mask = (~non_numeric_mask) & (~oob_mask)
    out = work.loc[keep_mask, df.columns].copy()

    labs_logger.info(
        f"remove_labs_out_of_bounds: kept {len(out):,} / {n0:,} rows "
        f"(dropped {n_non_numeric + n_oob:,}: non-numeric={n_non_numeric:,}, out-of-bounds={n_oob:,})."
    )
    return out


def add_feature_name_by_itemid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'feature_name' column using labs.csv.
    """
    labs_map = pd.read_csv(utils.LABS_CSV)

    # Required columns
    req = {"itemid", "feature name"}
    missing = req - set(labs_map.columns)
    if missing:
        raise ValueError(f"{utils.LABS_CSV} missing required columns: {sorted(missing)}")

    labs_map = labs_map.astype({"itemid": "int64"})
    # make sure each itemid has only one feature name:
    n_names = (
        labs_map.groupby("itemid", as_index=False)["feature name"]
        .nunique(dropna=False)
        .rename(columns={"feature name": "n_names"})
    )
    dup_bad = n_names[n_names["n_names"] > 1]
    if not dup_bad.empty:
        labs_logger.error("Multiple distinct 'feature name' values detected for some itemids in labs.csv.")
        for r in dup_bad.itertuples(index=False):
            labs_logger.error(f"  itemid={int(r.itemid)} has {int(r.n_names)} distinct names")
        raise ValueError("Non-unique itemid->feature name mapping in labs.csv")

    mapping = dict(zip(labs_map["itemid"].to_numpy(), labs_map["feature name"].to_numpy()))

    out = df.copy()
    out["itemid"] = out["itemid"].astype("int64")
    out["feature_name"] = out["itemid"].map(mapping).astype("string")

    miss = int(out["feature_name"].isna().sum())
    if miss:
        # this should never happen.
        labs_logger.error(f"{miss:,} rows failed to map itemid->feature_name unexpectedly.")
        raise ValueError("Some rows failed to map feature_name unexpectedly.")
    labs_logger.info(f"add_feature_name_by_itemid: mapped feature_name for {len(out):,} rows.")
    return out


def create_labs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps columns: subject_id, admission_id, itemid, charttime, valuenum, value, valueuom, feature_name
    """
    labs_logger.info(f"Filtering rows where BOTH value and valuenum are NAs for {len(df):,} rows:")
    df = remove_labs_missing_value_and_valuenum(df)

    labs_logger.info("Dropping non-numeric / out-of-bounds valuenum rows using labs.csv bounds:")
    df = remove_labs_out_of_bounds(df)

    labs_logger.info("Adding feature_name via itemid mapping from labs.csv")
    df = add_feature_name_by_itemid(df)

    labs_logger.info(f"Writing labs to: {utils.LABS_PATH}")
    df.to_parquet(utils.LABS_PATH, index=False)
    labs_logger.info("Done.")

    return df


if __name__ == "__main__":
    create_all_labs_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    base_labs_df = pd.read_parquet(utils.FIRST_48H_LABS_PATH)
    create_labs_df(base_labs_df)
