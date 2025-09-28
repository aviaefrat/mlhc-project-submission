from pathlib import Path

import pandas as pd
import duckdb

from . import utils

vitals_logger = utils.make_logger("vitals", "vitals.log")


def create_all_vitals_for_filtered_cohort_0_48h(cohort_path: Path = None, db_path: Path = None, ):
    """
    Pull vitals from CHARTEVENTS restricted to t in [admittime, admittime+48h),
    for admissions in cohort_path and ITEMIDs in data/vitals.csv.
    Writes directly to Parquet for speed.
    """
    vitals_map = pd.read_csv(utils.VITALS_CSV)
    itemids = vitals_map["itemid"].astype("int64").unique().tolist()

    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        # give DuckDB more RAM/threads (it was too slow, vitals is big)
        import os
        con.execute("PRAGMA threads=%d" % os.cpu_count())
        con.execute("PRAGMA memory_limit='28GB'")

        out_path = utils.FIRST_48H_VITALS_PATH.as_posix()

        vitals_logger.info(
            f"Running single-pass SQL with 48h filter; writing to {utils.FIRST_48H_VITALS_PATH}"
        )

        sql = f"""
        CREATE OR REPLACE TEMP TABLE filtered_cohort AS
        SELECT
            CAST(subject_id AS INTEGER)   AS subject_id,
            CAST(admission_id AS INTEGER) AS admission_id,
            CAST(admittime AS TIMESTAMP)  AS admittime
        FROM read_parquet('{cohort_path.as_posix()}');

        COPY (
            SELECT
                ce.subject_id,
                ce.admission_id,
                ce.itemid,
                ce.charttime,
                ce.valuenum,
                ce.value,
                ce.valueuom
            FROM (
                SELECT
                    CAST(subject_id AS INTEGER)   AS subject_id,
                    CAST(hadm_id    AS INTEGER)   AS admission_id,
                    CAST(itemid     AS INTEGER)   AS itemid,
                    CAST(charttime  AS TIMESTAMP) AS charttime,
                    CAST(valuenum   AS DOUBLE)    AS valuenum,
                    CAST(value      AS VARCHAR)   AS value,
                    CAST(valueuom   AS VARCHAR)   AS valueuom
                FROM chartevents
                WHERE CAST(itemid AS INTEGER) IN ({",".join(str(x) for x in itemids)})
            ) AS ce
            JOIN filtered_cohort AS m
              ON ce.admission_id = m.admission_id
            WHERE ce.charttime >= m.admittime
              AND ce.charttime <  m.admittime + INTERVAL 48 HOUR
        ) TO '{out_path}' (FORMAT PARQUET);
        """

        con.execute(sql)
        vitals_logger.info("Done writing filtered vitals parquet.")

    finally:
        con.close()


def remove_vitals_missing_value_and_valuenum(df) -> pd.DataFrame:

    n_before = len(df)
    value_na = df["value"].isna()
    valuenum_na = df["valuenum"].isna()

    both_na_mask = value_na & valuenum_na

    # For the logging later - rows that are NA in one field but not the other
    only_value_na_mask = value_na & ~valuenum_na
    only_valuenum_na_mask = ~value_na & valuenum_na

    n_both = int(both_na_mask.sum())
    n_only_value = int(only_value_na_mask.sum())
    n_only_valuenum = int(only_valuenum_na_mask.sum())

    if n_both > 0:
        vitals_logger.info(
            f"Dropping {n_both:,} / {n_before:,} rows where BOTH `value` and `valuenum` are NA."
        )
        per_item_drop = (
            df.loc[both_na_mask, "itemid"]
              .value_counts()
              .sort_values(ascending=False)
        )
        for itemid, n in per_item_drop.items():
            vitals_logger.info(f"  itemid {int(itemid)}: dropped {int(n):,} row(s)")
    else:
        vitals_logger.info("No rows where BOTH `value` and `valuenum` are NA.")

    if n_only_value or n_only_valuenum:
        vitals_logger.info(
            "Kept rows with exactly one field present:"
            f" only `value` NA (but `valuenum` present): {n_only_value:,};"
            f" only `valuenum` NA (but `value` present): {n_only_valuenum:,}."
        )

    df = df.loc[~both_na_mask].copy()
    vitals_logger.info(f"Kept {len(df):,} rows after filtering (from {n_before:,}).")
    return df


def remove_vitals_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # load bounds
    vitals_map = pd.read_csv(utils.VITALS_CSV)
    bounds = (
        vitals_map[["itemid", "min", "max"]]
        .astype({"itemid": "int64"})
        .rename(columns={"min": "min_bound", "max": "max_bound"})
    )

    work = df.copy()
    work["itemid"] = work["itemid"].astype("int64")
    work = work.merge(bounds, on="itemid", how="left")

    # any non-numeric valuenum will be drooped   # todo fallback to `value`?
    valuenum_num = pd.to_numeric(work["valuenum"], errors="coerce")
    non_numeric_mask = valuenum_num.isna()

    n_non_numeric = int(non_numeric_mask.sum())
    if n_non_numeric:
        vitals_logger.info(
            f"remove_vitals_out_of_bounds: dropping {n_non_numeric:,} row(s) with non-numeric `valuenum`."
        )
        per_item_non_numeric = (
            work.loc[non_numeric_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        for itemid, cnt in per_item_non_numeric.items():
            vitals_logger.info(f"  itemid {int(itemid)}: non-numeric valuenum rows dropped = {int(cnt):,}")

    work["valuenum"] = valuenum_num

    # for logging the number of out of bounds
    have_bounds = work["min_bound"].notna() & work["max_bound"].notna()
    oob_mask = have_bounds & work["valuenum"].notna() & (
        (work["valuenum"] < work["min_bound"]) | (work["valuenum"] > work["max_bound"])
    )
    n_oob = int(oob_mask.sum())
    if n_oob:
        vitals_logger.info(
            f"remove_vitals_out_of_bounds: dropping {n_oob:,} row(s) out of bounds (exclusive of equals)."
        )
        per_item_oob = (
            work.loc[oob_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        for itemid, cnt in per_item_oob.items():
            vmin = work.loc[work["itemid"] == itemid, "min_bound"].iloc[0]
            vmax = work.loc[work["itemid"] == itemid, "max_bound"].iloc[0]
            vitals_logger.info(
                f"  itemid {int(itemid)}: out-of-bounds dropped = {int(cnt):,} (allowed [{vmin}, {vmax}])"
            )

    # for logging when no bounds were found (shouldn't happen but just in case we manually add more vitals and don't copy the information correctly)
    no_bounds_mask = ~have_bounds
    n_no_bounds = int(no_bounds_mask.sum())
    if n_no_bounds:
        n_itemids_missing = work.loc[no_bounds_mask, "itemid"].nunique()
        vitals_logger.info(
            f"remove_vitals_out_of_bounds: {n_no_bounds:,} row(s) across {n_itemids_missing:,} itemid(s) "
            f"were skipped (no bounds found in {utils.VITALS_CSV}). Rows kept."
        )

    keep_mask = (~non_numeric_mask) & (~oob_mask)
    out = work.loc[keep_mask, df.columns].copy()

    vitals_logger.info(
        f"remove_vitals_out_of_bounds: kept {len(out):,} / {n0:,} rows "
        f"(dropped {n_non_numeric + n_oob:,}: non-numeric={n_non_numeric:,}, out-of-bounds={n_oob:,})."
    )

    return out


def add_feature_name_by_itemid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'feature_name' column by mapping each row's itemid using utils.VITALS_CSV.
    """
    vitals_map = pd.read_csv(utils.VITALS_CSV)

    # ensure required columns exist
    required_cols = {"itemid", "feature name"}
    missing = required_cols - set(vitals_map.columns)
    if missing:
        raise ValueError(f"{utils.VITALS_CSV} missing required columns: {sorted(missing)}")

    # datatypes
    vitals_map = vitals_map.astype({"itemid": "int64"})
    df = df.copy()
    df["itemid"] = df["itemid"].astype("int64")

    # make sure each itemid has only one feature name:
    name_counts = (
        vitals_map.groupby("itemid", as_index=False)["feature name"]
        .nunique(dropna=False)
        .rename(columns={"feature name": "n_names"})
    )
    dup_bad = name_counts[name_counts["n_names"] > 1]
    if not dup_bad.empty:
        vitals_logger.error(
            "Detected itemid(s) with multiple distinct 'feature name' values in vitals.csv. aborting."
        )
        for r in dup_bad.itertuples(index=False):
            vitals_logger.error(f"  itemid={int(r.itemid)} has {int(r.n_names)} distinct names")
        raise ValueError("Non-unique itemid->feature name mapping in vitals.csv")

    mapping_df = vitals_map.loc[:, ["itemid", "feature name"]].drop_duplicates()
    mapping = dict(zip(mapping_df["itemid"].to_numpy(), mapping_df["feature name"].to_numpy()))

    df["feature_name"] = df["itemid"].map(mapping)
    df["feature_name"] = df["feature_name"].astype("string")

    # Logging
    n_total = len(df)
    n_mapped = int(df["feature_name"].notna().sum())
    n_missing = n_total - n_mapped
    vitals_logger.info(
        f"add_feature_name_by_itemid: mapped feature_name for {n_mapped:,} / {n_total:,} rows "
        f"({n_missing:,} missing)."
    )

    # Per-itemid counts for missing mappings
    miss_mask = df["feature_name"].isna()
    if miss_mask.any():
        per_item_missing = (
            df.loc[miss_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        vitals_logger.warning(
            f"add_feature_name_by_itemid: {int(miss_mask.sum()):,} rows missing feature_name "
            f"({len(per_item_missing):,} itemid(s) not found in {utils.VITALS_CSV})."
        )
        for itemid, cnt in per_item_missing.items():
            vitals_logger.warning(f"  itemid {int(itemid)}: {int(cnt):,} row(s) without mapping")

    return df


def _normalize_temperature_itemids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Fahrenheit itemid 223761 to Celsius and alias TempF -> TempC.
    Leave itemid 676 (Celsius) unchanged.
    """
    df = df.copy()
    # Detect TempF rows
    is_f = df["itemid"] == 223761
    n_f = int(is_f.sum())
    if n_f:
        df.loc[is_f & df["valuenum"].notna(), "valuenum"] = (
            (df.loc[is_f & df["valuenum"].notna(), "valuenum"] - 32.0) * (5.0 / 9.0)
        )
        df.loc[is_f, "feature_name"] = "TempC"
        df.loc[is_f, "valueuom"] = "C"
    vitals_logger.info(f"Temperature normalization: converted {n_f:,} TempF rows to TempC.")
    return df


def create_vitals_df(df: pd.DataFrame) -> pd.DataFrame:
    vitals_logger.info(f"Removing rows where both value and valuenum are NAs for {len(df):,} rows:")
    df = remove_vitals_missing_value_and_valuenum(df)

    vitals_logger.info("Dropping non-numeric / out-of-bounds valuenum rows:")
    df = remove_vitals_out_of_bounds(df)

    vitals_logger.info("Adding feature_name via itemid mapping from vitals.csv")
    df = add_feature_name_by_itemid(df)

    vitals_logger.info("Normalizing temperature: itemid 223761 (F) -> TempC")
    df = _normalize_temperature_itemids(df)

    vitals_logger.info(f"Writing vitals to: {utils.VITALS_PATH}")
    df.to_parquet(utils.VITALS_PATH, index=False)
    vitals_logger.info("Done.")

    new_df = pd.read_parquet(utils.VITALS_PATH)
    print(new_df.columns.tolist())
    print(new_df.head(3))
    return df


if __name__ == "__main__":
    create_all_vitals_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    base_vitals_df = pd.read_parquet(utils.FIRST_48H_VITALS_PATH)
    create_vitals_df(base_vitals_df)


# todo (along with ones already in the code:
#  canonize/normalize valueuom, and remove values that don't make sense according to the uom (irrespective of the expected min/max values)
#  handle valuenums that the only reason they fail is because they include the uom.