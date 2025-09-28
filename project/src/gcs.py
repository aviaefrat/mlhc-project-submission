import os
from pathlib import Path

import duckdb
import pandas as pd

from . import utils

gcs_logger = utils.make_logger("gcs", "gcs.log")


def create_all_gcs_for_filtered_cohort_0_48h(cohort_path: Path = None, db_path: Path = None) -> None:
    """
    Pull GCS from CHARTEVENTS restricted to t in [admittime, admittime+48h),
    for admissions in cohort_path and ITEMIDs in data/gcs.csv.
    Writes directly to Parquet for speed.
    """
    gcs_map = pd.read_csv(utils.GCS_CSV)
    itemids = gcs_map["itemid"].astype("int64").unique().tolist()

    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA threads=%d" % os.cpu_count())
        con.execute("PRAGMA memory_limit='28GB'")

        out_path = utils.FIRST_48H_GCS_PATH.as_posix()

        gcs_logger.info(
            f"Running 0–48h GCS CHARTEVENTS extract -> {utils.FIRST_48H_GCS_PATH}"
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
        gcs_logger.info("Done writing GCS parquet.")

    finally:
        con.close()


def remove_gcs_missing_value_and_valuenum(df: pd.DataFrame) -> pd.DataFrame:

    n_before = len(df)
    value_na = df["value"].isna()
    valuenum_na = df["valuenum"].isna()

    both_na_mask = value_na & valuenum_na
    only_value_na_mask = value_na & ~valuenum_na
    only_valuenum_na_mask = ~value_na & valuenum_na

    n_both = int(both_na_mask.sum())
    n_only_value = int(only_value_na_mask.sum())
    n_only_valuenum = int(only_valuenum_na_mask.sum())

    gcs_logger.info(f"Filtering rows where BOTH value and valuenum are NAs for {n_before:,} rows:")
    if n_both > 0:
        gcs_logger.info(f"Dropping {n_both:,} / {n_before:,} rows with BOTH NA.")
        per_item_drop = (
            df.loc[both_na_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        for itemid, n in per_item_drop.items():
            gcs_logger.info(f"  itemid {int(itemid)}: dropped {int(n):,}")
    else:
        gcs_logger.info("No rows where BOTH `value` and `valuenum` are NA.")

    if n_only_value or n_only_valuenum:
        gcs_logger.info(
            "Kept rows with exactly one field present: "
            f"only `value` NA (valuenum present) = {n_only_value:,}; "
            f"only `valuenum` NA (value present) = {n_only_valuenum:,}."
        )

    out = df.loc[~both_na_mask].copy()
    gcs_logger.info(f"Kept {len(out):,} rows after filtering (from {n_before:,}).")
    return out


def remove_gcs_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    # todo: if we will have time, consider fallback to parsing numeric info from `value` if `valuenum` is NA.
    n0 = len(df)

    # Bounds from CSV
    gcs_map = pd.read_csv(utils.GCS_CSV)
    bounds = (
        gcs_map[["itemid", "min", "max"]]
        .astype({"itemid": "int64"})
        .rename(columns={"min": "min_bound", "max": "max_bound"})
    )

    work = df.copy()
    work["itemid"] = work["itemid"].astype("int64")
    work = work.merge(bounds, on="itemid", how="left")

    # Coerce to numeric
    valuenum_num = pd.to_numeric(work["valuenum"], errors="coerce")
    non_numeric_mask = valuenum_num.isna()
    n_non_numeric = int(non_numeric_mask.sum())
    if n_non_numeric:
        gcs_logger.info(
            f"remove_gcs_out_of_bounds: dropping {n_non_numeric:,} row(s) with non-numeric `valuenum`."
        )
        per_item_non_numeric = (
            work.loc[non_numeric_mask, "itemid"].value_counts().sort_values(ascending=False)
        )
        for itemid, cnt in per_item_non_numeric.items():
            gcs_logger.info(f"  itemid {int(itemid)}: non-numeric valuenum rows dropped = {int(cnt):,}")

    work["valuenum"] = valuenum_num

    # Out-of-bounds (including the limits)
    have_num = work["valuenum"].notna()
    oob_mask = have_num & (
        (work["valuenum"] < work["min_bound"]) | (work["valuenum"] > work["max_bound"])
    )

    # zero-specific logging
    zero_oob_mask = have_num & (work["valuenum"] == 0)
    if int(zero_oob_mask.sum()) > 0:
        per_item_zero = work.loc[zero_oob_mask, "itemid"].value_counts().sort_values(ascending=False)
        gcs_logger.info("Zeros encountered (treated as out-of-bounds) per itemid:")
        for itemid, cnt in per_item_zero.items():
            gcs_logger.info(f"  itemid {int(itemid)}: zeros dropped = {int(cnt):,}")

    n_oob = int(oob_mask.sum())
    if n_oob:
        gcs_logger.info(
            f"remove_gcs_out_of_bounds: dropping {n_oob:,} row(s) out of bounds (allowed inclusive)."
        )
        per_item_oob = work.loc[oob_mask, "itemid"].value_counts().sort_values(ascending=False)
        for itemid, cnt in per_item_oob.items():
            vmin = work.loc[work["itemid"] == itemid, "min_bound"].iloc[0]
            vmax = work.loc[work["itemid"] == itemid, "max_bound"].iloc[0]
            gcs_logger.info(
                f"  itemid {int(itemid)}: out-of-bounds dropped = {int(cnt):,} (allowed [{vmin}, {vmax}])"
            )

    keep_mask = (~non_numeric_mask) & (~oob_mask)
    out = work.loc[keep_mask, df.columns].copy()

    gcs_logger.info(
        f"remove_gcs_out_of_bounds: kept {len(out):,} / {n0:,} rows "
        f"(dropped {n_non_numeric + n_oob:,}: non-numeric={n_non_numeric:,}, out-of-bounds={n_oob:,})."
    )

    return out


def add_feature_name_by_itemid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'feature_name' column by mapping each row's itemid using utils.GCS_CSV.
    """
    gcs_map = pd.read_csv(utils.GCS_CSV).astype({"itemid": "int64"})
    mapping = dict(zip(gcs_map["itemid"].to_numpy(), gcs_map["feature name"].to_numpy()))

    out = df.copy()
    out["itemid"] = out["itemid"].astype("int64")
    out["feature_name"] = out["itemid"].map(mapping).astype("string")

    # Logging
    n_total = len(out)
    n_mapped = int(out["feature_name"].notna().sum())
    n_missing = n_total - n_mapped
    gcs_logger.info(
        f"add_feature_name_by_itemid: mapped feature_name for {n_mapped:,} / {n_total:,} rows "
        f"({n_missing:,} missing)."
    )
    return out


def create_gcs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline:
      1) drop rows where BOTH value and valuenum are NAs
      2) drop non-numeric or out-of-bounds valuenum
      3) add feature_name using GCS.csv
      4) write to utils.GCS_PATH
    """
    gcs_logger.info(f"Filtering BOTH-NA rows for {len(df):,} rows:")
    df = remove_gcs_missing_value_and_valuenum(df)

    gcs_logger.info("Dropping non-numeric / out-of-bounds valuenum rows:")
    df = remove_gcs_out_of_bounds(df)

    gcs_logger.info("Adding feature_name via itemid mapping from gcs.csv")
    df = add_feature_name_by_itemid(df)

    gcs_logger.info(f"Writing GCS to: {utils.GCS_PATH}")
    df.to_parquet(utils.GCS_PATH, index=False)
    gcs_logger.info("Done writing cleaned GCS parquet.")

    new_df = pd.read_parquet(utils.GCS_PATH)
    print(new_df.columns.tolist())
    print(new_df.head(3))
    return df


if __name__ == "__main__":
    create_all_gcs_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    base_gcs_df = pd.read_parquet(utils.FIRST_48H_GCS_PATH)
    create_gcs_df(base_gcs_df)
