from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

from . import utils

LOG = utils.make_logger("noteevents", "noteevents.log")


def create_all_noteevents_for_filtered_cohort(cohort_path: Path = None, db_path: Path = None, overwrite: bool = True) -> pd.DataFrame:
    out_path: Path = utils.FIRST_48H_NOTEEVENTS_PATH

    if out_path.exists() and not overwrite:
        LOG.info(f"File exists, not overwriting: {out_path}")
        return pd.read_parquet(out_path)

    # Load cohort with first-admission info
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    cohort = pd.read_parquet(cohort_path)[["subject_id", "admission_id", "admittime"]].copy()
    # enforce dtypes
    cohort["subject_id"] = cohort["subject_id"].astype(utils.ID_DTYPE)
    cohort["admission_id"] = cohort["admission_id"].astype(utils.ID_DTYPE)
    # ensure datetime
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], utc=False)

    n_admissions = len(cohort)
    LOG.info(f"Loaded cohort: {n_admissions} (subject_id, admission_id) pairs")

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        con.register("cohort", cohort)

        # NOTE: DATE-only fallback up to date(admittime + 1 days)
        sql = r"""
             SELECT
                c.subject_id::INTEGER      AS subject_id,
                c.admission_id::INTEGER    AS admission_id,
                ne.category                AS category,
                ne.description             AS description,
                CAST(ne.charttime AS TIMESTAMP) AS charttime,
                CAST(ne.chartdate AS DATE)      AS chartdate,
                CAST(ne.iserror AS INTEGER)     AS iserror,
                ne.text                    AS text,
                CASE
                    WHEN ne.charttime IS NULL AND ne.chartdate IS NOT NULL THEN TRUE
                    ELSE FALSE
                END AS kept_via_chartdate_fallback
            FROM cohort AS c
            JOIN noteevents AS ne
              ON CAST(ne.subject_id AS INTEGER) = c.subject_id
             AND CAST(ne.hadm_id    AS INTEGER) = c.admission_id
            WHERE COALESCE(CAST(ne.iserror AS INTEGER), 0) <> 1
              AND (
                    (ne.charttime IS NOT NULL
                     AND CAST(ne.charttime AS TIMESTAMP) >= c.admittime
                     AND CAST(ne.charttime AS TIMESTAMP) <  c.admittime + INTERVAL 48 HOUR)
                   OR
                    (ne.charttime IS NULL AND ne.chartdate IS NOT NULL
                     AND CAST(ne.chartdate AS DATE) >= CAST(c.admittime AS DATE)
                     AND CAST(ne.chartdate AS DATE) <= CAST(c.admittime + INTERVAL 1 DAY AS DATE))
                  )
        """
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    # Attach admittime to compute hours_since_admit, then drop it
    df = df.merge(cohort, on=["subject_id", "admission_id"], how="left")
    # Normalize dtypes
    df["subject_id"] = df["subject_id"].astype(utils.ID_DTYPE)
    df["admission_id"] = df["admission_id"].astype(utils.ID_DTYPE)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce", utc=False)
    # Compute hours_since_admit only when charttime exists
    delta = (df["charttime"] - df["admittime"])
    with np.errstate(invalid="ignore"):
        hours = delta.dt.total_seconds() / 3600.0
    df["hours_since_admit"] = hours.where(df["charttime"].notna(), np.nan)

    # Reorder/select columns
    df = df[
        [
            "subject_id",
            "admission_id",
            "category",
            "description",
            "charttime",
            "chartdate",
            "iserror",
            "text",
            "kept_via_chartdate_fallback",
            "hours_since_admit",
        ]
    ]

    # Logging
    total_notes = len(df)
    kept_via_date = int(df["kept_via_chartdate_fallback"].sum())
    kept_via_time = total_notes - kept_via_date

    admit_with_note = df[["subject_id", "admission_id"]].drop_duplicates().shape[0]

    # Per-category counts
    cat_counts = df["category"].value_counts(dropna=False)
    cat_log = "\n".join(f"  {k}: {v}" for k, v in cat_counts.items())

    time_mask = ~df["kept_via_chartdate_fallback"]
    if total_notes > 0 and time_mask.any():
        hs = df.loc[time_mask, "hours_since_admit"]
        LOG.info(
            "Kept via charttime: %d notes | min_hours=%.2f, p50=%.2f, p90=%.2f, max_hours=%.2f",
            kept_via_time,
            float(hs.min()),
            float(hs.median()),
            float(hs.quantile(0.90)),
            float(hs.max()),
        )
    else:
        LOG.info("Kept via charttime: %d notes", kept_via_time)

    LOG.info("Kept via chartdate fallback: %d notes", kept_via_date)
    LOG.info("Total kept notes: %d", total_notes)
    LOG.info("Admissions with ≥1 kept note: %d / %d", admit_with_note, n_admissions)
    LOG.info("Per-category counts (kept):\n%s", cat_log)


    # drop helper/internal columns before save
    df_to_save = df.drop(columns=["iserror", "kept_via_chartdate_fallback"])
    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_parquet(out_path, compression="snappy", index=False)
    LOG.info(f"Wrote NOTEEVENTS (first 48h) to: {out_path} [rows={len(df_to_save)}]")
    return df_to_save


if __name__ == "__main__":
    create_all_noteevents_for_filtered_cohort(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
