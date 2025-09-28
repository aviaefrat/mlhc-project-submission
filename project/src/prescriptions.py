from pathlib import Path

import duckdb
import pandas as pd

from . import utils

LOG = utils.make_logger("prescriptions", "prescriptions.log")


def _std_txt(s: pd.Series) -> pd.Series:
    """Uppercase, strip, collapse internal whitespace. keep NA as NA."""
    out = pd.Series(s, dtype="string")
    out = out.str.strip().str.upper()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def _std_unit(s: pd.Series) -> pd.Series:
    """remove dots/spaces, uppercase."""
    out = pd.Series(s, dtype="string")
    out = out.str.upper().str.replace(r"[.\s]", "", regex=True)
    return out


def extract_prescriptions_first_48h(cohort_path: Path = None, db_path: Path = None, overwrite: bool = True) -> pd.DataFrame:
    """
    Query PRESCRIPTIONS for the (first) admissions in cohort_path and keep rows whose
    [STARTDATE, ENDDATE) overlaps [admittime, admittime+48h).
    """
    if utils.FIRST_48H_PRESCRIPTIONS_PATH.exists() and not overwrite:
        LOG.info(f"Loading existing {utils.FIRST_48H_PRESCRIPTIONS_PATH}")
        return pd.read_parquet(utils.FIRST_48H_PRESCRIPTIONS_PATH)

    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    cohort = pd.read_parquet(cohort_path).astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )
    if "admittime" not in cohort.columns:
        raise ValueError("cohort_path must include 'admittime' for windowing.")
    LOG.info(f"Cohort rows: {len(cohort):,}")

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE cohort AS
            SELECT
              CAST(subject_id  AS INTEGER)   AS subject_id,
              CAST(admission_id AS INTEGER)  AS admission_id,
              CAST(admittime   AS TIMESTAMP) AS admittime
            FROM read_parquet(?)
        """, [cohort_path.as_posix()])

        sql = """
        WITH base AS (
          SELECT
            p.SUBJECT_ID::INTEGER   AS subject_id,
            p.HADM_ID::INTEGER      AS admission_id,
            p.STARTDATE::TIMESTAMP  AS startdate,
            p.ENDDATE::TIMESTAMP    AS enddate,
            p.DRUG_TYPE::VARCHAR    AS drug_type,
            p.DRUG::VARCHAR         AS drug,
            p.DRUG_NAME_POE::VARCHAR      AS drug_name_poe,
            p.DRUG_NAME_GENERIC::VARCHAR  AS drug_name_generic,
            p.FORMULARY_DRUG_CD::VARCHAR  AS formulary_drug_cd,
            p.GSN::VARCHAR          AS gsn,
            p.NDC::VARCHAR          AS ndc,
            p.PROD_STRENGTH::VARCHAR AS prod_strength,
            p.DOSE_VAL_RX::VARCHAR   AS dose_val_rx,
            p.DOSE_UNIT_RX::VARCHAR  AS dose_unit_rx,
            p.FORM_VAL_DISP::VARCHAR AS form_val_disp,
            p.FORM_UNIT_DISP::VARCHAR AS form_unit_disp,
            p.ROUTE::VARCHAR         AS route
          FROM prescriptions p
          INNER JOIN cohort c
            ON p.HADM_ID::INTEGER = c.admission_id
        ),
        win AS (
          SELECT
            b.*,
            c.admittime                            AS window_start,
            c.admittime + INTERVAL 48 HOUR         AS window_end
          FROM base b
          INNER JOIN cohort c
            ON b.admission_id = c.admission_id
        ),
        overlapped AS (
          SELECT
            *,
            GREATEST(startdate, window_start) AS overlap_start,
            LEAST(enddate,   window_end)      AS overlap_end
          FROM win
        )
        SELECT
          *,
          CASE
            WHEN overlap_start IS NULL OR overlap_end IS NULL THEN 0.0
            WHEN overlap_end <= overlap_start THEN 0.0
            ELSE CAST(date_diff('minute', overlap_start, overlap_end) AS DOUBLE) / 60.0
          END AS overlap_hours_0_48h
        FROM overlapped
        WHERE
          overlap_start IS NOT NULL
          AND overlap_end   IS NOT NULL
          AND overlap_end > overlap_start
        """
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    # basic type cleanup
    for col in ["startdate", "enddate", "window_start", "window_end", "overlap_start", "overlap_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["overlap_hours_0_48h"] = pd.to_numeric(df["overlap_hours_0_48h"], errors="coerce")

    LOG.info(f"Extracted overlapping prescriptions rows: {len(df):,}")
    df.to_parquet(utils.FIRST_48H_PRESCRIPTIONS_PATH, index=False)
    LOG.info(f"Wrote {utils.FIRST_48H_PRESCRIPTIONS_PATH}")
    return df


if __name__ == "__main__":
    extract_prescriptions_first_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
