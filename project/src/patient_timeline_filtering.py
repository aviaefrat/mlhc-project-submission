from pathlib import Path


import duckdb
import pandas as pd

from . import utils

first_adm_logger = utils.make_logger("first_admissions", "first_admissions.log")
min54h_logger = utils.make_logger("min_54h_first_admissions", "min_54h_first_admissions.log")
age_filter_logger = utils.make_logger("age_filter", "age_filter.log")
second_admissions_logger = utils.make_logger("second_admissions", "second_admissions.log")


def get_subject_ids_from_cohort_csv_path(cohort_csv_path: Path | str = None):
    cohort_csv_path = cohort_csv_path or utils.INITIAL_COHORT_CSV
    subject_ids = pd.read_csv(cohort_csv_path)["subject_id"].tolist()
    return subject_ids

def create_first_admissions_df(subject_ids: list, db_path: Path = None, overwrite: bool = True) -> pd.DataFrame:
    """
    Create a dataframe with the first chronological admission per subject.
    Output columns:
        - subject_id (int32)
        - admission_id (int32)  # MIMIC hadm_id

    Will save the file to: utils.FIRST_ADMISSIONS_PATH
    """
    if utils.FIRST_ADMISSIONS_PATH.exists() and not overwrite:
        first_adm_logger.info(
            f"First admissions file already exists. Loading it from {utils.FIRST_ADMISSIONS_PATH}"
        )
        return pd.read_parquet(utils.FIRST_ADMISSIONS_PATH)

    first_adm_logger.info(f"Initial cohort subjects: {len(subject_ids):,}")

    sql = """
          SELECT CAST(subject_id AS INTEGER)  AS subject_id,
                 CAST(hadm_id AS INTEGER)     AS admission_id,
                 CAST(admittime AS TIMESTAMP) AS admittime
          FROM admissions
          WHERE CAST(subject_id AS INTEGER) IN ?
          """

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        admissions = con.execute(sql, [subject_ids]).fetchdf()
    finally:
        con.close()

    # Ensure expected dtypes
    admissions = admissions.astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")

    # drop rows with null admittime (log count if any).
    na_admission = admissions["admittime"].isna().sum()
    if na_admission:
        first_adm_logger.info(f"Dropping {na_admission} admission rows with null admittime")
        admissions = admissions.dropna(subset=["admittime"])

    # Sort so earliest admission per subject is first. tie-break by hadm_id.
    admissions = admissions.sort_values(["subject_id", "admittime", "admission_id"], kind="mergesort")

    # take the first row per subject_id after sorting.
    first_admissions_df = (
        admissions.drop_duplicates(subset=["subject_id"], keep="first")
        .loc[:, ["subject_id", "admission_id"]]
        .reset_index(drop=True)
    )

    # teport cohort subjects with no admissions (if any).
    found_subject_ids = set(first_admissions_df["subject_id"].tolist())
    subjects_with_no_admission = [sid for sid in subject_ids if sid not in found_subject_ids]
    if subjects_with_no_admission:
        first_adm_logger.info(
            f"{len(subjects_with_no_admission):,} cohort subject(s) have no rows in admissions."
        )
        for subject_id in subjects_with_no_admission:
            first_adm_logger.warning(f"No admission found for subject_id={subject_id}")
    else:
        first_adm_logger.info("All cohort subjects have at least one admission.")

    # save
    first_admissions_df.to_parquet(utils.FIRST_ADMISSIONS_PATH, index=False)
    first_adm_logger.info(
        f"Wrote first admissions to {utils.FIRST_ADMISSIONS_PATH} (rows={len(first_admissions_df):,})."
    )
    return first_admissions_df


def create_min_54h_first_admissions_df(db_path: Path = None, overwrite: bool = True) -> pd.DataFrame:
    """
    Keep only first admissions whose hospitalization lasted at least 54 hours.
    Output parquet columns:
      - subject_id (int32)
      - admission_id (int32)
      - admittime (TIMESTAMP)
      - dischtime (TIMESTAMP)
    """
    if utils.MIN_54H_FIRST_ADMISSIONS_PATH.exists() and not overwrite:
        min54h_logger.info(f"First≥54h file already exists. Loading it from {utils.MIN_54H_FIRST_ADMISSIONS_PATH}")
        return pd.read_parquet(utils.MIN_54H_FIRST_ADMISSIONS_PATH)

    # Load first admissions (subject_id, admission_id)
    first_admissions_df = pd.read_parquet(utils.FIRST_ADMISSIONS_PATH).astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )
    min54h_logger.info(f"Loaded first admissions: {len(first_admissions_df):,} rows")

    # get admittime and dischtime for those hadm_ids from DuckDB
    hadm_ids = first_admissions_df["admission_id"].tolist()
    sql = """
          SELECT CAST(hadm_id AS INTEGER)     AS admission_id,
                 CAST(admittime AS TIMESTAMP) AS admittime,
                 CAST(dischtime AS TIMESTAMP) AS dischtime
          FROM admissions
          WHERE CAST(hadm_id AS INTEGER) IN ?
          """

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        admission_times = con.execute(sql, [hadm_ids]).fetchdf()
    finally:
        con.close()

    # Normalize dtypes/timestamps
    admission_times = admission_times.astype({"admission_id": utils.ID_DTYPE})
    admission_times["admittime"] = pd.to_datetime(admission_times["admittime"], errors="coerce")
    admission_times["dischtime"] = pd.to_datetime(admission_times["dischtime"], errors="coerce")

    admit_and_disch_times = first_admissions_df.merge(admission_times, on="admission_id", how="left")

    # Log rows with missing admittime and/or dischtime, then drop them
    na_mask = admit_and_disch_times["admittime"].isna() | admit_and_disch_times["dischtime"].isna()
    if na_mask.any():
        rows_with_na = admit_and_disch_times.loc[na_mask, ["subject_id", "admission_id", "admittime", "dischtime"]]
        min54h_logger.info(
            f"Missing times for {len(rows_with_na):,} admission(s) (each listed below; will be excluded)"
        )
        for row_with_na in rows_with_na.itertuples(index=False):
            missing = []
            if pd.isna(row_with_na.admittime):
                missing.append("admittime")
            if pd.isna(row_with_na.dischtime):
                missing.append("dischtime")
            min54h_logger.warning(
                f"subject_id={int(row_with_na.subject_id)}, admission_id={int(row_with_na.admission_id)} "
                f"missing: {', '.join(missing)}"
            )
        admit_and_disch_times = admit_and_disch_times.loc[~na_mask].copy()
    else:
        min54h_logger.info("No missing admittime/dischtime.")

    # Duration (hours) and sanity checks
    admit_and_disch_times["duration_hours"] = (
            (admit_and_disch_times["dischtime"] - admit_and_disch_times["admittime"])
    ) / pd.Timedelta(hours=1)

    nonpositive_duration_mask = admit_and_disch_times["duration_hours"] <= 0
    if nonpositive_duration_mask.any():
        bad_rows = admit_and_disch_times.loc[
            nonpositive_duration_mask,
            ["subject_id", "admission_id", "admittime", "dischtime", "duration_hours"],
        ]
        min54h_logger.info(
            f"Non-positive durations for {len(bad_rows):,} admission(s) (each listed below; will be excluded)"
        )
        for row in bad_rows.itertuples(index=False):
            admittime = row.admittime.isoformat() if pd.notna(row.admittime) else "NaT"
            dischtime = row.dischtime.isoformat() if pd.notna(row.dischtime) else "NaT"
            min54h_logger.warning(
                f"subject_id={int(row.subject_id)}, admission_id={int(row.admission_id)}, "
                f"admittime={admittime}, dischtime={dischtime}, duration_hours={float(row.duration_hours):.3f}"
            )
        admit_and_disch_times = admit_and_disch_times.loc[~nonpositive_duration_mask].copy()
    else:
        min54h_logger.info("No non-positive durations.")

    # todo if we have time, also check for unreasonable long durations (like tens of years etc., we can look more at the mimic versions w.r.t dates to get a reasonable estimate for what are durations that are ~100% unreasonable)

    # Keep stays that are at least 54 hours.
    kept_df = (
        admit_and_disch_times.loc[
            admit_and_disch_times["duration_hours"] >= 54,
            ["subject_id", "admission_id", "admittime", "dischtime"],
        ]
        .astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})
        .reset_index(drop=True)
    )

    # make sure types are correct
    assert kept_df["subject_id"].dtype == utils.ID_DTYPE
    assert kept_df["admission_id"].dtype == utils.ID_DTYPE
    assert str(kept_df["admittime"].dtype).startswith("datetime64[")
    assert str(kept_df["dischtime"].dtype).startswith("datetime64[")

    dropped = len(admit_and_disch_times) - len(kept_df)
    min54h_logger.info(
        f"Kept {len(kept_df):,} / {len(admit_and_disch_times):,} admissions (dropped {dropped:,} for <54h)."
    )
    # save
    kept_df.to_parquet(utils.MIN_54H_FIRST_ADMISSIONS_PATH, index=False)
    min54h_logger.info(f"Wrote ≥54h first admissions to {utils.MIN_54H_FIRST_ADMISSIONS_PATH}")
    return kept_df


def create_min_54h_first_admissions_age_filtered_df(db_path: Path = None, overwrite: bool = True) -> pd.DataFrame:
    """
    Filter MIN_54H_FIRST_ADMISSIONS_PATH to patients whose age at admittime is in [18, 90).

    Drops rows with: NA age (missing dob), negative age, age in [0,17], or age >=90.
    Logs aggregate value counts of drop reasons.
    Saves to utils.MIN_54H_FIRST_ADMISSIONS_FILTERED_AGE_PATH.
    """
    path = utils.MIN_54H_FIRST_ADMISSIONS_PATH
    out_path = utils.MIN_54H_FIRST_ADMISSIONS_FILTERED_AGE_PATH

    if out_path.exists() and not overwrite:
        age_filter_logger.info(f"[age_filter] Output exists and overwrite=False. Loading {out_path}")
        return pd.read_parquet(out_path)

    if not path.exists():
        raise FileNotFoundError(f"Required input parquet not found: {path}")

    # load base cohort
    df = pd.read_parquet(path).astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )
    age_filter_logger.info(f"[age_filter] Loaded {len(df):,} rows from {path}")

    # get dob for these subjects
    subjects = df["subject_id"].astype("int64").unique().tolist()
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
            SELECT
                CAST(subject_id AS INTEGER)   AS subject_id,
                CAST(dob        AS TIMESTAMP) AS dob
            FROM patients
            WHERE CAST(subject_id AS INTEGER) IN ?
        """
        pats = con.execute(sql, [subjects]).fetchdf()
    finally:
        con.close()

    pats = pats.astype({"subject_id": utils.ID_DTYPE})
    pats["dob"] = pd.to_datetime(pats["dob"], errors="coerce")

    work = df.merge(pats, on="subject_id", how="left", validate="many_to_one").copy()
    work["admittime"] = pd.to_datetime(work["admittime"], errors="coerce")

    # floor age in years at admittime
    admit_md = work["admittime"].dt.month * 100 + work["admittime"].dt.day
    dob_md   = work["dob"].dt.month * 100 + work["dob"].dt.day
    year_diff = work["admittime"].dt.year - work["dob"].dt.year
    adjust = (admit_md < dob_md).astype("Int64")  # <NA> propagates where dob is NA
    work["age"] = (year_diff - adjust).astype("Int64")

    drop_reason = pd.Series(pd.NA, index=work.index, dtype="string")

    # 1. NA age (missing dob)
    drop_reason[work["age"].isna()] = "na_age"

    # 2. Negative age
    remaining = drop_reason.isna()
    drop_reason[remaining & (work["age"] < 0)] = "negative_age"

    # 3. Under 18
    remaining = drop_reason.isna()
    drop_reason[remaining & work["age"].between(0, 17)] = "under18"

    # 4. 90 and older
    remaining = drop_reason.isna()
    drop_reason[remaining & (work["age"] >= 90)] = "age90plus"

    to_drop = drop_reason.notna()
    n_drop = int(to_drop.sum())
    keep = work.loc[~to_drop].copy().astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )

    if n_drop:
        reasons = ["na_age", "negative_age", "under18", "age90plus"]
        counts = {r: int((drop_reason.loc[to_drop] == r).sum()) for r in reasons}
        age_filter_logger.info(
            f"[age_filter] Dropped {n_drop:,} / {len(work):,} rows. Reason counts: {counts}"
        )
        if sum(counts.values()) != n_drop:
            age_filter_logger.warning(
                f"[age_filter] Sanity check: reason sum {sum(counts.values())} != n_drop {n_drop}"
            )
    else:
        age_filter_logger.info("[age_filter] No rows dropped; all ages within [18,90).")

    # Keep only the cohort columns (no dob/age)
    keep_cols = ["subject_id", "admission_id", "admittime", "dischtime"]
    keep = keep.loc[:, keep_cols].copy().astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(out_path, index=False)
    age_filter_logger.info(
        f"[age_filter] Wrote filtered cohort to {out_path} (rows={len(keep):,}, dropped={n_drop:,})."
    )
    return keep


def create_second_admission_df(db_path: Path = None, overwrite: bool = True) -> pd.DataFrame:
    """
    For every subject_id in utils.FILTERED_COHORT, get the second (chronological) hospital
    admission from ADMISSIONS. If a subject has no second admission, return NA values for:
      - second_admission_id, second_admittime, second_dischtime.

    Then log (but don't drop) subjects whose second admission has a non-positive duration (dischtime <= admittime).
    Also log rows where second_admittime and/or second_dischtime are NA, and report the total count.

    Output schema:
      - subject_id (Int32)
      - second_admission_id (Int32)         # NA if no second admission
      - second_admittime (datetime64[ns])   # NA if no second admission
      - second_dischtime (datetime64[ns])   # NA if no second admission

    Saves to: data/pipeline/1-3-helper-second_admissions.parquet
    """
    if utils.SECOND_ADMISSIONS_PATH.exists() and not overwrite:
        second_admissions_logger.info(
            f"Second admissions file already exists. Loading it from {utils.SECOND_ADMISSIONS_PATH}"
        )
        return pd.read_parquet(utils.SECOND_ADMISSIONS_PATH)

    # load our filtered cohort
    base = pd.read_parquet(utils.FILTERED_COHORT_PATH).astype({"subject_id": utils.ID_DTYPE})
    filtered_cohort_subject_ids = base["subject_id"].astype("int64").unique().tolist()
    n_subjects = len(filtered_cohort_subject_ids)
    second_admissions_logger.info(f"Loaded base (filtered) cohort: {n_subjects:,} unique subject(s)")

    # get all admissions for those subjects and pick the 2nd by (admittime, hadm_id)
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
        WITH cand AS (
            SELECT
                CAST(subject_id AS INTEGER)   AS subject_id,
                CAST(hadm_id    AS INTEGER)   AS admission_id,
                CAST(admittime  AS TIMESTAMP) AS admittime,
                CAST(dischtime  AS TIMESTAMP) AS dischtime
            FROM admissions
            WHERE CAST(subject_id AS INTEGER) IN ?
        ),
        ranked AS (
            SELECT subject_id, admission_id, admittime, dischtime,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id
                    ORDER BY admittime NULLS LAST, admission_id
                ) AS rn
            FROM cand
        )
        SELECT
            subject_id,
            admission_id      AS second_admission_id,
            admittime         AS second_admittime,
            dischtime         AS second_dischtime
        FROM ranked
        WHERE rn = 2
        """
        second_admissions = con.execute(sql, [filtered_cohort_subject_ids]).fetchdf()
    finally:
        con.close()

    out = (
        pd.DataFrame({"subject_id": pd.Series(filtered_cohort_subject_ids, dtype="int64")})
        .merge(second_admissions, on="subject_id", how="left", validate="one_to_one")
    )

    # Dtypes:
    out = out.astype({"subject_id": utils.ID_DTYPE})
    out["second_admission_id"] = out["second_admission_id"].astype("Int32")
    out["second_admittime"] = pd.to_datetime(out["second_admittime"], errors="coerce")
    out["second_dischtime"] = pd.to_datetime(out["second_dischtime"], errors="coerce")

    # Log subjects with no second admission
    no_second_mask = out["second_admission_id"].isna()
    n_no_second = int(no_second_mask.sum())
    second_admissions_logger.info(
        f"Subjects without a second admission: {n_no_second:,} / {len(out):,}"
    )

    # Log rows with NA in either second_admittime or second_dischtime, but only if there is a second_admission_id
    na_time_mask = (
        out["second_admission_id"].notna()
        & (out["second_admittime"].isna() | out["second_dischtime"].isna())
    )
    n_na_time = int(na_time_mask.sum())
    if n_na_time:
        second_admissions_logger.info(
            f"Rows with NA in second_admittime and/or second_dischtime "
            f"(where second_admission_id is present): {n_na_time:,}"
        )
        for row in out.loc[
            na_time_mask, ["subject_id", "second_admission_id", "second_admittime", "second_dischtime"]
        ].itertuples(index=False):
            second_admissions_logger.warning(
                f"NA time(s) for second admission -> subject_id={int(row.subject_id)}, "
                f"second_admission_id={row.second_admission_id}, "
                f"second_admittime={row.second_admittime}, second_dischtime={row.second_dischtime}"
            )
    else:
        second_admissions_logger.info("No NA times detected among rows with a second_admission_id.")

    # Compute duration only for rows with all fields present. we log non-positive durations.
    # Also create the boolean column `nonpositive_duration`
    out["nonpositive_duration"] = False  # default
    have_all = (
        out["second_admission_id"].notna()
        & out["second_admittime"].notna()
        & out["second_dischtime"].notna()
    )
    if have_all.any():
        dur_hours = (
            (out.loc[have_all, "second_dischtime"] - out.loc[have_all, "second_admittime"])
            / pd.Timedelta(hours=1)
        )
        nonpos_idx = dur_hours[dur_hours <= 0].index
        # mark the boolean column (others remain False)
        out.loc[nonpos_idx, "nonpositive_duration"] = True

        n_nonpos = int(len(nonpos_idx))
        if n_nonpos:
            second_admissions_logger.warning(
                f"Found {n_nonpos:,} subject(s) with NON-POSITIVE second-admission duration "
                f"(logged only; not dropped)."
            )
            for row in out.loc[
                nonpos_idx, ["subject_id", "second_admission_id", "second_admittime", "second_dischtime"]
            ].itertuples(index=False):
                second_admissions_logger.warning(
                    f"NON-POS-DUR second admission -> subject_id={int(row.subject_id)}, "
                    f"second_admission_id={row.second_admission_id}, "
                    f"second_admittime={row.second_admittime}, second_dischtime={row.second_dischtime}"
                )
        else:
            second_admissions_logger.info(
                "No non-positive durations detected among rows with all fields present."
            )
    else:
        second_admissions_logger.info(
            "No rows had second_admission_id and both times present; skipping duration checks."
        )

    # save
    utils.SECOND_ADMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(utils.SECOND_ADMISSIONS_PATH, index=False)
    second_admissions_logger.info(
        f"Wrote second admissions to {utils.SECOND_ADMISSIONS_PATH} "
        f"(rows={len(out):,}; without second admission={n_no_second:,}; "
        f"with NA time(s)={n_na_time:,}; nonpositive_duration=True={int(out['nonpositive_duration'].sum()):,})."
    )

    return out


if __name__ == "__main__":
    initial_subject_ids = get_subject_ids_from_cohort_csv_path(cohort_csv_path=utils.INITIAL_COHORT_CSV)
    create_first_admissions_df(subject_ids=initial_subject_ids, db_path=utils.DB_PATH)
    create_min_54h_first_admissions_df(db_path=utils.DB_PATH)
    create_min_54h_first_admissions_age_filtered_df(db_path=utils.DB_PATH)
    create_second_admission_df(db_path=utils.DB_PATH)
