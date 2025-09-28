from pathlib import Path

import duckdb
import pandas as pd

from . import utils

labels_logger = utils.make_logger("labels", "labels.log")


def create_mortality_df(cohort_path: Path, db_path: Path = None) -> pd.DataFrame:
    """
    Build mortality labels for cohort_path.

    Output parquet: utils.MORTALITY_LABEL with columns:
      - subject_id   (int32)
      - admission_id (int32)
      - mortality    (boolean)  # death during admission OR within 30 days post-discharge
    """
    # Load cohort (subject_id, admission_id, admittime, dischtime)
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    base = pd.read_parquet(cohort_path).astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )
    labels_logger.info(f"[mortality] Loaded base cohort: {len(base):,} rows from {cohort_path}")

    # get HOSPITAL_EXPIRE_FLAG for those hadm_ids
    hadm_ids = base["admission_id"].astype("int64").unique().tolist()

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql_adm = """
            SELECT
                CAST(hadm_id AS INTEGER)                AS admission_id,
                CAST(hospital_expire_flag AS INTEGER)   AS hospital_expire_flag
            FROM admissions
            WHERE CAST(hadm_id AS INTEGER) IN ?
        """
        adm_flags = con.execute(sql_adm, [hadm_ids]).fetchdf()

        # get DOD for those subject_ids
        subject_ids = base["subject_id"].astype("int64").unique().tolist()
        sql_pat = """
            SELECT
                CAST(subject_id AS INTEGER)   AS subject_id,
                CAST(dod        AS TIMESTAMP) AS dod
            FROM patients
            WHERE CAST(subject_id AS INTEGER) IN ?
        """
        pat_dod = con.execute(sql_pat, [subject_ids]).fetchdf()
    finally:
        con.close()

    # normalize types
    adm_flags = adm_flags.astype({"admission_id": utils.ID_DTYPE})
    pat_dod = pat_dod.astype({"subject_id": utils.ID_DTYPE})
    pat_dod["dod"] = pd.to_datetime(pat_dod["dod"], errors="coerce")

    # Merge flags + dod onto base
    df = base.merge(adm_flags, on="admission_id", how="left").merge(pat_dod, on="subject_id", how="left")

    # death during admission (treat missing as False)
    death_during_admission = (df["hospital_expire_flag"] == 1)

    # death within 30 days post-discharge (that didn't occur in the hospital)
    disch_plus_30 = df["dischtime"] + pd.Timedelta(days=30)
    death_in_30_days_after_discharge = ((~death_during_admission) & (df["dod"] <= disch_plus_30))

    # Final mortality
    mortality = (death_during_admission | death_in_30_days_after_discharge).astype("boolean")

    out = df.loc[:, ["subject_id", "admission_id"]].copy()
    out["mortality"] = mortality

    ## LOGGING AND REPORTS BELOW, THEN SAVING
    # invalid hospital_expire_flag
    hef_na_mask = df["hospital_expire_flag"].isna()
    if hef_na_mask.any():
        labels_logger.info(f"[mortality] Missing hospital_expire_flag rows: {int(hef_na_mask.sum()):,} (logging individually)")
        for row in df.loc[hef_na_mask, ["subject_id", "admission_id", "hospital_expire_flag"]].itertuples(index=False):
            labels_logger.warning(
                f"[mortality] NA hospital_expire_flag -> subject_id={int(row.subject_id)}, admission_id={int(row.admission_id)}"
            )

    # Rows where the 30-day comparison is NA
    death_within_30_days_na_mask = death_in_30_days_after_discharge.isna()
    if death_within_30_days_na_mask.any():
        labels_logger.info(f"[mortality] NA in 30-day comparison rows: {int(death_within_30_days_na_mask.sum()):,} (logging individually)")
        for row in df.loc[death_within_30_days_na_mask, ["subject_id", "admission_id", "dod", "dischtime"]].itertuples(index=False):
            labels_logger.warning(
                f"[mortality] NA 30d comparison -> subject_id={int(row.subject_id)}, "
                f"admission_id={int(row.admission_id)}, dod={row.dod}, dischtime={row.dischtime}"
            )

    # Log cases where DOD < dischtime (pre-discharge death recorded as DOD but flag == 0)
    odd_pre_discharge = (~death_during_admission) & df["dod"].notna() & (df["dod"] < df["dischtime"])
    if odd_pre_discharge.any():
        labels_logger.warning(
            f"[mortality] {int(odd_pre_discharge.sum()):,} rows have DOD < dischtime while HOSPITAL_EXPIRE_FLAG == 0."
        )
        for row in df.loc[odd_pre_discharge, ["subject_id","admission_id","dod","dischtime"]].head(20).itertuples(index=False):
            labels_logger.warning(
                f"[mortality] PRE-DISCH? subj={int(row.subject_id)}, hadm={int(row.admission_id)}, dod={row.dod}, disch={row.dischtime}"
            )

    # recompute component counts/overlap
    n_deaths_in_admission = int(death_during_admission.sum())
    n_deaths_within_30_days_post_discharge = int(death_in_30_days_after_discharge.sum())
    n_overlap = int((death_during_admission & death_in_30_days_after_discharge).sum())
    labels_logger.info(
        f"[mortality] Component counts (disjoint): during_admission={n_deaths_in_admission:,}, within_30d_post_discharge={n_deaths_within_30_days_post_discharge:,}, overlap={n_overlap:,}"
    )
    if n_overlap:
        labels_logger.warning(f"[mortality] Unexpected non-zero overlap: {n_overlap}")

    # report basic stats
    pos = int(out["mortality"].sum())
    neg = int((~out["mortality"]).sum())
    labels_logger.info(
        f"[mortality] Wrote labels to {utils.MORTALITY_LABEL_PATH} | "
        f"N={len(out):,}, positives={pos:,} ({pos / len(out) * 100:.2f}%), "
        f"negatives={neg:,} ({neg / len(out) * 100:.2f}%)"
    )
    # save
    out.to_parquet(utils.MORTALITY_LABEL_PATH, index=False)
    return out


def create_prolonged_stay_df(cohort_path: Path) -> pd.DataFrame:
    """
    prolonged_stay := (dischtime > admittime + 7 days)

    Output parquet: utils.utils.PROLONGED_STAY_LABEL_PATH with columns:
      - subject_id    (int32)
      - admission_id  (int32)
      - prolonged_stay (boolean)
    """
    # Load cohort (subject_id, admission_id, admittime, dischtime)
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    df = pd.read_parquet(cohort_path).astype(
        {"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE}
    )
    labels_logger.info(
        f"[prolonged_stay] Loaded base cohort: {len(df):,} rows from {cohort_path}"
    )

    # Ensure timestamps
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")

    # Label: dischtime > admittime + 7 days
    seven_days_after_admit = df["admittime"] + pd.Timedelta(days=7)
    prolonged = (df["dischtime"] > seven_days_after_admit).astype("boolean")

    out = df.loc[:, ["subject_id", "admission_id"]].copy()
    out["prolonged_stay"] = prolonged

    # logging
    pos = int(out["prolonged_stay"].sum())
    neg = len(out) - pos
    labels_logger.info(
        f"[prolonged_stay] Wrote labels to {utils.PROLONGED_STAY_LABEL_PATH} | "
        f"N={len(out):,}, positives={pos:,} ({pos / len(out) * 100:.2f}%), "
        f"negatives={neg:,} ({neg / len(out) * 100:.2f}%)"
    )
    # saving
    out.to_parquet(utils.PROLONGED_STAY_LABEL_PATH, index=False)
    return out


def create_readmission_df(cohort_path: Path, second_admissions_path: Path, overwrite: bool = True) -> pd.DataFrame:
    """
    Build the readmission label dataframe.

    Inputs (Parquet):
      - second_admissions_path: columns
          subject_id (Int32), second_admission_id (Int32, nullable),
          second_admittime (datetime64[ns], nullable), nonpositive_duration (bool)
      - cohort_path: columns
          subject_id (Int32), dischtime (datetime64[ns], non-null)

    Output (Parquet): utils.READMISSION_LABEL_PATH with columns
      - subject_id (Int32)
      - second_admission_id (Int32, nullable)  # identical to SECOND_ADMISSIONS
      - readmission (bool)                     # 30 days (inclusive)

    """
    if utils.READMISSION_LABEL_PATH.exists() and not overwrite:
        labels_logger.info(
            f"Readmission file already exists. Loading it from {utils.READMISSION_LABEL_PATH}"
        )
        return pd.read_parquet(utils.READMISSION_LABEL_PATH)

    # Load inputs
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    second_admissions_path = second_admissions_path or utils.SECOND_ADMISSIONS_PATH
    second_adm = pd.read_parquet(second_admissions_path).copy()
    cohort = pd.read_parquet(cohort_path).copy()

    # Dtypes & timestamps
    second_adm["subject_id"] = second_adm["subject_id"].astype(utils.ID_DTYPE)
    cohort["subject_id"] = cohort["subject_id"].astype(utils.ID_DTYPE)
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"], errors="raise")
    if "second_admittime" in second_adm:
        second_adm["second_admittime"] = pd.to_datetime(second_adm["second_admittime"], errors="coerce")

    # Merge one-to-one on subject_id
    df = cohort[["subject_id", "dischtime"]].merge(
        second_adm[["subject_id", "second_admission_id", "second_admittime", "nonpositive_duration"]],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    # Normalize to calendar dates
    discharge_date = df["dischtime"].dt.normalize()
    second_date = df["second_admittime"].dt.normalize()

    # Masks
    has_second_time = df["second_admittime"].notna()
    bad_order_mask = has_second_time & (df["second_admittime"] < df["dischtime"])
    bad_duration_mask = df["nonpositive_duration"].fillna(False)

    # Logging for bad data (labels will be forced False)
    if bad_order_mask.any():
        labels_logger.warning(
            f"Found {int(bad_order_mask.sum()):,} subject(s) with second_admittime < dischtime (label forced False)."
        )
        for sid, aid, adm_t, dis_t in df.loc[
            bad_order_mask, ["subject_id", "second_admission_id", "second_admittime", "dischtime"]
        ].itertuples(index=False):
            labels_logger.warning(
                f"second admit < first discharge -> subject_id={int(sid)}, "
                f"second_admission_id={aid}, second_admittime={adm_t}, dischtime={dis_t}"
            )

    if bad_duration_mask.any():
        labels_logger.warning(
            f"Found {int(bad_duration_mask.sum()):,} subject(s) with nonpositive second-admission duration (label forced False)."
        )
        for sid, aid in df.loc[bad_duration_mask, ["subject_id", "second_admission_id"]].itertuples(index=False):
            labels_logger.warning(
                f"nonpositive second admission duration -> subject_id={int(sid)}, second_admission_id={aid}"
            )

    # Valid rows for the calendar rule
    good_mask = has_second_time & (~bad_order_mask) & (~bad_duration_mask)

    # Inclusive 30 calendar-day rule
    within_30 = good_mask & (second_date <= (discharge_date + pd.Timedelta(days=30)))

    # Build label
    readmission_columm = pd.Series(False, index=df.index, dtype="bool")
    readmission_columm.loc[within_30] = True

    # build output df
    readmission_df = pd.DataFrame(
        {
            "subject_id": df["subject_id"].astype("Int32"),
            "second_admission_id": df["second_admission_id"].astype("Int32"),
            "readmission": readmission_columm.astype("bool"),
        }
    )

    # save
    utils.READMISSION_LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    readmission_df.to_parquet(utils.READMISSION_LABEL_PATH, index=False)
    # logging
    labels_logger.info(
        f"Wrote readmission labels to {utils.READMISSION_LABEL_PATH} "
        f"(rows={len(readmission_df):,}; positives={int(readmission_df['readmission'].sum()):,})."
    )

    return readmission_df


if __name__ == "__main__":
    create_mortality_df(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    create_prolonged_stay_df(cohort_path=utils.FILTERED_COHORT_PATH)
    create_readmission_df(cohort_path=utils.FILTERED_COHORT_PATH, second_admissions_path=utils.SECOND_ADMISSIONS_PATH)
    pass
