from pathlib import Path
import json

import duckdb
import pandas as pd

from . import utils

# AGE_FIG_PATH = utils.FIGURES_DIR / "age.png"

logger = utils.make_logger("demographic_features", "demographic_features.log")


def add_age(df: pd.DataFrame, db_path: Path = None) -> pd.DataFrame:
    """
    Compute floor age (years) at admittime.
    Assumes df has valid:
      - subject_id (int32)
      - admission_id (int32)
      - admittime (datetime64[ns])
    And that all subject_ids have valid DOBs (we already made sure of these things when we created the filtered cohort.).
    """
    out = df.copy()

    # Get the dobs from DuckDB
    subject_ids = out["subject_id"].astype("int64").unique().tolist()
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        patients_sql = """
            SELECT
                CAST(subject_id AS INTEGER)   AS subject_id,
                CAST(dob        AS TIMESTAMP) AS dob
            FROM patients
            WHERE CAST(subject_id AS INTEGER) IN ?
        """
        patients_df = con.execute(patients_sql, [subject_ids]).fetchdf()
    finally:
        con.close()

    # compute floor age at admittime
    out = out.merge(patients_df.astype({"subject_id": utils.ID_DTYPE}), on="subject_id", how="left")
    out["dob"] = pd.to_datetime(out["dob"], errors="raise")          # assume valid
    out["admittime"] = pd.to_datetime(out["admittime"], errors="raise")

    admit_md = out["admittime"].dt.month * 100 + out["admittime"].dt.day
    dob_md   = out["dob"].dt.month       * 100 + out["dob"].dt.day
    years    = out["admittime"].dt.year - out["dob"].dt.year
    before_bday = (admit_md < dob_md).astype("int64")

    age_years = (years - before_bday).astype("int64")
    result = df.copy()
    result["age"] = age_years.astype(utils.AGE_DTYPE)
    return result


def add_ethnicity(df: pd.DataFrame, db_path: Path = None) -> pd.DataFrame:
    """
    Add two columns based on ADMISSIONS.ethnicity:
      - eth_raw: raw ethnicity standardized (strip + lowercase)
      - eth_broad: consolidated ethnicity category using our handcrafted rules
    """

    hadm_ids = df["admission_id"].astype("int64").unique().tolist()

    # get ethnicity from DuckDB
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
            SELECT CAST(hadm_id AS INTEGER)   AS admission_id,
                   CAST(ethnicity AS VARCHAR) AS ethnicity
            FROM admissions
            WHERE CAST(hadm_id AS INTEGER) IN ?
        """
        adm = con.execute(sql, [hadm_ids]).fetchdf()
    finally:
        con.close()

    out = df.merge(adm.astype({"admission_id": utils.ID_DTYPE}), on="admission_id", how="left")

    # Standardize: strip + lowercase (eth_raw)
    ethnicity_raw = (
        pd.Series(out["ethnicity"], dtype="string")
        .str.strip()
        .str.lower()
    )
    out = out.drop(columns=["ethnicity"])
    out["eth_raw"] = ethnicity_raw

    # Build eth_broad based on our rules:
    eth_broad = pd.Series(pd.NA, index=out.index, dtype="string")
    raw_ethnicity = out["eth_raw"]

    # simple groups:
    eth_broad[raw_ethnicity == "patient declined to answer"] = "DECLINED"
    eth_broad[raw_ethnicity == "other"] = "OTHER"
    eth_broad[raw_ethnicity == "multi race ethnicity"] = "MULTI RACE"
    eth_broad[(raw_ethnicity == "unknown/not specified") | (raw_ethnicity == "unable to obtain")] = "UNKNOWN"

    # AFRICAN AMERICAN
    eth_broad[raw_ethnicity.str.contains("african american", na=False)] = "AFRICAN AMERICAN"

    # AFRICAN (black/african WITHOUT 'american')
    mask_african = (raw_ethnicity.str.contains("black", na=False) | raw_ethnicity.str.contains("african", na=False)) & (~raw_ethnicity.str.contains("american", na=False))
    eth_broad[mask_african] = "AFRICAN"

    # CARIBBEAN
    eth_broad[(raw_ethnicity == "black/haitian") | raw_ethnicity.str.contains("caribbean", na=False)] = "CARIBBEAN"

    # NATIVE AMERICAN
    mask_native_american = (
        raw_ethnicity.str.contains("native american", na=False)
        | raw_ethnicity.str.contains("alaska native", na=False)
        | raw_ethnicity.str.contains("american indian", na=False)
    )
    eth_broad[mask_native_american] = "NATIVE AMERICAN"

    # PACIFIC ISLANDER
    eth_broad[raw_ethnicity.str.contains("native hawaiian", na=False) | raw_ethnicity.str.contains("pacific islander", na=False)] = "PACIFIC ISLANDER"

    # HISPANIC (including portuguese, south/central american)
    mask_hispanic = (
        (raw_ethnicity == "portuguese")
        | raw_ethnicity.str.contains("hispanic", na=False)
        | raw_ethnicity.str.contains("latino", na=False)
        | raw_ethnicity.str.contains("central american", na=False)
        | raw_ethnicity.str.contains("south american", na=False)
    )
    eth_broad[mask_hispanic] = "HISPANIC"

    # ASIAN
    eth_broad[raw_ethnicity.str.contains("asian", na=False)] = "ASIAN"

    # WHITE
    eth_broad[raw_ethnicity.str.contains("white", na=False)] = "WHITE"

    # MIDDLE EASTERN
    eth_broad[raw_ethnicity.str.contains("middle eastern", na=False) | raw_ethnicity.str.contains("arab", na=False)] = "MIDDLE EASTERN"

    out["eth_broad"] = eth_broad

    logger.info(f"=== ETH_RAW value counts ===\n {out['eth_raw'].value_counts(dropna=False).to_string()}")
    logger.info(f"=== ETH_BROAD value counts ===\n {out['eth_broad'].value_counts(dropna=False).to_string()}")
    logger.info(f"NA counts: eth_raw={out['eth_raw'].isna().sum()}, eth_broad={out['eth_broad'].isna().sum()}")

    return out


def add_gender(df: pd.DataFrame, db_path: Path = None) -> pd.DataFrame:
    """
    gender standardized -> uppercase + strip
    """
    subject_ids = df["subject_id"].astype("int64").unique().tolist()

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
            SELECT CAST(subject_id AS INTEGER) AS subject_id,
                   GENDER AS gender_raw
            FROM patients
            WHERE CAST(subject_id AS INTEGER) IN ?
        """
        pat = con.execute(sql, [subject_ids]).fetchdf()
    finally:
        con.close()

    out = df.merge(pat.astype({"subject_id": utils.ID_DTYPE}), on="subject_id", how="left")

    # Standardize: strip + uppercase (preserve NA)
    raw_gender = pd.Series(out["gender_raw"], dtype="string")
    standardized_gender = raw_gender.str.strip().str.upper()

    out["gender"] = standardized_gender.astype("string")
    out = out.drop(columns=["gender_raw"])

    logger.info(f"=== GENDER value counts ===\n{out['gender'].value_counts(dropna=False).to_string()}")
    logger.info(f"GENDER NA counts: gender={out['gender'].isna().sum()}")

    return out


def add_insurance(df: pd.DataFrame, db_path: Path = None) -> pd.DataFrame:
    """
    insurance standardized -> lowercase + strip
    """
    hadm_ids = df["admission_id"].astype("int64").unique().tolist()

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
            SELECT CAST(hadm_id AS INTEGER)   AS admission_id,
                   CAST(insurance AS VARCHAR) AS insurance
            FROM admissions
            WHERE CAST(hadm_id AS INTEGER) IN ?
        """
        admissions_insurance = con.execute(sql, [hadm_ids]).fetchdf()
    finally:
        con.close()

    out = df.merge(
        admissions_insurance.astype({"admission_id": utils.ID_DTYPE}),
        on="admission_id",
        how="left",
    )

    # Standardize: strip + lowercase
    raw_insurance = pd.Series(out["insurance"], dtype="string")
    standardized_insurance = raw_insurance.str.strip().str.lower()

    allowed_insurance_values = {"medicare", "private", "medicaid", "government", "self pay"}

    # Keep allowed labels, everything else -> NA
    out["insurance"] = standardized_insurance.where(standardized_insurance.isin(allowed_insurance_values), other=pd.NA).astype("string")

    logger.info(f"=== INSURANCE value counts ===\n{out['insurance'].value_counts(dropna=False).to_string()}")
    logger.info(f"INSURANCE NA count: {out['insurance'].isna().sum()}")

    return out


# def add_language(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Join ADMISSIONS.language by admission_id and add:
#       - language : lowercase, letters-only (all non-letters removed).
#         Then collapse prefix codes to the longest observed superstring (within the current batch).
#         Additionally:
#           * On first (training) run: save the post-merge set of language strings to data/seen_languages.json
#           * On subsequent (test) runs: load that file and remap current values:
#               - if a value is a prefix of any seen language, map to the longest matching seen language
#               - otherwise -> utils.UNKNOWN_VALUE
#
#     Logging:
#       - Do NOT log rows where the raw value is NA (missing).
#       - Log rows where the cleaned value becomes empty (raw non-NA -> cleaned-empty).
#       - On test runs, log rows remapped to utils.UNKNOWN_VALUE due to unseen/not-matching prefixes.
#
#     Missing handling:
#       - Raw NA -> utils.UNKNOWN_VALUE (no logging).
#       - Cleaned-empty -> utils.UNKNOWN_VALUE (logged).
#     """
#     hadm_ids = df["admission_id"].astype("int64").unique().tolist()
#
#     db_path = db_path or utils.DB_PATH
#     con = duckdb.connect(str(db_path))
#     try:
#         sql = """
#             SELECT CAST(hadm_id AS INTEGER)    AS admission_id,
#                    CAST(language AS VARCHAR)   AS language
#             FROM admissions
#             WHERE CAST(hadm_id AS INTEGER) IN ?
#         """
#         languages_from_admission = con.execute(sql, [hadm_ids]).fetchdf()
#     finally:
#         con.close()
#
#     out = df.merge(
#         languages_from_admission.astype({"admission_id": utils.ID_DTYPE}),
#         on="admission_id",
#         how="left",
#     )
#
#     # Raw values (nullable)
#     raw_language = pd.Series(out["language"], dtype="string")
#
#     # Clean: keep letters only, lowercase (NaNs remain NaN here)
#     cleaned_language = raw_language.str.replace(r"[^A-Za-z]+", "", regex=True).str.lower()
#
#     # Log ONLY rows where raw is non-NA but cleaned becomes empty
#     empty_after_clean_mask = raw_language.notna() & (cleaned_language.fillna("").str.len() == 0)
#     if empty_after_clean_mask.any():
#         logger.info(
#             f"add_language: {int(empty_after_clean_mask.sum())} row(s) with cleaned-empty language; logging individually."
#         )
#         for subject_id, admission_id, raw_val in out.loc[
#             empty_after_clean_mask, ["subject_id", "admission_id", "language"]
#         ].itertuples(index=False):
#             logger.warning(
#                 f"Cleaned-empty language: subject_id={int(subject_id)}, admission_id={int(admission_id)}, "
#                 f"raw='{str(raw_val)}' -> {utils.UNKNOWN_VALUE}"
#             )
#
#     cleaned_language_unique_vals = (
#         cleaned_language.dropna()[cleaned_language.dropna().str.len() > 0]
#         .unique()
#         .tolist()
#     )
#
#     # Longest-first for deterministic “most specific” mapping
#     longest_first = sorted(cleaned_language_unique_vals, key=len, reverse=True)
#
#     # Map each token to the longest language superstring, or itself
#     prefix_matching_map = {
#         v: next((w for w in longest_first if w != v and w.startswith(v)), v)
#         for v in cleaned_language_unique_vals
#     }
#
#     # Initial normalization (train/test agnostic):
#     normalized_language = cleaned_language.copy()
#     normalized_language = normalized_language.mask(
#         normalized_language.fillna("").str.len() == 0, other=utils.UNKNOWN_VALUE
#     )
#     normalized_language = normalized_language.fillna(utils.UNKNOWN_VALUE)
#
#     to_map_mask = normalized_language.ne(utils.UNKNOWN_VALUE)
#     if to_map_mask.any():
#         normalized_language.loc[to_map_mask] = (
#             normalized_language.loc[to_map_mask].map(prefix_matching_map).astype("string")
#         )
#
#     current_seen = (
#         pd.Series(normalized_language[to_map_mask].dropna().unique(), dtype="string")
#         .loc[lambda s: s.ne(utils.UNKNOWN_VALUE)]
#         .tolist()
#     )
#     current_seen_sorted = sorted(set(current_seen), key=lambda x: (-len(x), x))
#
#     if utils.SEEN_LANGUAGES_PATH.exists():
#         with open(utils.SEEN_LANGUAGES_PATH, "r", encoding="utf-8") as f:
#             seen_list = json.load(f)
#
#         # Longest-first for prefix matching
#         seen_sorted = sorted(set(seen_list), key=lambda x: (-len(x), x))
#
#         # Remap current values to longest matching seen prefix; else utils.UNKNOWN_VALUE
#         vals = normalized_language[to_map_mask].unique().tolist()
#         remap_dict = {
#             v: next((s for s in seen_sorted if s.startswith(v)), utils.UNKNOWN_VALUE)
#             for v in vals
#         }
#
#         normalized_language.loc[to_map_mask] = (
#             normalized_language.loc[to_map_mask].map(remap_dict).astype("string")
#         )
#
#         forced_unknown_mask = to_map_mask & normalized_language.eq(utils.UNKNOWN_VALUE)
#         if forced_unknown_mask.any():
#             logger.info(
#                 f"add_language: {int(forced_unknown_mask.sum())} row(s) mapped to {utils.UNKNOWN_VALUE} due to unseen prefixes."
#             )
#             for subject_id, admission_id, v in out.loc[
#                 forced_unknown_mask, ["subject_id", "admission_id", "language"]
#             ].itertuples(index=False):
#                 logger.warning(
#                     f"Unseen language after prefix check: subject_id={int(subject_id)}, "
#                     f"admission_id={int(admission_id)}, raw='{str(v)}' -> {utils.UNKNOWN_VALUE}"
#                 )
#     else:
#         # save the seen languages as json
#         utils.SEEN_LANGUAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
#         with open(utils.SEEN_LANGUAGES_PATH, "w", encoding="utf-8") as f:
#             json.dump(current_seen_sorted, f, ensure_ascii=False, indent=2)
#         logger.info(
#             f"add_language: saved {len(current_seen_sorted)} seen language(s) to {utils.SEEN_LANGUAGES_PATH}"
#         )
#
#     out["language"] = normalized_language.astype("string")
#     return out


def add_admission_type(df: pd.DataFrame, db_path: Path = None) -> pd.DataFrame:
    """
    admission_type standardized -> UPPERCASE + strip
    """
    hadm_ids = df["admission_id"].astype("int64").unique().tolist()

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
            SELECT CAST(hadm_id AS INTEGER)        AS admission_id,
                   CAST(admission_type AS VARCHAR) AS admission_type
            FROM admissions
            WHERE CAST(hadm_id AS INTEGER) IN ?
        """
        adm = con.execute(sql, [hadm_ids]).fetchdf()
    finally:
        con.close()

    out = df.merge(adm.astype({"admission_id": utils.ID_DTYPE}), on="admission_id", how="left")

    # Standardize: strip + UPPERCASE (preserve NA)
    raw_adm_type = pd.Series(out["admission_type"], dtype="string")
    standardized_admission_type = raw_adm_type.str.strip().str.upper()

    allowed_admission_types = {"EMERGENCY", "NEWBORN", "ELECTIVE", "URGENT"}  # todo verify we don't have newborns? (since 1-3-3 restricts the ages to 18+)

    # Keep allowed labels, everything else -> NA
    out["admission_type"] = standardized_admission_type.where(
        standardized_admission_type.isin(allowed_admission_types),
        other=pd.NA
    ).astype("string")

    logger.info(f"=== ADMISSION_TYPE value counts ===\n{out['admission_type'].value_counts(dropna=False).to_string()}")
    logger.info(f"ADMISSION_TYPE NA count: {out['admission_type'].isna().sum()}")

    return out


def add_admission_location(df: pd.DataFrame, db_path: Path = None) -> pd.DataFrame:
    """
    admission_location standardized -> UPPERCASE + strip
    """
    hadm_ids = df["admission_id"].astype("int64").unique().tolist()

    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    try:
        sql = """
            SELECT CAST(hadm_id AS INTEGER)             AS admission_id,
                   CAST(admission_location AS VARCHAR)  AS admission_location
            FROM admissions
            WHERE CAST(hadm_id AS INTEGER) IN ?
        """
        adm = con.execute(sql, [hadm_ids]).fetchdf()
    finally:
        con.close()

    out = df.merge(adm.astype({"admission_id": utils.ID_DTYPE}), on="admission_id", how="left")

    # Standardize: strip + UPPERCASE (preserve NA)
    raw_loc = pd.Series(out["admission_location"], dtype="string")
    standardized_admission_location = raw_loc.str.strip().str.upper()

    # Set of allowed values (see https://mimic.mit.edu/docs/iii/tables/admissions/#admission_location)
    allowed_admission_locations = {
        "EMERGENCY ROOM ADMIT",
        "TRANSFER FROM HOSP/EXTRAM",
        "TRANSFER FROM OTHER HEALT",
        "CLINIC REFERRAL/PREMATURE",
        "** INFO NOT AVAILABLE **",
        "TRANSFER FROM SKILLED NUR",
        "TRSF WITHIN THIS FACILITY",
        "HMO REFERRAL/SICK",
        "PHYS REFERRAL/NORMAL DELI",
    }

    # Keep allowed labels, everything else -> NA
    out["admission_location"] = standardized_admission_location.where(
        standardized_admission_location.isin(allowed_admission_locations),
        other=pd.NA
    ).astype("string")

    logger.info(f"=== ADMISSION_LOCATION value counts ===\n{out['admission_location'].value_counts(dropna=False).to_string()}")
    logger.info(f"ADMISSION_LOCATION NA count: {out['admission_location'].isna().sum()}")

    return out


def create_demographics_df(cohort_path: Path = None, db_path: Path = None) -> pd.DataFrame:

    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    logger.info(f"Loading: {cohort_path}")
    df = pd.read_parquet(cohort_path).astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})

    db_path = db_path or utils.DB_PATH

    # AGE
    logger.info(f"Computing age at admission for {len(df):,} rows")
    df = add_age(df, db_path=db_path)
    # logger.info(f"Plotting age histogram to: {AGE_FIG_PATH}")
    # plot_age_histogram(df, AGE_FIG_PATH)

    # ETHNICITY (we add two columns)
    logger.info(f"Computing ethnicity for {len(df):,} rows")
    df = add_ethnicity(df, db_path=db_path)

    # GENDER
    logger.info(f"Computing gender for {len(df):,} rows")
    df = add_gender(df, db_path=db_path)

    # INSURANCE
    logger.info(f"Computing insurance for {len(df):,} rows")
    df = add_insurance(df, db_path=db_path)

    # LANGUAGE - todo in the removal of the utils.UNKNOWN_VALUE, because we also create a file there, the amount of time it would take me to refactor this is not worthwhile. So for now I'll just won't add the language, Or's not sure we should use the language anyway. So although I wanted to showcase the prefix collapse + the languages file, until I'm 100% of the training features pipeline, let's just not use the language.
    # logger.info(f"Computing language for {len(df):,} rows")
    # df = add_language(df)

    # ADMISSION TYPE
    logger.info(f"Computing admission_type for {len(df):,} rows")
    df = add_admission_type(df, db_path=db_path)

    # ADMISSION LOCATION
    logger.info(f"Computing admission_location for {len(df):,} rows")
    df = add_admission_location(df, db_path=db_path)

    # remove the admittime and dischtime column if they exist:
    df = df.drop(columns=["admittime", "dischtime"], errors="ignore")  # ignore beacause we don't want pandas to scream if one of these columns do not exist in `df`.

    logger.info(f"Writing demographics to: {utils.DEMOGRAPHICS_PATH}")
    logger.info(f"columns in demographics: {df.columns}")
    df.to_parquet(utils.DEMOGRAPHICS_PATH, index=False)
    logger.info("Done.")

    return df


if __name__ == "__main__":
    create_demographics_df(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
