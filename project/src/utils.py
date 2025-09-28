import logging
from pathlib import Path

import pandas as pd

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

ROOT = _project_root()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "mimiciii.duckdb"
INITIAL_COHORT_CSV = DATA_DIR / "initial_cohort.csv"
VITALS_CSV = DATA_DIR / "vitals.csv"
LABS_CSV = DATA_DIR / "labs.csv"
GCS_CSV = DATA_DIR / "gcs.csv"

PIPELINE_DIR = DATA_DIR / "pipeline"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

# label files:
MORTALITY_LABEL_PATH = PIPELINE_DIR / '1-2-1-mortality.parquet'
PROLONGED_STAY_LABEL_PATH = PIPELINE_DIR / '1-2-2-prolonged-stay.parquet'
READMISSION_LABEL_PATH = PIPELINE_DIR / '1-2-3-readmission.parquet'

FIRST_ADMISSIONS_PATH = PIPELINE_DIR / "1-3-1-first_admissions.parquet"
MIN_54H_FIRST_ADMISSIONS_PATH = PIPELINE_DIR / "1-3-2-min_54h_first_admissions.parquet"
MIN_54H_FIRST_ADMISSIONS_FILTERED_AGE_PATH = PIPELINE_DIR / "1-3-3-min_54h_first_admissions_filtered_age_path.parquet"

# ===== VERY IMPORTANT!!! =====
FILTERED_COHORT_PATH = MIN_54H_FIRST_ADMISSIONS_FILTERED_AGE_PATH
# =============================

SECOND_ADMISSIONS_PATH = PIPELINE_DIR / "1-3-helper-second_admissions.parquet"

DEMOGRAPHICS_PATH = PIPELINE_DIR / "1-4-1a-demographics.parquet"

# vitals data
FIRST_48H_VITALS_PATH = DATA_DIR / "vitals_first_48h.parquet"
VITALS_PATH = PIPELINE_DIR / "1-4-1b-vitals.parquet"
VITALS_FEATURES_PATH = PIPELINE_DIR / "1-4-1b-vitals_features.parquet"

# labs data
FIRST_48H_LABS_PATH = DATA_DIR / "labs_first_48h.parquet"
LABS_PATH = PIPELINE_DIR / "1-4-1c-labs.parquet"
LABS_FEATURES_PATH = PIPELINE_DIR / "1-4-1c-labs_features.parquet"

# gcs data
FIRST_48H_GCS_PATH = DATA_DIR / "gcs_first_48h.parquet"
GCS_PATH = PIPELINE_DIR / "1-4-1d1-gcs.parquet"
GCS_FEATURES_PATH = PIPELINE_DIR / "1-4-1d1-gcs_features.parquet"

# prescriptions data
FIRST_48H_PRESCRIPTIONS_PATH = DATA_DIR / "prescriptions_first_48h.parquet"
PRESCRIPTIONS_FEATURES_PATH = PIPELINE_DIR / "1-4-1d2-prescriptions_features.parquet"

# noteevents data
FIRST_48H_NOTEEVENTS_PATH = DATA_DIR / "noteevents_first_48h.parquet"
NOTEEVENTS_FEATURES_PATH = PIPELINE_DIR / "1-4-1d3-noteevents.parquet"

# embeddings data
EMBEDDINGS_FEATURES_PATH_200 = PIPELINE_DIR / "1-4-1d4-embeddings_features_200.parquet"
EMBEDDINGS_FEATURES_PATH_512 = PIPELINE_DIR / "1-4-1d4-embeddings_features_512.parquet"

EMBEDDINGS_FEATURES_PATH = EMBEDDINGS_FEATURES_PATH_200


LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# FIGURES_DIR = ROOT / "figures"
# FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AGE_DTYPE = "Int16"
ID_DTYPE = "int32"
SEEN_LANGUAGES_PATH = DATA_DIR / "seen_languages.json"


def make_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(LOGS_DIR / filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def get_parquet_value_counts(parquet_path: Path, cols: list | str = None):

    df = pd.read_parquet(parquet_path)
    cols = cols or df.columns.values.tolist()
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        print(f"\n=== {col} ===")
        print(df[col].value_counts(dropna=False).sort_values(ascending=False))
