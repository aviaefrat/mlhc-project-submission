### lovely and totally not ugly patch for loading the preprocessors =\
import sys, importlib
try:
    import src  # no problem in our local development, but fails in project.zip.
except ModuleNotFoundError:
    import project.src as psrc
    sys.modules['src'] = psrc
    for a,b in {
        'src.utils':'project.src.utils',
        'src.modeling':'project.src.modeling',
        'src.modeling.preprocessing':'project.src.modeling.preprocessing',
        'src.modeling.io':'project.src.modeling.io',
        'src.modeling.metrics':'project.src.modeling.metrics',
        'src.modeling.split':'project.src.modeling.split',
        'src.modeling.calibration':'project.src.modeling.calibration',
    }.items():
        try: sys.modules[a] = importlib.import_module(b)
        except Exception: pass
### end of ugly patch

from pathlib import Path

import numpy as np
import pandas as pd

# Our project modules
from .src import utils
from .src.patient_timeline_filtering import (
    create_first_admissions_df,
    create_min_54h_first_admissions_df,
    create_min_54h_first_admissions_age_filtered_df,
)
from .src.demographic_features import create_demographics_df

from .src.vitals import (
    create_all_vitals_for_filtered_cohort_0_48h,
    create_vitals_df,
)
from .src.vitals_features import build_vitals_features

from .src.labs import (
    create_all_labs_for_filtered_cohort_0_48h,
    create_labs_df,
)
from .src.labs_features import build_labs_features

from .src.gcs import (
    create_all_gcs_for_filtered_cohort_0_48h,
    create_gcs_df,
)
from .src.gcs_features import build_gcs_features

from .src.prescriptions import extract_prescriptions_first_48h
from .src.prescriptions_features import build_prescriptions_features

from .src.embeddings_features import build_embeddings_features

from .src.modeling.io import load_model_bundle


def _merge_on_keys(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    # merge dataframes on the correct ids.
    keys = []
    for k in ["subject_id", "admission_id"]:
        if k in left.columns and k in right.columns:
            keys.append(k)
    if not keys:
        raise ValueError("No common join keys between frames.")
    return left.merge(right, on=keys, how="inner", validate="one_to_one")


def _build_inference_features() -> pd.DataFrame:
    """
    Load all feature parquet files created by the feature builders and create a single DataFrame with one row per subject_id.
    We use the dvlgpe features (without non-embedding noteevents), like we did in training time
    """
    frames: list[pd.DataFrame] = []

    # Demographics
    demo = pd.read_parquet(utils.DEMOGRAPHICS_PATH)
    frames.append(demo)

    # Vitals features
    vitf = pd.read_parquet(utils.VITALS_FEATURES_PATH)
    frames.append(vitf)

    # Labs features
    labsf = pd.read_parquet(utils.LABS_FEATURES_PATH)
    frames.append(labsf)

    # GCS features
    gcsf = pd.read_parquet(utils.GCS_FEATURES_PATH)
    frames.append(gcsf)

    # Prescriptions features
    prxf = pd.read_parquet(utils.PRESCRIPTIONS_FEATURES_PATH)
    frames.append(prxf)

    # Embeddings features
    embf = pd.read_parquet(utils.EMBEDDINGS_FEATURES_PATH)
    frames.append(embf)

    # merge the features dataframes
    merged = frames[0]
    for df in frames[1:]:
        merged = _merge_on_keys(merged, df)

    return merged


def _predict_with_bundle(bundle_dir: Path, X_df: pd.DataFrame) -> np.ndarray:
    """
    Load (preprocessor, model, calibrator) and return calibrated probabilities.
    """
    pre, model, calibrator = load_model_bundle(bundle_dir)
    if calibrator is None:
        raise RuntimeError(f"Missing calibrator.joblib in {bundle_dir}")

    # Transform using the *saved* preprocessor
    Xmat = pre.transform(X_df)
    p = calibrator.predict_proba(Xmat)[:, 1]
    return p


def run_pipeline_on_unseen_data(subject_ids, con):
    """
    Run your full pipeline, from data loading to prediction.

    :param subject_ids: A list of subject IDs of an unseen test set.
    :type subject_ids: List[int]

    :param con: A DuckDB connection object.
    :type con: duckdb.connection.Connection

    :return: DataFrame with the following columns:
                - subject_id: Subject IDs, which in some cases can be different due to your analysis.
                - mortality_proba: Prediction probabilities for mortality.
                - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
                - readmission_proba: Prediction probabilities for readmission.
    :rtype: pandas.DataFrame
    """
    # Our builders rely on a db_path and not on a duckdb connection itself. So let's get the path from the given duckdb connection.
    db_path = con.execute("PRAGMA database_list").fetchone()[2]

    # All of our builders reference utils.DB_PATH, so temporarily ensure it points to the given duckdb connection.
    # (utils.DB_PATH is defined at import time. we overwrite the Path object here.)
    utils.DB_PATH = Path(db_path)

    # Build utils.FILTERED_COHORT_PATH (first admissions -> >=54h -> age filter) for the subject_ids:
    create_first_admissions_df(subject_ids=subject_ids, db_path=utils.DB_PATH)
    create_min_54h_first_admissions_df(db_path=utils.DB_PATH)
    create_min_54h_first_admissions_age_filtered_df(db_path=utils.DB_PATH)

    # Create the dvlgpe features for this cohort:
    # Demographics
    create_demographics_df(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)

    # Vitals
    create_all_vitals_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    base_vitals_df = pd.read_parquet(utils.FIRST_48H_VITALS_PATH)
    create_vitals_df(base_vitals_df)
    build_vitals_features(cohort_path=utils.FILTERED_COHORT_PATH)

    # Labs
    create_all_labs_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    base_labs_df = pd.read_parquet(utils.FIRST_48H_LABS_PATH)
    create_labs_df(base_labs_df)
    build_labs_features(cohort_path=utils.FILTERED_COHORT_PATH)

    # GCS
    create_all_gcs_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    base_gcs_df = pd.read_parquet(utils.FIRST_48H_GCS_PATH)
    create_gcs_df(base_gcs_df)
    build_gcs_features(cohort_path=utils.FILTERED_COHORT_PATH)

    # Prescriptions
    extract_prescriptions_first_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)
    build_prescriptions_features(cohort_path=utils.FILTERED_COHORT_PATH)

    # Embeddings
    build_embeddings_features(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)

    # Join all features into one df
    X_df = _build_inference_features().copy()

    # Make sure the subject_id column exists and it is ints.
    if "subject_id" not in X_df.columns:
        raise RuntimeError("Assembled features are missing 'subject_id'.")
    X_df["subject_id"] = X_df["subject_id"].astype(int)

    # Load artifacts and predict for each target
    base = Path(__file__).resolve().parent / "reports"

    mort_dir = base / "mortality"
    los_dir = base / "prolonged_stay"
    readm_dir = base / "readmission"

    mortality_p = _predict_with_bundle(mort_dir, X_df)
    prolonged_p = _predict_with_bundle(los_dir, X_df)
    readmit_p = _predict_with_bundle(readm_dir, X_df)

    # Build the output dataframe
    out = pd.DataFrame(
        {
            "subject_id": X_df["subject_id"].astype(int).to_numpy(),
            "mortality_proba": mortality_p.astype(float),
            "prolonged_LOS_proba": prolonged_p.astype(float),
            "readmission_proba": readmit_p.astype(float),
        }
    )

    return out
