import pandas as pd

from . import utils

LOGGER = utils.make_logger("data_build", "data_build.log")


def _enforce_ids_type(df: pd.DataFrame) -> pd.DataFrame:
    if "admission_id" in df.columns:
        df["admission_id"] = df["admission_id"].astype(utils.ID_DTYPE)
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(utils.ID_DTYPE)
    return df


def _get_keys_to_join_on(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """ prefer ['admission_id','subject_id'] when both present, else whichever exists."""
    keys = []
    for k in ["subject_id", "admission_id"]:
        if (k in left.columns) and (k in right.columns):
            keys.append(k)
    if not keys:
        raise ValueError("No common join keys between frames.")
    return keys


def load_and_merge(
    include_demographics: bool = True,
    include_vitals: bool = True,
    include_labs: bool = True,
    include_gcs: bool = True,
    include_prescriptions: bool = True,
    include_noteevents: bool = False,
    include_embeddings: bool = True,
    target: str = "",
) -> pd.DataFrame:
    """
    Build the modeling table by joining any of the features DEMOGRAPHICS, VITALS_FEATURES, LABS_FEATURES,
    GCS_FEATURES, PRESCRIPTIONS_FEATURES, NOTEEVENTS_FEATURES, and EMBEDDINGS_FEATURES. then join the chosen label.

    Returns:
        pd.DataFrame with one row per (subject_id, admission_id) with to the chosen label.
    """
    frames: list[tuple[str, pd.DataFrame]] = []

    if include_demographics:
        demo = _enforce_ids_type(pd.read_parquet(utils.DEMOGRAPHICS_PATH))
        frames.append(("demographics", demo))
        LOGGER.info(f"Loaded DEMOGRAPHICS: {len(demo):,} rows, {demo.shape[1]:,} cols")

    if include_vitals:
        vit = _enforce_ids_type(pd.read_parquet(utils.VITALS_FEATURES_PATH))
        frames.append(("vitals_features", vit))
        LOGGER.info(f"Loaded VITALS_FEATURES: {len(vit):,} rows, {vit.shape[1]:,} cols")

    if include_labs:
        labs = _enforce_ids_type(pd.read_parquet(utils.LABS_FEATURES_PATH))
        frames.append(("labs_features", labs))
        LOGGER.info(f"Loaded LABS_FEATURES: {len(labs):,} rows, {labs.shape[1]:,} cols")

    if include_gcs:
        gcs = _enforce_ids_type(pd.read_parquet(utils.GCS_FEATURES_PATH))
        frames.append(("gcs_features", gcs))
        LOGGER.info(f"Loaded GCS_FEATURES: {len(gcs):,} rows, {gcs.shape[1]:,} cols")

    if include_prescriptions:
        prescriptions = _enforce_ids_type(pd.read_parquet(utils.PRESCRIPTIONS_FEATURES_PATH))
        frames.append(("prescriptions_features", prescriptions))
        LOGGER.info(f"Loaded PRESCRIPTIONS_FEATURES: {len(prescriptions):,} rows, {prescriptions.shape[1]:,} cols")

    if include_noteevents:
        noteevents = _enforce_ids_type(pd.read_parquet(utils.NOTEEVENTS_FEATURES_PATH))
        frames.append(("noteevents_features", noteevents))
        LOGGER.info(f"Loaded NOTEEVENTS_FEATURES: {len(noteevents):,} rows, {noteevents.shape[1]:,} cols")

    if include_embeddings:
        embeddings = _enforce_ids_type(pd.read_parquet(utils.EMBEDDINGS_FEATURES_PATH))
        frames.append(("ebeddings_features", embeddings))
        LOGGER.info(f"Loaded EMBEDDINGS_FEATURES: {len(embeddings):,} rows, {embeddings.shape[1]:,} cols")

    if not frames:
        raise ValueError("No feature sources selected. Include at least one feature table.")

    # Iteratively join the features we asked to use (based on subject_id and/or admission_id)
    name0, merged = frames[0]
    for name, df in frames[1:]:
        keys_to_join_features_on = _get_keys_to_join_on(merged, df)
        merged = merged.merge(df, on=keys_to_join_features_on, how="inner", validate="one_to_one")
        LOGGER.info(f"Merged {name0}+{name}: {len(merged):,} rows, {merged.shape[1]:,} cols")
        name0 = f"{name0}+{name}"

    label_spec = {
        "mortality": (utils.MORTALITY_LABEL_PATH, "mortality"),
        "prolonged_stay": (utils.PROLONGED_STAY_LABEL_PATH, "prolonged_stay"),
        "readmission": (utils.READMISSION_LABEL_PATH, "readmission"),
    }
    if target not in label_spec:
        raise ValueError(f"Unknown target '{target}'. Expected one of {list(label_spec.keys())}.")

    label_path, label_name = label_spec[target]
    labels_df = _enforce_ids_type(pd.read_parquet(label_path))

    keys_to_join_label_to_features_on = _get_keys_to_join_on(merged, labels_df)

    LOGGER.info(f"Joining label '{label_name}' using keys: {keys_to_join_label_to_features_on}")
    merged = merged.merge(labels_df[keys_to_join_label_to_features_on + [label_name]], on=keys_to_join_label_to_features_on, how="inner", validate="one_to_one")

    if merged.duplicated(subset=keys_to_join_label_to_features_on).any():
        raise ValueError(f"Merged table contains duplicate rows after label join on keys={keys_to_join_label_to_features_on}.")

    LOGGER.info(
        f"Merged design matrix (features + {label_name}): {len(merged):,} rows, {merged.shape[1]:,} columns. "
        f"[features include: demographics={include_demographics}, vitals={include_vitals}, labs={include_labs}, gcs={include_gcs}, prescriptions={include_prescriptions}, notes={include_embeddings}]"
    )
    return merged
