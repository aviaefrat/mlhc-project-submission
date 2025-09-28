from pathlib import Path
import numpy as np
import pandas as pd

from . import utils

LOG = utils.make_logger("noteevents_features", "noteevents_features.log")


CATEGORIES = [
    "Radiology",
    "Nursing/other",
    "Nursing",
    "ECG",
    "Physician ",
    "Echo",
    "Respiratory ",
    "General",
    "Nutrition",
    "Social Work",
    "Rehab Services",
    "Case Management ",
]


DESCRIPTIONS = [
    "Physician Resident Progress Note",
    "Intensivist Note",
    "Physician Resident Admission Note",
    "Physician Attending Progress Note",
    "Physician Attending Admission Note - MICU",
    "ICU Note - CVI",
    "Physician Surgical Admission Note",
    "Physician Attending Admission Note",
    "Physician Resident/Attending Progress Note - MICU",
    "Physician Resident/Attending Admission Note - MICU",
    "Physician Fellow / Attending Progress Note - MICU",
    "Physician Surgical Progress Note",
    "Cardiology Physician Note",
    "Cardiology Comprehensive Physician Note",
    "Physician Fellow/Attending Progress Note - MICU",
    "CHEST (PORTABLE AP)",
    "CT HEAD W/O CONTRAST",
    "CHEST PORT. LINE PLACEMENT",
    "CHEST (PRE-OP PA & LAT)",
    "CHEST (PA & LAT)",
    "MR HEAD W & W/O CONTRAST",
    "CAROTID SERIES COMPLETE",
    "PORTABLE ABDOMEN",
    "CT CHEST W/O CONTRAST",
    "MR HEAD W/O CONTRAST",
    "CT CHEST W/CONTRAST",
    "BY DIFFERENT PHYSICIAN",
    "CT ABDOMEN W/O CONTRAST",
    "CTA CHEST W&W/O C&RECONS, NON-CORONARY",
    "CTA HEAD W&W/O C & RECONS",
    "CT ABDOMEN W/CONTRAST",
    "BY SAME PHYSICIAN",
    "BILAT LOWER EXT VEINS",
    "CT C-SPINE W/O CONTRAST",
    "LIVER OR GALLBLADDER US (SINGLE ORGAN)",
    "ABDOMEN U.S. (COMPLETE STUDY)",
    "MR CERVICAL SPINE W/O CONTRAST",
    "RENAL U.S.",
    "P LIVER OR GALLBLADDER US (SINGLE ORGAN) PORT",
    "CT ABD W&W/O C",
    "P RENAL U.S. PORT",
    "SEL CATH 3RD ORDER THOR",
    "P BILAT LOWER EXT VEINS PORT",
    "CTA CHEST W&W/O C &RECONS",
    "MR CERVICAL SPINE",
    "T-SPINE",
    "P CAROTID SERIES COMPLETE PORT",
    "VEN DUP EXTEXT BIL (MAP/DVT)",
    "DISTINCT PROCEDURAL SERVICE",
    "CTA ABD W&W/O C & RECONS",
    "CHEST (SINGLE VIEW)",
    "CT SINUS/MANDIBLE/MAXILLOFACIAL W/O CONTRAST",
    "US ABD LIMIT, SINGLE ORGAN",
    "CT T-SPINE W/O CONTRAST",
    "EMBO TRANSCRANIAL",
    "DUPLEX DOPP ABD/PEL",
    "CT L-SPINE W/O CONTRAST",
    "CT PELVIS W/CONTRAST",
    "ABDOMEN (SUPINE & ERECT)",
    "EMBO NON NEURO",
    "CT PELVIS W/O CONTRAST",
    "P ABDOMEN U.S. (COMPLETE STUDY) PORT",
    "MR HEAD W/ CONTRAST",
    "MR C-SPINE W& W/O CONTRAST",
    "CT HEAD W/ & W/O CONTRAST",
    "O L-SPINE (AP & LAT) IN O.R.",
    "CTA NECK W&W/OC & RECONS",
    "P ABDOMEN (SUPINE ONLY) PORT",
    "ERCP BILIARY&PANCREAS BY GI UNIT",
    "Nursing Progress Note",
    "Nursing Transfer Note",
]

def _as_int32(s: pd.Series) -> pd.Series:
    # Ensure pandas Int32 with zeros instead of NA
    return s.fillna(0).astype("Int32")

def _as_int16(s: pd.Series) -> pd.Series:
    # Ensure pandas Int16 with zeros filled beforehand
    return s.astype("Int16")

def build_noteevents_features(cohort_path: Path = None) -> pd.DataFrame:
    """
    Build noteevents features for every subject_id in cohort_path, using the file at
    utils.FIRST_48H_NOTEEVENTS_PATH.

    Outputs to utils.NOTEEVENTS_FEATURES_PATH:
      - subject_id (int32)
      - n_total_notes (Int32)
      - n_{CATEGORY}_notes for each CATEGORY in CATEGORIES (Int32)
      - first_note_category (string) [earliest non-NA charttime, else 'NO_CATEGORY']
      - full_hours_until_first_note (Int16) [floor(min hours_since_admit) if any, else 48]
      - has_{DESCRIPTION} (boolean) for each DESCRIPTION in DESCRIPTIONS (pandas BooleanDtype)
    """
    out_path: Path = utils.NOTEEVENTS_FEATURES_PATH

    # loqad cohort
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    cohort = pd.read_parquet(cohort_path, columns=["subject_id"]).copy()
    cohort["subject_id"] = cohort["subject_id"].astype(utils.ID_DTYPE)
    cohort = cohort.drop_duplicates().sort_values("subject_id").reset_index(drop=True)
    n_subjects = len(cohort)

    # Load noteevents (first 48h) for these subjects
    cols_needed = ["subject_id", "category", "description", "charttime", "hours_since_admit"]
    ne = pd.read_parquet(utils.FIRST_48H_NOTEEVENTS_PATH, columns=cols_needed).copy()
    ne["subject_id"] = ne["subject_id"].astype(utils.ID_DTYPE)

    LOG.info(f"Building noteevents features for {n_subjects} cohort subjects from {utils.FIRST_48H_NOTEEVENTS_PATH.name}")
    LOG.info("Raw notes rows: %d", len(ne))


    total_counts = ne.groupby("subject_id").size().rename("n_total_notes")
    total_counts = total_counts.reindex(cohort["subject_id"], fill_value=0)
    total_counts = _as_int32(total_counts)

    cat_counts = (
        ne[["subject_id", "category"]]
        .groupby(["subject_id", "category"])
        .size()
        .unstack(fill_value=0)
    )
    # ensure all 12 category columns exist, ordered as in CATEGORIES
    for cat in CATEGORIES:
        if cat not in cat_counts.columns:
            cat_counts[cat] = 0
    cat_counts = cat_counts[CATEGORIES]
    cat_counts = cat_counts.reindex(cohort["subject_id"], fill_value=0)
    cat_counts = cat_counts.apply(_as_int32)

    # keep only rows with non-NA charttime to determine the "first" note
    timed = ne.dropna(subset=["charttime"]).copy()
    if not timed.empty:
        timed["charttime"] = pd.to_datetime(timed["charttime"], errors="coerce", utc=False)
        timed = timed.dropna(subset=["charttime"])

        # deterministic: earliest by charttime, then by category for tie-breaks
        timed = timed.sort_values(["subject_id", "charttime", "category"])

        first_timed = (
            timed.drop_duplicates(subset=["subject_id"], keep="first")
                 .loc[:, ["subject_id", "category", "hours_since_admit"]]
                 .rename(columns={"category": "first_note_category"})
        )
    else:
        first_timed = pd.DataFrame(columns=["subject_id", "first_note_category", "hours_since_admit"])

    first_join = cohort.merge(first_timed, on="subject_id", how="left")

    # first_note_category: NO_CATEGORY if no timed note
    first_note_category = (
        first_join["first_note_category"]
        .astype("string")
        .fillna("NO_CATEGORY")
    )

    # full_hours_until_first_note:
    # use hours_since_admit from the earliest timed note. if NA, set 48
    full_hours_until_first_note = (
        np.floor(first_join["hours_since_admit"].astype("float64"))
        .clip(lower=0, upper=48)
    )
    full_hours_until_first_note = (
        pd.Series(full_hours_until_first_note)
        .fillna(48)
        .astype("Int16")
    )

    n_no_category = int((first_note_category == "NO_CATEGORY").sum())
    LOG.info("Subjects with NO_CATEGORY as first_note_category: %d / %d", n_no_category, n_subjects)


    has_desc_df = pd.DataFrame(index=cohort["subject_id"])
    for desc in DESCRIPTIONS:
        col = f"has_{desc}"
        mask = (ne["description"] == desc)
        present = mask.groupby(ne["subject_id"]).any()
        present = present.reindex(cohort["subject_id"], fill_value=False)
        has_desc_df[col] = present.astype("boolean")


    out = pd.DataFrame({"subject_id": cohort["subject_id"].values.astype(utils.ID_DTYPE)})
    out["n_total_notes"] = total_counts.values.astype("Int32")

    # add per-category counts
    for cat in CATEGORIES:
        out[f"n_{cat}_notes"] = cat_counts[cat].values.astype("Int32")

    # first-note features
    out["first_note_category"] = first_note_category
    out["full_hours_until_first_note"] = full_hours_until_first_note

    # Add description booleans
    for desc in DESCRIPTIONS:
        col = f"has_{desc}"
        out[col] = has_desc_df[col].astype("boolean").values

    # logging statistics
    zero_notes = int((out["n_total_notes"] == 0).sum())
    LOG.info("Subjects with zero notes in the 48h window: %d / %d", zero_notes, n_subjects)
    if len(out) != n_subjects:
        LOG.warning("Feature rows (%d) != cohort subjects (%d). Check joins.", len(out), n_subjects)

    try:
        q = out.loc[out["n_total_notes"] > 0, "full_hours_until_first_note"].astype("Int16")
        LOG.info(
            "full_hours_until_first_note quantiles (subjects with ≥1 timed note): min=%d, p25=%d, p50=%d, p75=%d, max=%d",
            int(q.min()), int(q.quantile(0.25)), int(q.median()), int(q.quantile(0.75)), int(q.max())
        )
    except Exception:
        LOG.info("Could not compute quantiles for full_hours_until_first_note (possibly no timed notes).")

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    LOG.info("Wrote noteevents features to: %s [rows=%d, cols=%d]", out_path, len(out), out.shape[1])

    return out


if __name__ == "__main__":
    build_noteevents_features(cohort_path=utils.FILTERED_COHORT_PATH)