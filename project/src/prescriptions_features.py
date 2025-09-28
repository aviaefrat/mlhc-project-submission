import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils

LOG = utils.make_logger("prescriptions_features", "prescriptions_features.log")

def _std_txt(s: pd.Series) -> pd.Series:
    out = pd.Series(s, dtype="string")
    out = out.str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    return out

def _cap_hours(x: pd.Series, cap: float = 48.0) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").clip(lower=0, upper=cap)

def _compile(rx: str) -> re.Pattern:
    return re.compile(rx, flags=re.IGNORECASE)

# medication families
FAMILY_RX: dict[str, list[re.Pattern]] = {

    "antibiotic": [
        _compile(r"\bVANCOMYCIN\b"),
        _compile(r"\bCEF[AOIU]\w+"),
        _compile(r"\bPIPERACILLIN\b"), _compile(r"\bTAZOBACTAM\b"),
        _compile(r"\bLEVOFLOXACIN\b"), _compile(r"\bCIPROFLOXACIN\b"),
        _compile(r"\bMEROPENEM\b"), _compile(r"\bERTAPENEM\b"),
        _compile(r"\bAZITHROMYCIN\b"),
    ],

    "insulin": [_compile(r"\bINSULIN\b"), _compile(r"SLIDING SCALE")],

    "diuretic_loop": [_compile(r"\bFUROSEMIDE\b")],
    "statin": [_compile(r"\b(ATORVASTATIN|SIMVASTATIN|ROSUVASTATIN|PRAVASTATIN)\b")],
    "beta_blocker": [_compile(r"\bMETOPROLOL\b")],
    "anticoagulant": [_compile(r"\bHEPARIN\b"), _compile(r"\bENOXAPARIN\b"), _compile(r"\bWARFARIN\b")],
    "antiplatelet": [_compile(r"\bASPIRIN\b"), _compile(r"\bCLOPIDOGREL\b")],

    "ppi": [_compile(r"\b(PANTOPRAZOLE|OMEPRAZOLE|ESOMEPRAZOLE|LANSOPRAZOLE)\b")],
    "h2_blocker": [_compile(r"\b(RANITIDINE|FAMOTIDINE)\b")],
    "laxative": [_compile(r"\b(DOCUSATE|SENNA|BISACODYL|POLYETHYLENE GLYCOL|MILK OF MAGNESIA)\b")],

    "opioid": [_compile(r"\b(MORPHINE|HYDROMORPHONE|FENTANYL|OXYCODONE)\b")],
    "benzodiazepine": [_compile(r"\b(LORAZEPAM|MIDAZOLAM|DIAZEPAM)\b")],
    "bronchodilator": [_compile(r"\b(ALBUTEROL|IPRATROPIUM)\b")],
}

# "routes"
IV_ROUTES = {"IV", "IV DRIP", "IVPCA", "IV BOLUS"}
PO_ROUTES = {"PO", "ORAL", "PO/NG", "NG"}

def _match_any(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)

from typing import Iterable, Tuple

def _union_duration_hours(starts: Iterable[pd.Timestamp],
                          ends: Iterable[pd.Timestamp]) -> float:
    intervals: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in zip(starts, ends):
        if pd.isna(s) or pd.isna(e):
            continue
        if e <= s:
            continue
        intervals.append((pd.Timestamp(s), pd.Timestamp(e)))
    if not intervals:
        return 0.0

    intervals.sort(key=lambda x: x[0])
    total = pd.Timedelta(0)
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]: # overlap/contiguous
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += (cur_e - cur_s)
            cur_s, cur_e = s, e
    total += (cur_e - cur_s)
    return float(total / pd.Timedelta(hours=1))


def _family_hits(row: pd.Series) -> dict[str, int]:
    name = row.get("std_generic")
    if pd.isna(name) or not name:
        name = row.get("std_drug")
    if pd.isna(name) or not name:
        return {fam: 0 for fam in FAMILY_RX.keys()}
    s = str(name)
    return {fam: int(_match_any(pats, s)) for fam, pats in FAMILY_RX.items()}

def _family_hours(row: pd.Series) -> dict[str, float]:
    """
    Exposure hours per family
    """
    hrs = float(row.get("overlap_hours_0_48h") or 0.0)
    hrs = min(max(hrs, 0.0), 48.0)
    name = row.get("std_generic")
    if pd.isna(name) or not name:
        name = row.get("std_drug")
    if pd.isna(name) or not name:
        return {f"{fam}_hours_0_48h": 0.0 for fam in FAMILY_RX.keys()}
    s = str(name)
    return {f"{fam}_hours_0_48h": (hrs if _match_any(pats, s) else 0.0) for fam, pats in FAMILY_RX.items()}


def build_prescriptions_features(cohort_path: Path = None, save: bool = True) -> pd.DataFrame:
    """
    Build prescriptions features from data/prescriptions_first_48h.parquet and cohort_path.

    Output:
      - n_orders_0_48h, n_unique_generic_0_48h, n_unique_route_0_48h
      - n_iv_orders_0_48h, n_po_orders_0_48h, frac_iv_orders_0_48h
      - iv_any_0_48h, po_any_0_48h, no_meds_0_48h
      - <family>_any_0_48h flags
      - <family>_hours_0_48h
      - <family>_coverage_0_48h (clip(sum_hours, 48)/48)
    """
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    cohort = pd.read_parquet(cohort_path).loc[:, ["subject_id", "admission_id"]].drop_duplicates()
    cohort = cohort.astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})
    base = cohort.copy()

    if not utils.FIRST_48H_PRESCRIPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{utils.FIRST_48H_PRESCRIPTIONS_PATH} not found. Run `python -m src.prescriptions` first to create it."
        )
    df = pd.read_parquet(utils.FIRST_48H_PRESCRIPTIONS_PATH)

    # Normalize text
    for src, dst in [("drug_name_generic", "std_generic"), ("drug", "std_drug"), ("route", "std_route")]:
        df[dst] = _std_txt(df[src]) if src in df.columns else pd.Series(pd.NA, dtype="string")

    # routes
    df["route_iv_flag"] = df["std_route"].isin(IV_ROUTES).astype("int8")
    df["route_po_flag"] = df["std_route"].isin(PO_ROUTES).astype("int8")
    df["overlap_hours_0_48h"] = _cap_hours(df.get("overlap_hours_0_48h", 0.0))

    # prescriptions families
    fam_hit_df = df.apply(_family_hits, axis=1, result_type="expand")
    fam_hrs_df = df.apply(_family_hours, axis=1, result_type="expand")
    df = pd.concat([df, fam_hit_df, fam_hrs_df], axis=1)

    # aggregate to admission level
    grp = df.groupby(["subject_id", "admission_id"], observed=True)
    agg = pd.DataFrame(index=grp.size().index).reset_index()

    # Core counts
    agg["n_orders_0_48h"] = grp.size().astype("int32").values
    agg["n_unique_generic_0_48h"] = grp["std_generic"].nunique(dropna=True).astype("int32").values
    agg["n_unique_route_0_48h"] = grp["std_route"].nunique(dropna=True).astype("int32").values
    agg["n_iv_orders_0_48h"] = grp["route_iv_flag"].sum().astype("int32").values
    agg["n_po_orders_0_48h"] = grp["route_po_flag"].sum().astype("int32").values

    denom = agg["n_orders_0_48h"].replace({0: pd.NA})
    agg["frac_iv_orders_0_48h"] = (agg["n_iv_orders_0_48h"] / denom).fillna(0.0).astype("float32")
    agg["iv_any_0_48h"] = (agg["n_iv_orders_0_48h"] > 0).astype("int8")
    agg["po_any_0_48h"] = (agg["n_po_orders_0_48h"] > 0).astype("int8")
    agg["no_meds_0_48h"] = (agg["n_orders_0_48h"] == 0).astype("int8")

    # make sure we have the columns we need to continue
    if ("overlap_start" not in df.columns) or ("overlap_end" not in df.columns):
        raise RuntimeError(
            "prescriptions_first_48h must contain overlap_start and overlap_end timestamps for union coverage.")

    for fam in FAMILY_RX.keys():
        flag_col = f"{fam}_any_0_48h"
        hrs_col = f"{fam}_hours_0_48h"
        cov_col = f"{fam}_coverage_0_48h"

        # any
        agg[flag_col] = grp[fam].max().astype("int8").values

        # total hours
        hrs = grp[hrs_col].sum().astype("float32").values
        agg[hrs_col] = hrs

        # coverage (overlapping hours do not count extrea here)
        fam_rows = df[df[fam] == 1].loc[:, ["subject_id", "admission_id", "overlap_start", "overlap_end"]]
        if len(fam_rows):
            cov_union = (
                fam_rows
                .groupby(["subject_id", "admission_id"], observed=True)[["overlap_start", "overlap_end"]]
                .apply(lambda g: _union_duration_hours(g["overlap_start"], g["overlap_end"]))
                .reset_index(name=cov_col)
            )

            cov_union[cov_col] = np.minimum(cov_union[cov_col].astype("float32"), 48.0) / 48.0
        else:
            cov_union = pd.DataFrame(columns=["subject_id", "admission_id", cov_col])

        agg = agg.merge(cov_union, on=["subject_id", "admission_id"], how="left")
        agg[cov_col] = agg[cov_col].fillna(0.0).astype("float32")

    # sanity check: the coverage should be [0,1] and never exceed total hours/48
    for fam in FAMILY_RX.keys():
        hrs_col = f"{fam}_hours_0_48h"
        cov_col = f"{fam}_coverage_0_48h"
        if cov_col in agg and hrs_col in agg:
            assert (agg[cov_col] >= 0).all() and (agg[cov_col] <= 1).all()
            # allow a tiny epsilon for float error
            assert (agg[cov_col] <= (agg[hrs_col] / 48.0) + 1e-6).all()

    # Join to cohort
    out = base.merge(agg, on=["subject_id", "admission_id"], how="left")

    count_cols = [
        "n_orders_0_48h", "n_unique_generic_0_48h", "n_unique_route_0_48h",
        "n_iv_orders_0_48h", "n_po_orders_0_48h"
    ]
    flag_cols = ["iv_any_0_48h", "po_any_0_48h", "no_meds_0_48h"] + [f"{fam}_any_0_48h" for fam in FAMILY_RX.keys()]
    hour_cols = [f"{fam}_hours_0_48h" for fam in FAMILY_RX.keys()]
    cov_cols  = [f"{fam}_coverage_0_48h" for fam in FAMILY_RX.keys()]

    for c in count_cols:
        out[c] = out[c].fillna(0).astype("int32")
    for c in flag_cols:
        out[c] = out[c].fillna(0).astype("int8")
    for c in hour_cols + ["frac_iv_orders_0_48h"] + cov_cols:
        out[c] = out[c].fillna(0.0).astype("float32")

    out = out.astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})

    LOG.info("Prescriptions features: rows=%d, cols=%d", len(out), out.shape[1])
    if save:
        utils.PRESCRIPTIONS_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(utils.PRESCRIPTIONS_FEATURES_PATH, index=False)
        LOG.info("Wrote prescriptions features to: %s", utils.PRESCRIPTIONS_FEATURES_PATH)
    return out


if __name__ == "__main__":
    build_prescriptions_features(cohort_path=utils.FILTERED_COHORT_PATH)
