from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import utils

LOG = utils.make_logger("gcs_features", "gcs_features.log")

EPS = 1e-8
TOP_N_FEATURES = 10  # how many features to list in the "most measured" summary

WINDOW_EDGES = [0.0, 12.0, 24.0, 36.0, 48.0]
WINDOW_LABELS = pd.CategoricalDtype(categories=[0, 1, 2, 3], ordered=True)


@dataclass(frozen=True)
class WindowedStats:
    n: int | float
    min: float | None
    max: float | None
    mean: float | None
    std: float | None  # n==0 -> NA; n==1 -> 0; n>=2 -> sample std (ddof=1)


def _assign_window(hours_since_admit: pd.Series) -> pd.Categorical:
    """Map hours [0,48) into windows {0,1,2,3}."""
    out = pd.cut(
        hours_since_admit,
        bins=WINDOW_EDGES,
        right=False,
        labels=[0, 1, 2, 3],
        include_lowest=True,
        ordered=True,
    )
    return out.astype(WINDOW_LABELS)


def _load_inputs(cohort_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:

    # Cleaned GCS table (already filtered/cleaned in gcs.py and includes 'feature_name')
    gcs_df = pd.read_parquet(utils.GCS_PATH).loc[
        :, ["subject_id", "admission_id", "feature_name", "charttime", "valuenum"]
    ].copy()
    gcs_df["charttime"] = pd.to_datetime(gcs_df["charttime"], errors="coerce")
    gcs_df["feature_name"] = gcs_df["feature_name"].astype("string")

    # Cohort with admission times
    cohort_df = pd.read_parquet(cohort_path).loc[
        :, ["subject_id", "admission_id", "admittime"]
    ].copy()
    cohort_df = cohort_df.astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})
    cohort_df["admittime"] = pd.to_datetime(cohort_df["admittime"], errors="coerce")

    # 'canonical' feature name from GCS_CSV
    csv = pd.read_csv(utils.GCS_CSV)
    if "feature name" not in csv.columns:
        raise ValueError(f"{utils.GCS_CSV} must have a 'feature name' column")
    feature_names_all = (
        csv["feature name"].astype("string").drop_duplicates().tolist()
    )

    return gcs_df, cohort_df, feature_names_all


def _constrain_to_first_48h(gcs_df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:

    df = gcs_df.merge(
        cohort_df, on=["admission_id", "subject_id"], how="inner", validate="many_to_one"
    )
    hours = (df["charttime"] - df["admittime"]) / pd.Timedelta(hours=1)
    df["hours_since_admit"] = hours.astype("float")

    mask = df["hours_since_admit"].between(WINDOW_EDGES[0], WINDOW_EDGES[-1], inclusive="left")
    if not mask.all():
        LOG.info(f"Dropping {int((~mask).sum()):,} row(s) outside [0,48) despite prior filtering.")
    df = df.loc[mask].copy()

    df["window"] = _assign_window(df["hours_since_admit"])
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


def _compute_admission_level_features(
        df48: pd.DataFrame,
        feature_names_all: list[str],
        cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (subject_id, feature_name), compute:
      - n_admission_measurements
      - first_admission_measurement
      - last_admission_measurement
      - difference_last_first_admission_measurements = last - first
    """
    nonna = df48.loc[
        df48["feature_name"].notna() & df48["valuenum"].notna(),
        ["subject_id", "feature_name", "charttime", "valuenum"],
    ].copy()

    cnt = (
        nonna.groupby(["subject_id", "feature_name"], observed=True)["valuenum"]
        .size()
        .rename("n_admission_measurements")
        .reset_index()
    )

    first_idx = (
        nonna.sort_values(["subject_id", "feature_name", "charttime"], kind="mergesort")
        .groupby(["subject_id", "feature_name"], observed=True, sort=False)
        .head(1)
    )
    first = first_idx.loc[:, ["subject_id", "feature_name", "valuenum"]].rename(
        columns={"valuenum": "first_admission_measurement"}
    )

    last_idx = (
        nonna.sort_values(["subject_id", "feature_name", "charttime"], kind="mergesort")
        .groupby(["subject_id", "feature_name"], observed=True, sort=False)
        .tail(1)
    )
    last = last_idx.loc[:, ["subject_id", "feature_name", "valuenum"]].rename(
        columns={"valuenum": "last_admission_measurement"}
    )

    subjects = cohort_df["subject_id"]

    base = (
        subjects.to_frame(name="subject_id")
        .assign(key=1)
        .merge(pd.DataFrame({"feature_name": feature_names_all, "key": 1}), on="key", how="left")
        .drop(columns=["key"])
    )

    adm = (
        base.merge(cnt, on=["subject_id", "feature_name"], how="left")
            .merge(first, on=["subject_id", "feature_name"], how="left")
            .merge(last, on=["subject_id", "feature_name"], how="left")
    )

    adm["n_admission_measurements"] = adm["n_admission_measurements"].fillna(0).astype("int64")
    adm["difference_last_first_admission_measurements"] = (
        adm["last_admission_measurement"] - adm["first_admission_measurement"]
    )

    wide = {}
    for col in [
        "n_admission_measurements",
        "first_admission_measurement",
        "last_admission_measurement",
        "difference_last_first_admission_measurements",
    ]:
        tmp = adm.pivot(index="subject_id", columns="feature_name", values=col)
        tmp.columns = [f"{c}_{col}" for c in tmp.columns]
        wide[col] = tmp

    out = pd.concat(wide.values(), axis=1).reset_index()
    return out


def _window_agg(group: pd.Series) -> WindowedStats:
    """Aggregate a single column 'valuenum' within a (subject_id, feature_name, window) group."""
    n = int(group.size)
    if n == 0:
        return WindowedStats(n=0, min=pd.NA, max=pd.NA, mean=pd.NA, std=pd.NA)
    arr = group.to_numpy(dtype=float, copy=False)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    vmean = float(np.mean(arr))
    if n == 1:
        vstd = 0.0
    else:
        vstd = float(np.std(arr, ddof=1))
    return WindowedStats(n=n, min=vmin, max=vmax, mean=vmean, std=vstd)


def _compute_per_window_long(df48: pd.DataFrame, feature_names_all: list[str]) -> pd.DataFrame:
    """
    Return long-form per-window stats with one row per (subject_id, feature_name, window) and columns: n, min, max, mean, std.
    """
    work = df48.loc[
        df48["feature_name"].notna() & df48["valuenum"].notna(),
        ["subject_id", "feature_name", "window", "valuenum"],
    ].copy()

    agg = (
        work.groupby(["subject_id", "feature_name", "window"], observed=True)["valuenum"]
        .apply(_window_agg)
        .reset_index(name="stats")
    )

    stats_df = pd.concat(
        [
            agg.drop(columns=["stats"]),
            pd.DataFrame(agg["stats"].tolist(), index=agg.index),
        ],
        axis=1,
    )

    subjects = df48["subject_id"].drop_duplicates()
    full = (
        subjects.to_frame(name="subject_id")
        .assign(key=1)
        .merge(pd.DataFrame({"feature_name": feature_names_all, "key": 1}), on="key")
        .merge(pd.DataFrame({"window": WINDOW_LABELS.categories, "key": 1}), on="key")
        .drop(columns=["key"])
    )

    full = full.merge(stats_df, on=["subject_id", "feature_name", "window"], how="left")
    full["n"] = full["n"].fillna(0).astype("int64")
    full["window"] = full["window"].astype(WINDOW_LABELS)
    full = full.sort_values(["subject_id", "feature_name", "window"])
    return full


def _ffill_stats_within_feature(full_long: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill min/max/mean/std across windows within each (subject_id, feature_name).
    """
    full_long = full_long.sort_values(["subject_id", "feature_name", "window"], kind="mergesort").copy()
    stat_cols = ["min", "max", "mean", "std"]
    filled = full_long.groupby(["subject_id", "feature_name"], sort=False)[stat_cols].ffill()
    full_long[stat_cols] = filled.values
    return full_long


def _pivot_per_window_wide(ffilled_long: pd.DataFrame) -> pd.DataFrame:
    """
    Produce wide columns for each feature_name and window:
      - {feature_name}_window{i}_n_measurements
      - {feature_name}_window{i}_min / _max / _mean / _std
    """
    pieces = []
    for col, suffix in [
        ("n", "n_measurements"),
        ("min", "min"),
        ("max", "max"),
        ("mean", "mean"),
        ("std", "std"),
    ]:
        p = ffilled_long.pivot_table(
            index="subject_id",
            columns=["feature_name", "window"],
            values=col,
            dropna=False,
            observed=False,
        )
        flat_cols = []
        for (fname, win) in p.columns:
            win = int(win)
            flat_cols.append(f"{fname}_window{win}_{suffix}")
        p.columns = flat_cols
        pieces.append(p)

    wide = pd.concat(pieces, axis=1).reset_index()
    for c in [c for c in wide.columns if c.endswith("_n_measurements")]:
        wide[c] = wide[c].fillna(0).astype("int64")
    return wide


def _compute_between_window_diffs(ffilled_long: pd.DataFrame, feature_names_all: list[str]) -> pd.DataFrame:
    """
    Using ffilled window means, compute for each feature_name and subject_d:
      - diff0/1/2 baseline_mean: w0_mean - w(i+1)_mean
      - diff0/1/2 previous_mean: w(i+1)_mean - w(i)_mean
      - diff0/1/2 sign_mean: POS/NEG/ZERO with (using EPS)
    """
    m = ffilled_long.pivot_table(
        index="subject_id", columns=["feature_name", "window"], values="mean",
        dropna=False, observed=False
    )

    out_blocks = []
    idx = m.index

    for fname in feature_names_all:
        cols = [(fname, 0), (fname, 1), (fname, 2), (fname, 3)]
        Mi = pd.DataFrame({
            w: (m[cols[w]] if (fname, w) in m.columns else pd.Series(np.nan, index=idx))
            for w in range(4)
        })

        blocks = {}
        for i in range(3):
            baseline = Mi[0] - Mi[i + 1]
            previous = Mi[i + 1] - Mi[i]

            sign = pd.Series(pd.NA, index=idx, dtype="string")
            valid = previous.notna()
            if valid.any():
                pos_mask = valid & (previous > EPS)
                neg_mask = valid & (previous < -EPS)
                zero_mask = valid & ~(pos_mask | neg_mask)

                sign.loc[pos_mask] = "POS"
                sign.loc[neg_mask] = "NEG"
                sign.loc[zero_mask] = "ZERO"

            blocks[f"{fname}_diff{i}_baseline_mean"] = baseline
            blocks[f"{fname}_diff{i}_previous_mean"] = previous
            blocks[f"{fname}_diff{i}_sign_mean"] = sign

        out_blocks.append(pd.DataFrame(blocks, index=idx))

    out = pd.concat(out_blocks, axis=1)
    out.reset_index(inplace=True)
    return out


def _stable_column_order(
    adm_wide: pd.DataFrame,
    win_wide: pd.DataFrame,
    diff_wide: pd.DataFrame,
    feature_names_all: Iterable[str],
) -> list[str]:
    cols = ["subject_id", "admission_id"]

    def present(colname: str) -> bool:
        return (colname in adm_wide.columns) or (colname in win_wide.columns) or (colname in diff_wide.columns)

    for fname in feature_names_all:
        for suffix in [
            "n_admission_measurements",
            "first_admission_measurement",
            "last_admission_measurement",
            "difference_last_first_admission_measurements",
        ]:
            name = f"{fname}_{suffix}"
            if present(name):
                cols.append(name)

        for i in range(4):
            for suffix in ["n_measurements", "min", "max", "mean", "std"]:
                name = f"{fname}_window{i}_{suffix}"
                if present(name):
                    cols.append(name)

        for i in range(3):
            for suffix in ["baseline_mean", "previous_mean", "sign_mean"]:
                name = f"{fname}_diff{i}_{suffix}"
                if present(name):
                    cols.append(name)

    existing = ["subject_id"]
    existing += [c for c in cols if c in adm_wide.columns or c in win_wide.columns or c in diff_wide.columns]
    existing = list(dict.fromkeys(existing))
    return existing


def _log_feature_summaries(features: pd.DataFrame, feature_names_all: list[str]) -> None:
    """
    Logs:
      - Top-N features by total n_admission_measurements
      - Per-feature NA fractions for first/last/diff and window means (post-ffill)
    """
    counts = []
    for fname in feature_names_all:
        col = f"{fname}_n_admission_measurements"
        if col in features.columns:
            counts.append((fname, int(features[col].fillna(0).sum())))
        else:
            counts.append((fname, 0))
    counts.sort(key=lambda x: x[1], reverse=True)
    LOG.info(f"Top {TOP_N_FEATURES} features by total measurements in 0–48h:")
    for fname, total in counts[:TOP_N_FEATURES]:
        LOG.info(f"  {fname}: total_n={total:,}")

    def frac_na(col: str) -> float:
        if col not in features.columns:
            return np.nan
        s = features[col]
        return float(s.isna().mean())

    LOG.info("Per-feature NA fractions (first/last/diff; window means after ffill: w0..w3):")
    for fname in feature_names_all:
        f_col = f"{fname}_first_admission_measurement"
        l_col = f"{fname}_last_admission_measurement"
        d_col = f"{fname}_difference_last_first_admission_measurements"
        w_cols = [f"{fname}_window{i}_mean" for i in range(4)]

        f_na = frac_na(f_col)
        l_na = frac_na(l_col)
        d_na = frac_na(d_col)
        w_na = [frac_na(c) for c in w_cols]

        LOG.info(
            "  %s: NA(first)=%.3f, NA(last)=%.3f, NA(diff)=%.3f, NA(mean by window)=%s",
            fname, f_na, l_na, d_na,
            "[" + ", ".join(f"{x:.3f}" if not np.isnan(x) else "nan" for x in w_na) + "]"
        )


def build_gcs_features(cohort_path: Path = None, save: bool = True) -> pd.DataFrame:
    """
    Build GCS features from utils.GCS_PATH and cohort_path.

    Output:
      - One row per subject_id (admission_id also included)
      - Admission-level features per feature_name:
          {feature_name}_n_admission_measurements
          {feature_name}_first_admission_measurement
          {feature_name}_last_admission_measurement
          {feature_name}_difference_last_first_admission_measurements
      - Per-window features (w0..w3):
          {feature_name}_window{i}_n_measurements
          {feature_name}_window{i}_min / _max / _mean / _std
        Stats are forward-filled across windows within (subject_id,feature_name).
        Std rule: n==0 -> NA; n==1 -> 0; n>=2 -> sample std (ddof=1)
      - Between-window features (computed on ffilled means):
          {feature_name}_diff{i}_baseline_mean
          {feature_name}_diff{i}_previous_mean
          {feature_name}_diff{i}_sign_mean in {POS,NEG,ZERO} with EPS tolerance
    """
    LOG.info("Building GCS features keyed by feature_name (not itemid).")
    LOG.info("Loading inputs...")
    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    gcs_df, cohort_df, feature_names_all = _load_inputs(cohort_path)

    na_name_rows = int((gcs_df["feature_name"].isna()).sum())
    if na_name_rows:
        LOG.warning(f"Rows with NA feature_name in GCS_PATH: {na_name_rows} (excluded from aggregations)")

    LOG.info("Constraining GCS to [0,48) and assigning windows...")
    df48 = _constrain_to_first_48h(gcs_df, cohort_df)

    LOG.info("Computing admission-level features...")
    adm_wide = _compute_admission_level_features(df48, feature_names_all, cohort_df)

    LOG.info("Computing per-window long-form aggregates...")
    long = _compute_per_window_long(df48, feature_names_all)

    LOG.info("Forward-filling stats (min/max/mean/std) across windows within (subject_id,feature_name)...")
    long_ff = _ffill_stats_within_feature(long)

    LOG.info("Pivoting per-window features to wide form...")
    win_wide = _pivot_per_window_wide(long_ff)

    LOG.info("Computing between-window diffs on ffilled means...")
    diff_wide = _compute_between_window_diffs(long_ff, feature_names_all)

    LOG.info("Merging all feature blocks...")
    base = cohort_df.loc[:, ["subject_id", "admission_id"]].drop_duplicates().copy()
    features = (
        base.merge(adm_wide, on="subject_id", how="left")
            .merge(win_wide, on="subject_id", how="left")
            .merge(diff_wide, on="subject_id", how="left")
    )

    ordered = _stable_column_order(adm_wide, win_wide, diff_wide, feature_names_all)
    cols_final = ["subject_id", "admission_id"] + [c for c in ordered if c not in ("subject_id", "admission_id")]
    cols_final = [c for c in cols_final if c in features.columns]
    features = features.loc[:, cols_final]

    features = features.astype({"subject_id": utils.ID_DTYPE, "admission_id": utils.ID_DTYPE})
    # Fill NA counts to 0 for all *_n_measurements and *_n_admission_measurements
    count_like_columns = [c for c in features.columns
                  if c.endswith("_n_measurements") or c.endswith("_n_admission_measurements")]
    if count_like_columns:
        features[count_like_columns] = features[count_like_columns].fillna(0)
        # keep them as ints
        for c in count_like_columns:
            features[c] = features[c].astype("int64")

    LOG.info(f"Final GCS feature matrix: rows={len(features):,}, cols={features.shape[1]:,}")

    # Summary logging
    count_cols = [c for c in features.columns if c.endswith("_n_admission_measurements")]
    subj_zero = (features[count_cols].sum(axis=1) == 0).sum() if count_cols else 0
    subj_nonzero = len(features) - subj_zero
    LOG.info(f"Subjects with zero GCS entries in 0–48h: {subj_zero:,}")
    LOG.info(f"Subjects with ≥1 GCS entry: {subj_nonzero:,}")

    n_feats = len(feature_names_all)
    if n_feats > 0 and count_cols:
        missing_frac = (features[count_cols] == 0).sum(axis=1) / n_feats
        LOG.info(
            "Missing feature coverage per subject "
            f"(fraction of features with zero measurements out of {n_feats} total): "
            f"mean={missing_frac.mean():.3f}, min={missing_frac.min():.3f}, max={missing_frac.max():.3f}"
        )

    _log_feature_summaries(features, feature_names_all)

    if save:
        utils.GCS_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(utils.GCS_FEATURES_PATH, index=False)
        LOG.info(f"Wrote GCS features to: {utils.GCS_FEATURES_PATH}")

    return features


if __name__ == "__main__":
    build_gcs_features(cohort_path=utils.FILTERED_COHORT_PATH)
