
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from .. import utils

LOGGER = utils.make_logger("preprocessing", "preprocessing.log")

MISSING_TOKEN = "MISSING"
OTHER_TOKEN = "OTHER"


class TopKCategoryMapper(BaseEstimator, TransformerMixin):
    """
    Keep top-K most frequent categories for a single column, map others to OTHER_TOKEN.
    Expects a DataFrame with exactly one column (the target categorical feature).
    """
    def __init__(self, k: int = 40):
        self.k = k
        self.keep_: Optional[set] = None
        self.feature_name_: Optional[str] = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("TopKCategoryMapper expects exactly one column.")
            self.feature_name_ = X.columns[0]
            s = X.iloc[:, 0].astype("object")
        else:
            # accept 2D array with a single column, although from the final way we build the data I (Avia) don't think this can happen. But let's keep it here just in case I'm wrong =)
            arr = np.asarray(X)
            if arr.ndim != 2 or arr.shape[1] != 1:
                raise ValueError("TopKCategoryMapper expects shape (n_samples, 1).")
            s = pd.Series(arr[:, 0], dtype="object")

        vc = (
            pd.Series(s)
            .astype("object")
            .value_counts(dropna=True)
            .sort_values(ascending=False)
        )
        self.keep_ = set(vc.head(self.k).index.tolist())
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0].astype("object")
        else:
            arr = np.asarray(X)
            s = pd.Series(arr[:, 0], dtype="object")

        def map_val(v):
            if pd.isna(v):
                return np.nan
            return v if v in self.keep_ else OTHER_TOKEN

        out = s.map(map_val)
        return out.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = self.feature_name_ or (input_features[0] if input_features else "feature")
        return np.array([name], dtype=object)


class AddAllMissingFlag(BaseEstimator, TransformerMixin):
    """
    Given a DataFrame/array of columns (practically, we only use it for our embedding features (*_enm)), emit a single column:
      1 if *all* inputs are missing on that row, else 0.
    Intended to run before any imputation.
    """
    def __init__(self):
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None):
        X = self._to_df(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._to_df(X)
        # Row-wise: all NA -> 1, else 0
        flag = X.isna().all(axis=1).astype("int8").to_numpy().reshape(-1, 1)
        return flag

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"col{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def get_feature_names_out(self, input_features=None):
        return np.array(["emb_missing"], dtype=object)


class Log1pColumns(BaseEstimator, TransformerMixin):
    """
    Apply log1p to columns (assumes numeric). If negative values appear, clips to 0 before log1p.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, 0.0)
        X = np.clip(X, 0.0, None)
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        # Add suffix for clarity
        if input_features is None:
            return None
        return np.array([f"{c}_log1p" for c in input_features], dtype=object)


@dataclass
class ColumnPlan:
    embeddings: List[str]
    categoricals: List[str]
    sign_mean_as_cat: List[str]
    count_like: List[str]          # e.g., n_*, *_n_admission_measurements, *_n_admissions
    flag_like: List[str]           # *_any_*, no_*
    other_numerics: List[str]      # all remaining numeric (incl. age)
    eth_raw_present: bool


def _present(cols: Iterable[str], X: pd.DataFrame) -> List[str]:
    cset = set(X.columns)
    return [c for c in cols if c in cset]


def discover_columns(X: pd.DataFrame) -> ColumnPlan:
    """
    Decide which columns go into which preprocessing blocks, based on names.
    - embeddings: starts with 'emb_'
    - categoricals: eth_raw, eth_broad, gender, insurance, admission_type, admission_location
    - sign_mean_as_cat: columns ending with '_sign_mean'
    - count_like: startswith 'n_' OR endswith '_n_admission_measurements'/'_n_admissions'
    - flag_like: contains '_any_' OR startswith 'no_'
    - other_numerics: remaining numeric columns excluding IDs and those already assigned
    """
    id_cols = [c for c in ["subject_id", "admission_id"] if c in X.columns]

    embeddings = [c for c in X.columns if isinstance(c, str) and c.startswith("emb_")]

    cat_explicit_all = ["eth_raw", "eth_broad", "gender", "insurance", "admission_type", "admission_location"]
    cat_explicit = _present(cat_explicit_all, X)

    sign_mean_as_cat = [c for c in X.columns if isinstance(c, str) and c.endswith("_sign_mean")]
    noteevents_categoricals = [c for c in X.columns if c == "first_note_category"]

    count_like = []
    for c in X.columns:
        if not isinstance(c, str):
            continue
        if c.startswith("n_") or c.endswith("_n_admission_measurements") or c.endswith("_n_admissions"):
            count_like.append(c)

    flag_like = [c for c in X.columns if isinstance(c, str) and ("_any_" in c or c.startswith("no_") or c.startswith("has_"))]

    # Build categorical list
    categoricals = list(dict.fromkeys(cat_explicit + sign_mean_as_cat + noteevents_categoricals))  # dedupe

    # other numerics = numeric cols not already assigned and not IDs
    assigned = set(id_cols) | set(embeddings) | set(categoricals) | set(count_like) | set(flag_like)
    numeric_candidates = X.select_dtypes(include=[np.number]).columns.tolist()
    other_numerics = [c for c in numeric_candidates if c not in assigned]

    # Explicitly include 'age' if not already captured (Avia: although I'm 99% sure it is already captured)
    if "age" in X.columns and "age" not in other_numerics and "age" not in assigned:
        other_numerics.append("age")

    # logging the columns we discovered
    LOGGER.info("Column discovery:")
    LOGGER.info("  embeddings: %d", len(embeddings))
    LOGGER.info("  categoricals (explicit + *_sign_mean): %d", len(categoricals))
    LOGGER.info("  count_like: %d", len(count_like))
    LOGGER.info("  flag_like: %d", len(flag_like))
    LOGGER.info("  other_numerics: %d", len(other_numerics))

    plan = ColumnPlan(
        embeddings=embeddings,
        categoricals=categoricals,
        sign_mean_as_cat=sign_mean_as_cat,
        count_like=count_like,
        flag_like=flag_like,
        other_numerics=other_numerics,
        eth_raw_present=("eth_raw" in categoricals),
    )
    return plan


def _to_float64_matrix(X):
    # because we had issues with pd.NAs what scikit learn doesn't like.
    try:
        if isinstance(X, pd.DataFrame):
            X = X.apply(pd.to_numeric, errors="coerce")
            return X.to_numpy(dtype=np.float64, copy=False)
        if hasattr(X, "to_numpy"):
            return X.to_numpy(dtype=np.float64, copy=False)
        return np.asarray(X, dtype=np.float64)
    except Exception:
        return np.asarray(X, dtype=np.float64)


def _to_object_with_nan(X):
    """
    Ensure categoricals are arrays of dtype object with np.nan for missing.
    """
    if isinstance(X, pd.DataFrame):
        X = X.astype("object")
        X = X.mask(pd.isna(X), other=np.nan)
        return X.to_numpy(copy=False)
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy(dtype=object, copy=False)
    else:
        arr = np.asarray(X, dtype=object)
    arr = np.where(pd.isna(arr), np.nan, arr)
    return arr



def build_preprocessor(plan: ColumnPlan) -> ColumnTransformer:
    """
    Build a ColumnTransformer like this:
      A) Embeddings (emb_*): emit emb_missing flag (pre-impute), then mean-impute emb_*.
      B) Categoricals: (eth_raw with TopK->OTHER) -> impute 'MISSING' -> OHE.
      C) Counts/Flags:
         - For count-like (n_*, *_n_admission_measurements, *_n_admissions):
             * per-column MissingIndicator + 0-impute + log1p clone
         - For flag-like (*_any_*, no_*, has*):
             * per-column MissingIndicator + 0-impute (no log1p)
      D) Other numerics : median-impute + missing indicators.
    """
    transformers: list[tuple[str, Pipeline | TransformerMixin, Sequence[str]]] = []

    # embeddings
    if plan.embeddings:
        # single missing flag
        emb_flag = Pipeline(steps=[
            ("flag", AddAllMissingFlag()),
        ])
        transformers.append(("emb_missing_flag", emb_flag, plan.embeddings))

        # The embeddings themselves (mean imputed)
        emb_pipe = Pipeline(steps=[
            ("to_float64", FunctionTransformer(_to_float64_matrix, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="mean")),
        ])
        transformers.append(("emb_values", emb_pipe, plan.embeddings))

    # Categoricals
    if plan.categoricals:
        # Split eth_raw vs others so we can apply TopK only to eth_raw
        cats_other = [c for c in plan.categoricals if c != "eth_raw"]
        if plan.eth_raw_present:
            eth_raw_pipe = Pipeline(steps=[
                ("to_object", FunctionTransformer(_to_object_with_nan, feature_names_out="one-to-one")),
                ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
                ("topk", TopKCategoryMapper(k=40)),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ])
            transformers.append(("cat_eth_raw", eth_raw_pipe, ["eth_raw"]))

        if cats_other:
            cat_pipe = Pipeline(steps=[
                ("to_object", FunctionTransformer(_to_object_with_nan, feature_names_out="one-to-one")),
                ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ])
            transformers.append(("cat_other", cat_pipe, cats_other))

    # Count-like (with log1p)
    if plan.count_like:
        # MissingIndicator (always one per column)
        count_ind = MissingIndicator(features="all", sparse=False)
        transformers.append(("count_missing_ind", count_ind, plan.count_like))

        # 0-impute branch
        count_impute = Pipeline(steps=[
            ("to_float64", FunctionTransformer(_to_float64_matrix, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ])
        transformers.append(("count_values", count_impute, plan.count_like))

        # log1p
        count_log1p = Pipeline(steps=[
            ("to_float64", FunctionTransformer(_to_float64_matrix, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("log1p", Log1pColumns()),
        ])
        transformers.append(("count_log1p", count_log1p, plan.count_like))

    # flag-like (0/1 style, no log1p)
    if plan.flag_like:
        flag_ind = MissingIndicator(features="all", sparse=False)
        transformers.append(("flag_missing_ind", flag_ind, plan.flag_like))

        flag_values = Pipeline(steps=[
            ("to_float64", FunctionTransformer(_to_float64_matrix, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ])
        transformers.append(("flag_values", flag_values, plan.flag_like))

    #  Other numerics (median + indicators)
    if plan.other_numerics:
        num_pipe = Pipeline(steps=[
            ("to_float64", FunctionTransformer(_to_float64_matrix, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ])
        transformers.append(("other_num", num_pipe, plan.other_numerics))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,     # allow sparse from OHE to pass through
        verbose_feature_names_out=False,
    )
    return pre
