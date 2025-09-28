from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def xgb_gain_importance_df(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract gain-based importances from an XGBClassifier.
    Returns a DataFrame with columns: feature, gain.
    """
    booster = model.get_booster()
    # XGBoost indexes features as f0,f1,... in the trained trees
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
    raw = booster.get_score(importance_type="gain")
    data = [(fmap.get(k, k), v) for k, v in raw.items()]
    df = pd.DataFrame(data, columns=["feature", "gain"]).sort_values("gain", ascending=False)
    return df


def plot_top_importances(df_imp: pd.DataFrame, outpath: Path, top_k: int = 30) -> None:
    top = df_imp.head(top_k)[::-1]
    plt.figure(figsize=(8, max(4, 0.25 * len(top))))
    plt.barh(top["feature"], top["gain"])
    plt.xlabel("Gain importance"); plt.title(f"Top {len(top)} features (XGBoost gain)")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
