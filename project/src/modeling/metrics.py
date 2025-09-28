from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.calibration import calibration_curve


from .. import utils

def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": roc_auc_score(y_true, p),
        "auprc": average_precision_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "prevalence": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def plot_roc_pr_calibration(y_true: np.ndarray, p: np.ndarray, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y_true, p)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, p)
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUROC={m['auroc']:.3f})")
    plt.tight_layout(); plt.savefig(outdir / "roc.png", dpi=150); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, p)
    plt.figure()
    plt.plot(rec, prec, lw=2)
    plt.hlines(m["prevalence"], 0, 1, linestyles="--")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUPRC={m['auprc']:.3f})")
    plt.tight_layout(); plt.savefig(outdir / "pr.png", dpi=150); plt.close()

    # calibration curve
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed fraction")
    plt.title(f"Calibration (Brier={m['brier']:.3f})")
    plt.tight_layout(); plt.savefig(outdir / "calibration.png", dpi=150); plt.close()


def evaluate_and_plot(y_true: np.ndarray, p: np.ndarray, outdir: Path) -> dict:
    """Compute metrics, save plots + JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, p)
    plot_roc_pr_calibration(y_true, p, outdir)
    with open(outdir / "training_metrics.json", "w") as f:
        json.dump(
            {
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "prevalence": metrics["prevalence"],
                "n_test": metrics["n"],
            },
            f, indent=2
        )
    return {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier": metrics["brier"],
        "prevalence": metrics["prevalence"],
        "n_test": metrics["n"],
    }


def save_bias_files(pred_df: pd.DataFrame, outdir: Path, logger) -> None:
    def _derive_eth_course(eth_raw: pd.Series) -> pd.Series:
        s = eth_raw.astype("string").str.lower()

        def map_eth(val: str) -> str:
            if val.startswith("white"):
                return "white"
            if val.startswith("black"):
                return "black"
            if val.startswith("asian"):
                return "asian"
            if val.startswith("hisp") or val.startswith("latin"):
                return "hispanic"
            return "other"

        return s.map(map_eth).astype("string")

    def _group_auprc(df_joined: pd.DataFrame, group_col: str, all_groups: list[str]) -> pd.DataFrame:
        rows = []
        for g in all_groups:
            sub = df_joined[df_joined[group_col] == g]
            cnt = int(len(sub))
            if cnt == 0:
                ap = float("nan")
            else:
                y = sub["y_true"].to_numpy()
                p = sub["y_pred"].to_numpy(dtype=float)
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())
                if n_pos == 0 or n_neg == 0:
                    # undefined AUPRC when only one class present
                    ap = float("nan")
                    logger.warning(
                        f"{group_col}='{g}': AUPRC undefined (positives={n_pos}, negatives={n_neg}); writing NaN.")
                else:
                    ap = float(average_precision_score(y, p))
            rows.append({group_col: g, "count": cnt, "AUPRC": ap})
        return pd.DataFrame(rows, columns=[group_col, "count", "AUPRC"])

    # Build the dataframe we need: test ids + y_true/y_pred + demographics
    _demo_cols = ["subject_id", "gender", "eth_broad", "eth_raw"]
    demographics_df = pd.read_parquet(utils.DEMOGRAPHICS_PATH).loc[:, _demo_cols].copy()

    test_join = pred_df.merge(demographics_df, on="subject_id", how="left", validate="one_to_one")
    test_join["eth_course"] = _derive_eth_course(test_join["eth_raw"])

    # Enumerate category sets:
    #   For gender/eth_broad: use all unique values present in the full demographics
    #   For eth_course: fixed canonical set like we saw in the course
    gender_all = sorted(demographics_df["gender"].astype("string").unique().tolist())
    eth_broad_all = sorted(demographics_df["eth_broad"].astype("string").unique().tolist())
    eth_course_all = ["white", "black", "asian", "hispanic", "other"]

    gender_df = _group_auprc(test_join, "gender", gender_all)
    eth_broad_df = _group_auprc(test_join, "eth_broad", eth_broad_all)
    eth_course_df = _group_auprc(test_join, "eth_course", eth_course_all)

    # save
    gender_df.to_csv(outdir / "gender_bias.csv", index=False)
    eth_broad_df.to_csv(outdir / "eth_broad_bias.csv", index=False)
    eth_course_df.to_csv(outdir / "eth_course_bias.csv", index=False)

    logger.info("Saved per-group AUPRC reports: gender_bias.csv, eth_broad_bias.csv, eth_course_bias.csv")
