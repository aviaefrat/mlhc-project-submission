import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

from . import utils
from .data_build import load_and_merge
from .modeling.preprocessing import discover_columns, build_preprocessor
from .modeling.split import split_train_val_test
from .modeling.metrics import evaluate_and_plot, save_bias_files
from .modeling.calibration import fit_isotonic_prefit
from .modeling.io import save_artifacts
from .modeling.explain import xgb_gain_importance_df, plot_top_importances

LOGGER = utils.make_logger("train", "train.log")


def get_best_hyperparams(target: str, spw):
    best_hyperparams = {
        "mortality": {
            'max_depth': 5,
            'learning_rate': 0.03,
            'reg_lambda': 1,
            'reg_alpha': 0,
            'min_child_weight': 5,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'scale_pos_weight': spw,
        },
        "prolonged_stay": {
            'max_depth': 7,
            'learning_rate': 0.03,
            'reg_lambda': 5,
            'reg_alpha': 1,
            'min_child_weight': 5,
            'subsample': 1.0,
            'colsample_bytree': 0.6,
            'scale_pos_weight': spw
        },
        "readmission": {
            'max_depth': 3,
            'learning_rate': 0.2,
            'reg_lambda': 0,
            'reg_alpha': 1,
            'min_child_weight': 5,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'scale_pos_weight': spw * 2
        }
    }
    return best_hyperparams[target]


@dataclass
class TrainConfig:
    target: str = "mortality"
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 1407
    learning_rate: float = 0.05
    n_estimators: int = 4000
    max_depth: int = 5
    min_child_weight: float = 4.0
    scale_pos_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    early_stopping_rounds: int = 100
    calibration: str = "isotonic"  # "isotonic" or "none"
    use_gpu: bool = False
    save_dir: Path = Path("reports")


def _features_used_string(
    include_demographics: bool,
    include_vitals: bool,
    include_labs: bool,
    include_gcs: bool,
    include_prescriptions: bool,
    include_noteevents: bool,
    include_embeddings: bool
) -> str:
    """
    Return the dvlgpne subset string with letters removed for features we use --exclude_FEATURE for in the training.
    The string containing all the letters of the features is d v l g p n e.
    """
    spec = [
        ("d", include_demographics),
        ("v", include_vitals),
        ("l", include_labs),
        ("g", include_gcs),
        ("p", include_prescriptions),
        ("n", include_noteevents),
        ("e", include_embeddings),
    ]
    return "".join(letter for letter, inc in spec if inc)

def _timestamped_experiment_dir(base: Path, target: str, tuning: str, features_used: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tune_mid = "_tune_" if tuning == "grid" else "_"
    experiment_dir = base / f"{target}{tune_mid}{features_used}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def make_xgb(cfg: TrainConfig, scale_pos_weight: float) -> XGBClassifier:
    params = dict(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        tree_method="hist",
        device="cuda" if cfg.use_gpu else "cpu",
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=cfg.seed,
        scale_pos_weight=scale_pos_weight,
        n_jobs=0,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    LOGGER.info(f"XGB params: {params}")
    return XGBClassifier(**params)


def grid_tune_on_val(Xtr, ytr, Xva, yva, base_cfg: TrainConfig, spw: float) -> dict:
    grid = {
        "max_depth": [3, 5, 7],
        # LR: one low, one mid, one fast
        "learning_rate": [0.03, 0.10, 0.20],
        "reg_lambda": [0, 1, 5],
        "reg_alpha": [0, 1],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    # add "scale_pos_weight" according to the target:
    scale_pos_weight_values = None
    if base_cfg.target == "mortality":
        scale_pos_weight_values = [spw * 0.5, spw, spw * 1.5]
    elif base_cfg.target == "prolonged_stay":
        scale_pos_weight_values = [spw * 0.75, spw, spw * 1.25]
    elif base_cfg.target == "readmission":
        scale_pos_weight_values = [spw * 0.5, spw, spw * 2]
    grid["scale_pos_weight"] = scale_pos_weight_values

    # if base_cfg.target == "readmission":
    #     grid = {
    #         "max_depth": [3],
    #         "learning_rate": [0.05, 0.10],
    #         "reg_lambda": [5, 10],
    #         "reg_alpha": [1, 2],
    #         "min_child_weight": [3, 5, 7],
    #         "subsample": [0.5, 0.6, 0.7],
    #         "colsample_bytree": [0.5, 0.6, 0.7],
    #         "gamma": [0, 1],
    #         "sampling_method": ["gradient_based"]
    #         "max_bin": [512],
    #         "scale_pos_weight": [spw * 0.25, spw * 0.5, spw * 0.75],
    #     }

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]

    device = "cuda" if base_cfg.use_gpu else "cpu"
    best_ap, best_params = -1.0, None

    for i, params in enumerate(combos, 1):
        cfg = TrainConfig(**{**base_cfg.__dict__})
        for k, v in params.items():
            setattr(cfg, k, v)

        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=cfg.seed,
            scale_pos_weight=cfg.scale_pos_weight,
            n_jobs=0,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ap = average_precision_score(yva, model.predict_proba(Xva)[:, 1])
        LOGGER.info(
            f"[grid {i}/{len(combos)}] params={params}  val_AUPRC={ap:.4f}  "
            f"best_iter={getattr(model, 'best_iteration', None)}"
        )
        if ap > best_ap:
            best_ap, best_params = ap, {**params, "n_estimators": cfg.n_estimators}

    LOGGER.info(f"Grid search best on VAL: AUPRC={best_ap:.4f}  params={best_params}")
    return best_params or {}


def main(argv: list[str] | None = None, target: str | None = None):
    parser = argparse.ArgumentParser(description="Train XGB binary classifier with calibration.")
    parser.add_argument("--target", type=str, choices=["mortality", "prolonged_stay", "readmission"], required=True)
    parser.add_argument("--save-dir", type=Path, default=utils.REPORTS_DIR)
    parser.add_argument("--seed", type=int, default=1407)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--tune", type=str, choices=["none", "grid"], default="none")
    parser.add_argument("--calibration", type=str, choices=["none", "isotonic"], default="isotonic")

    # features if we want to exclude
    parser.add_argument("--no-demographics", action="store_true", help="Exclude demographic features")
    parser.add_argument("--no-vitals", action="store_true", help="Exclude vitals features")
    parser.add_argument("--no-labs", action="store_true", help="Exclude labs features")
    parser.add_argument("--no-gcs", action="store_true", help="Exclude GCS features")
    parser.add_argument("--no-prescriptions", action="store_true", help="Exclude prescriptions features")
    parser.add_argument("--no-noteevents", action="store_true", help="Exclude non-embedding noteevents features")
    parser.add_argument("--no-embeddings", action="store_true", help="Exclude embeddings (w2v) noteevents features")

    if target is not None:
        # bypass argparse when we want to use the __main__ in pycharm
        args = parser.parse_args([f"--target={target}"])
    else:
        args = parser.parse_args(argv)

    cfg = TrainConfig(
        target=args.target,
        save_dir=args.save_dir,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        use_gpu=args.use_gpu,
        calibration = args.calibration,
    )

    # Resolve include flags
    include_demographics = not args.no_demographics
    include_vitals = not args.no_vitals
    include_labs = not args.no_labs
    include_gcs = not args.no_gcs
    include_prescriptions = not args.no_prescriptions
    include_noteevents = False # Avia: we ended up not using these features, so I just hardcoded `False` here as a hack...
    include_embeddings = not args.no_embeddings

    features_used = _features_used_string(
        include_demographics=include_demographics,
        include_vitals=include_vitals,
        include_labs=include_labs,
        include_gcs=include_gcs,
        include_prescriptions=include_prescriptions,
        include_noteevents=include_noteevents,
        include_embeddings=include_embeddings,
    )

    df = load_and_merge(
        include_demographics=include_demographics,
        include_vitals=include_vitals,
        include_labs=include_labs,
        include_gcs=include_gcs,
        include_prescriptions=include_prescriptions,
        include_noteevents=include_noteevents,
        include_embeddings=include_embeddings,
        target=cfg.target,
    ).copy()

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df, cfg.target, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed
    )

    # build preprocessor
    plan = discover_columns(X_train)
    preprocessor = build_preprocessor(plan)

    # transform sets (fit on train only)
    Xtr = preprocessor.fit_transform(X_train)
    Xva = preprocessor.transform(X_val)
    Xte = preprocessor.transform(X_test)

    # to save the feature names we ended up using into a file
    try:
        feat_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(Xtr.shape[1])]

    # label imbalance
    pos = float(np.sum(y_train == 1)); neg = float(np.sum(y_train == 0))
    spw = (neg / pos) if pos > 0 else 1.0
    LOGGER.info(f"scale_pos_weight (train): {spw:.3f}  [pos={pos:.0f}, neg={neg:.0f}]")

    # if no tuning, set the hyperparams to the ones we found when we ran our the grid:
    if args.tune == "none":
        best_hyperparams = get_best_hyperparams(target=cfg.target, spw=spw)
        cfg.__dict__.update(best_hyperparams)

    elif args.tune == "grid":
        LOGGER.info("Starting simple grid tuning on validation...")
        best_params = grid_tune_on_val(Xtr, y_train, Xva, y_val, cfg, spw)
        for k, v in best_params.items():
            setattr(cfg, k, v)
        # rebuild preprocessor (same columns), re-transform
        preprocessor = build_preprocessor(plan)
        Xtr = preprocessor.fit_transform(X_train)
        Xva = preprocessor.transform(X_val)
        Xte = preprocessor.transform(X_test)

    # Final model
    model = make_xgb(cfg, spw)
    model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
    LOGGER.info(f"Best iteration: {model.best_iteration}  Best score: {getattr(model, 'best_score', 'n/a')}")

    # Calibration
    calibrator = None
    if cfg.calibration == "isotonic":
        calibrator, predict_proba = fit_isotonic_prefit(model, Xva, y_val)
        p_test = predict_proba(Xte)
        LOGGER.info("Applied isotonic calibration using validation set.")
    else:
        p_test = model.predict_proba(Xte)[:, 1]

    ## outputs:
    outdir = _timestamped_experiment_dir(cfg.save_dir, cfg.target, args.tune, features_used)

    metrics = evaluate_and_plot(y_test, p_test, outdir)

    # Save per-row test predictions
    pred_df = X_test[["subject_id"]].copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = p_test
    pred_csv = outdir / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    # bias
    save_bias_files(pred_df=pred_df, outdir=outdir, logger=LOGGER)

    # Feature importances (gain)
    imp_df = xgb_gain_importance_df(model, feat_names)
    imp_csv = outdir / "feature_importance_gain.csv"
    imp_png = outdir / "feature_importance_gain_top30.png"
    imp_df.to_csv(imp_csv, index=False)
    plot_top_importances(imp_df, imp_png, top_k=30)

    # Split IDs
    splits = {
        "train_subject_ids": X_train["subject_id"].astype(int).tolist(),
        "val_subject_ids": X_val["subject_id"].astype(int).tolist(),
        "test_subject_ids": X_test["subject_id"].astype(int).tolist(),
    }
    save_artifacts(outdir, preprocessor, model, calibrator, feat_names, splits)

    # save run configuration
    run_args = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "target": cfg.target,
        "features_used": features_used,
        "include": {
            "demographics": include_demographics,
            "vitals": include_vitals,
            "labs": include_labs,
            "gcs": include_gcs,
            "prescriptions": include_prescriptions,
            "noteevents": include_noteevents,
            "embeddings": include_embeddings,
        },
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "val_size": cfg.val_size,
        "use_gpu": cfg.use_gpu,
        "calibration": cfg.calibration,
        "tuning": args.tune,
        "xgb_params_used": model.get_params(deep=False),
        "xgb_best_iteration": getattr(model, "best_iteration", None),
        "xgb_best_score": getattr(model, "best_score", None),
        "test_metrics": metrics,
    }
    with open(outdir / "run_args.json", "w") as f:
        json.dump(run_args, f, indent=2)

    LOGGER.info(f"Saved run artifacts to: {outdir}")
    LOGGER.info(f"Metrics: {metrics}")


def train_on_all_labels():
    targets = ["mortality", "prolonged_stay", "readmission"]

    for target in targets:
        main(target=target)

if __name__ == "__main__":
    main()
