import json
from pathlib import Path
import joblib


def save_artifacts(
    outdir: Path,
    preprocessor,
    model,
    calibrator,
    feature_names: list[str],
    splits: dict,
):
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, outdir / "preprocessor.joblib")
    joblib.dump(model, outdir / "model_xgb.joblib")
    if calibrator is not None:
        joblib.dump(calibrator, outdir / "calibrator.joblib")
    with open(outdir / "feature_columns.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(outdir / "train_val_test_indices.json", "w") as f:
        json.dump(splits, f, indent=2)


def load_model_bundle(dirpath: Path):
    pre = joblib.load(dirpath / "preprocessor.joblib")
    model = joblib.load(dirpath / "model_xgb.joblib")
    calib = (dirpath / "calibrator.joblib")
    calibrator = joblib.load(calib) if calib.exists() else None
    return pre, model, calibrator
