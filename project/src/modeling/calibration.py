from sklearn.calibration import CalibratedClassifierCV


def fit_isotonic_prefit(model, X_val, y_val):
    """
    Fit isotonic calibration on top of a prefit classifier.
    Returns (calibrator, predict_proba_fn) where predict_proba_fn(X) -> calibrated proba.
    """
    # cv='prefit' emits a FutureWarning in sklearn 1.7 (what we use) but it is ok for our purposes.
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)

    def predict_proba(X):
        return calibrator.predict_proba(X)[:, 1]

    return calibrator, predict_proba
