"""XGBoost baseline training and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_xgboost() -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required. Install it via requirements.txt.") from exc
    return xgb


@dataclass
class XGBResult:
    model: Any
    calibrator: Any | None
    feature_names: list[str]


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    params: dict[str, Any],
    feature_names: list[str],
) -> XGBResult:
    xgb = _require_xgboost()
    common_params = dict(
        max_depth=params.get("max_depth", 4),
        n_estimators=params.get("n_estimators", 200),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        random_state=42,
    )

    if task_type == "classification":
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_params,
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=params.get("early_stopping_rounds", 25),
            )
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common_params,
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=params.get("early_stopping_rounds", 25),
            )
        except TypeError:
            model.fit(X_train, y_train)

    calibrator = None
    if task_type == "classification":
        method = params.get("calibration", "none")
        if method and method != "none":
            calibrator = fit_calibrator(model, X_val, y_val, method)

    return XGBResult(model=model, calibrator=calibrator, feature_names=feature_names)


def fit_calibrator(model: Any, X_val: np.ndarray, y_val: np.ndarray, method: str) -> Any:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    probs = model.predict_proba(X_val)[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, y_val)
        return iso
    if method == "platt":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(probs.reshape(-1, 1), y_val)
        return lr
    return None


def predict_xgb(model: Any, calibrator: Any | None, X: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        probs = model.predict_proba(X)[:, 1]
        if calibrator is None:
            return probs
        if calibrator.__class__.__name__ == "IsotonicRegression":
            return calibrator.predict(probs)
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    return model.predict(X)
