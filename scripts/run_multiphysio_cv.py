from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from src.data.loader import load_or_generate
from src.data.preprocess import preprocess_dataframe
from src.data.schema import DataSchema
from src.training.metrics import expected_calibration_error, select_threshold_from_validation
from src.utils.io import load_merged_yaml, load_yaml


METRIC_KEYS = ["auroc", "auprc", "macro_f1", "balanced_accuracy", "brier", "ece"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    default_path = Path("src/config/default.yaml")
    if cfg_path.resolve() == default_path.resolve():
        return load_yaml(cfg_path)
    return load_merged_yaml(default_path, cfg_path)


def _make_run_dir(root: str | Path) -> Path:
    run_dir = Path(root) / f"multiphysio_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for child in ["fold_manifests", "models", "predictions"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _targets(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    configured = cfg.get("benchmark_targets", {})
    if not configured:
        raise ValueError("multiphysio_cv.yaml must define benchmark_targets.")
    return {str(name): dict(block) for name, block in configured.items()}


def _feature_matrix(frame: pd.DataFrame, schema: DataSchema) -> tuple[np.ndarray, list[str]]:
    feature_names = list(schema.physiology)
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing configured physiological feature columns: {missing}")
    return frame[feature_names].to_numpy(dtype=float), feature_names


def _split_train_validation(
    train_indices: np.ndarray,
    groups: np.ndarray,
    seed: int,
    validation_group_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_groups = np.unique(groups[train_indices])
    if len(train_groups) < 2:
        return train_indices, train_indices
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_group_fraction, random_state=seed)
    local = np.arange(len(train_indices))
    local_train, local_val = next(splitter.split(local, groups=groups[train_indices]))
    return train_indices[local_train], train_indices[local_val]


def _thresholded_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    pred = probs >= threshold
    unique = np.unique(y_true)
    auroc = float("nan") if unique.size < 2 else float(roc_auc_score(y_true, probs))
    auprc = float("nan") if unique.size < 2 else float(average_precision_score(y_true, probs))
    return {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "prevalence": float(np.mean(y_true)) if len(y_true) else 0.0,
        "dummy_auprc_baseline": float(np.mean(y_true)) if len(y_true) else 0.0,
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
        "auroc": auroc,
        "auprc": auprc,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": float(expected_calibration_error(probs, y_true, bins=15)),
        "threshold": float(threshold),
    }


def _model_specs(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    models_cfg = cfg.get("models", {})
    specs: dict[str, Any] = {}
    if bool(models_cfg.get("run_dummy", True)):
        specs["dummy_most_frequent"] = lambda: DummyClassifier(strategy="most_frequent", random_state=seed)
        specs["dummy_stratified"] = lambda: DummyClassifier(strategy="stratified", random_state=seed)
    if bool(models_cfg.get("run_logistic", True)):
        log_cfg = cfg.get("logistic_regression", {})
        specs["logistic_regression"] = lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=int(log_cfg.get("max_iter", 1000)),
                        C=float(log_cfg.get("C", 1.0)),
                        solver=str(log_cfg.get("solver", "lbfgs")),
                        class_weight=log_cfg.get("class_weight"),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if bool(models_cfg.get("run_random_forest", True)):
        rf_cfg = cfg.get("random_forest", {})
        specs["random_forest"] = lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(rf_cfg.get("n_estimators", 300)),
                        max_depth=rf_cfg.get("max_depth"),
                        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
                        class_weight=rf_cfg.get("class_weight"),
                        n_jobs=int(rf_cfg.get("n_jobs", -1)),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if bool(models_cfg.get("run_adaboost", True)):
        ada_cfg = cfg.get("adaboost", {})
        specs["adaboost"] = lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=int(ada_cfg.get("n_estimators", 200)),
                        learning_rate=float(ada_cfg.get("learning_rate", 0.5)),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if bool(models_cfg.get("run_xgb", True)) and XGBClassifier is not None:
        xgb_cfg = cfg.get("xgboost", {})
        specs["xgboost"] = lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=int(xgb_cfg.get("n_estimators", 200)),
                        max_depth=int(xgb_cfg.get("max_depth", 3)),
                        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                        subsample=float(xgb_cfg.get("subsample", 0.9)),
                        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
                        eval_metric=str(xgb_cfg.get("eval_metric", "logloss")),
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    elif bool(models_cfg.get("run_xgb", True)):
        specs["xgboost"] = None
    return specs


def _positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "steps") and hasattr(model[-1], "classes_"):
        classes = list(model[-1].classes_)
    else:
        classes = [0, 1]
    if 1 in classes:
        return probs[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def _markdown_table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "No successful rows."
    rows = frame[cols].sort_values(["target", "model"]).copy()
    for col in rows.columns:
        if pd.api.types.is_numeric_dtype(rows[col]):
            rows[col] = rows[col].map(lambda v: "nan" if pd.isna(v) else f"{float(v):.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows.to_numpy()]
    return "\n".join([header, sep, *body])


def _aggregate(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), block in per_fold.groupby(["target", "model"], dropna=False):
        row: dict[str, Any] = {"target": target, "model": model, "n_folds": int(len(block))}
        for metric in METRIC_KEYS + ["prevalence", "predicted_positive_rate"]:
            values = pd.to_numeric(block[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            row[f"{metric}_ci95_low"] = (
                float(np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else row[f"{metric}_mean"]
            )
            row[f"{metric}_ci95_high"] = (
                float(np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else row[f"{metric}_mean"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(run_dir: Path, aggregate: pd.DataFrame, failures: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    view_cols = [
        "target",
        "model",
        "n_folds",
        "auroc_mean",
        "auprc_mean",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "brier_mean",
        "ece_mean",
        "prevalence_mean",
    ]
    lines = [
        "# MultiPhysio Benchmark",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "This benchmark uses grouped subject cross-validation over `bio_features_60s.csv`. One row is a 60-second precomputed physiological feature interval, suitable for slow workload or affect estimation, not immediate safety intervention.",
        "",
        "Only physiological features are used: `hrv_mean_nn`, `eda_mean`, `emg_rmse`, and `rrv_mean_bb`. No fabricated role, specialization, experience, or subject-ID metadata is used as a feature.",
        "",
        "Targets: STAI-Y1 stress (`STAI >= 40`), NASA-TLX cognitive workload (`NASA >= 40`), SAM Valence (`Valence >= 3`), and SAM Arousal (`Arousal >= 3`).",
        "",
        "Models: Dummy most-frequent, Dummy stratified, Logistic Regression, Random Forest, AdaBoost, XGBoost. TFT is disabled because this benchmark uses tabular 60-second feature rows rather than a meaningful raw temporal stream.",
        "",
        "## Aggregate Metrics",
        "",
        "The table below shows fold means. `aggregate_results.csv` and `aggregate_results.json` include mean, standard deviation, median, and 95% confidence interval columns for every metric. `per_subject_metrics.csv` reports metrics for each held-out subject within each grouped fold.",
        "",
        _markdown_table(aggregate, view_cols),
        "",
        "## Failure Summary",
        "",
        "No failures." if not failures else "See `failures.json` for details.",
        "",
        "## Artifacts",
        "",
        "- `fold_manifests/fold_*.csv`",
        "- `per_fold_metrics.csv`",
        "- `per_subject_metrics.csv`",
        "- `aggregate_results.csv`",
        "- `failures.json`",
    ]
    (run_dir / "multiphysio_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    Path("docs/multiphysio_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str) -> Path:
    cfg = _load_config(config_path)
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    np.random.seed(seed)
    schema = DataSchema.from_config(cfg)
    raw = load_or_generate(cfg, schema)
    df = preprocess_dataframe(cfg, raw, schema)
    X, feature_names = _feature_matrix(df, schema)
    groups = df[schema.worker_id].astype(str).to_numpy()
    targets = _targets(cfg)

    run_dir = _make_run_dir(cfg.get("paths", {}).get("run_root", "experiments/runs"))
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    cv_cfg = cfg.get("cv", {})
    mode = str(cv_cfg.get("mode", "leave_one_subject_out"))
    if mode == "group_kfold":
        splitter = GroupKFold(n_splits=int(cv_cfg.get("n_splits", 5)))
        splits = list(splitter.split(X, groups=groups))
    else:
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(X, groups=groups))
    max_folds = cv_cfg.get("max_folds")
    if max_folds:
        splits = splits[: int(max_folds)]

    model_specs = _model_specs(cfg, seed)
    per_fold_rows: list[dict[str, Any]] = []
    per_subject_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    val_fraction = float(cv_cfg.get("validation_group_fraction", 0.2))

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        train_idx, val_idx = _split_train_validation(train_val_idx, groups, seed + fold_idx, val_fraction)
        manifest = pd.DataFrame(
            {
                "row_index": np.concatenate([train_idx, val_idx, test_idx]),
                "worker_id": groups[np.concatenate([train_idx, val_idx, test_idx])],
                "split": ["train"] * len(train_idx) + ["validation"] * len(val_idx) + ["test"] * len(test_idx),
            }
        )
        manifest.to_csv(run_dir / "fold_manifests" / f"fold_{fold_idx:03d}.csv", index=False)

        for target_name, target_info in targets.items():
            label_col = str(target_info["label_col"])
            y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float).astype(int)
            for model_name, factory in model_specs.items():
                if factory is None:
                    failures.append({"fold": fold_idx, "target": target_name, "model": model_name, "error": "xgboost_unavailable"})
                    continue
                try:
                    model = factory()
                    model.fit(X[train_idx], y[train_idx])
                    val_probs = _positive_proba(model, X[val_idx])
                    threshold_diag = select_threshold_from_validation(
                        y[val_idx],
                        val_probs,
                        policy="f1",
                        min_pred_rate=0.0,
                        max_pred_rate=1.0,
                        allow_pathological=True,
                    )
                    threshold = float(threshold_diag["threshold"])
                    test_probs = _positive_proba(model, X[test_idx])
                    metrics = _thresholded_metrics(y[test_idx], test_probs, threshold)
                    metrics.update(
                        {
                            "fold": fold_idx,
                            "target": target_name,
                            "model": model_name,
                            "source": target_info.get("source"),
                            "positive_definition": target_info.get("positive_definition"),
                            "test_subjects": ",".join(sorted(np.unique(groups[test_idx]))),
                            "threshold_source": "validation_only",
                        }
                    )
                    per_fold_rows.append(metrics)
                    pred_path = run_dir / "predictions" / f"fold_{fold_idx:03d}_{target_name}_{model_name}.csv"
                    pd.DataFrame(
                        {
                            "worker_id": groups[test_idx],
                            "y_true": y[test_idx],
                            "probability": test_probs,
                            "prediction": (test_probs >= threshold).astype(int),
                        }
                    ).to_csv(pred_path, index=False)
                    joblib.dump(model, run_dir / "models" / f"fold_{fold_idx:03d}_{target_name}_{model_name}.pkl")

                    for subject in sorted(np.unique(groups[test_idx])):
                        mask = groups[test_idx] == subject
                        subject_metrics = _thresholded_metrics(y[test_idx][mask], test_probs[mask], threshold)
                        subject_metrics.update({"fold": fold_idx, "target": target_name, "model": model_name, "subject": subject})
                        per_subject_rows.append(subject_metrics)
                except Exception as exc:
                    failures.append({"fold": fold_idx, "target": target_name, "model": model_name, "error": repr(exc)})

    per_fold = pd.DataFrame(per_fold_rows)
    per_subject = pd.DataFrame(per_subject_rows)
    aggregate = _aggregate(per_fold) if not per_fold.empty else pd.DataFrame()
    per_fold.to_csv(run_dir / "per_fold_metrics.csv", index=False)
    per_subject.to_csv(run_dir / "per_subject_metrics.csv", index=False)
    aggregate.to_csv(run_dir / "aggregate_results.csv", index=False)
    (run_dir / "per_fold_metrics.json").write_text(json.dumps(per_fold_rows, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "aggregate_results.json").write_text(
        json.dumps(aggregate.to_dict(orient="records"), indent=2, default=_json_default), encoding="utf-8"
    )
    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(run_dir, aggregate, failures, cfg)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped subject CV benchmark for MultiPhysio-HRC.")
    parser.add_argument("--config", default="src/config/multiphysio_cv.yaml")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"MultiPhysio CV artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
