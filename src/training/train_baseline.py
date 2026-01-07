"""Train baseline models (logistic regression and GRU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.schema import DataSchema
from ..data.splits import temporal_split
from ..models.baselines import (
    LogisticRegressionBaseline,
    SequenceGRUClassifier,
    assemble_tabular_features,
    build_sequence_tensors,
)
from ..utils.logging import configure_logging
from ..utils.seed import seed_everything


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_logistic(
    cfg: dict[str, Any],
    schema: DataSchema,
    df: pd.DataFrame,
    artifacts_dir: Path,
    logger,
) -> None:
    feature_cols = cfg["baseline"]["feature_columns"]
    split = temporal_split(df, time_col=schema.time_idx, val_ratio=cfg["data"]["val_ratio"], test_ratio=cfg["data"]["test_ratio"])
    X_train, y_train = assemble_tabular_features(split.train, feature_cols, schema.target)
    X_val, y_val = assemble_tabular_features(split.val, feature_cols, schema.target)
    X_test, y_test = assemble_tabular_features(split.test, feature_cols, schema.target)

    logger.info("Fitting logistic regression on %d samples.", len(X_train))
    model = LogisticRegressionBaseline()
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)
    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    val_ap = average_precision_score(y_val, val_probs)
    test_ap = average_precision_score(y_test, test_probs)
    logger.info("LogReg AUROC — val: %.3f | test: %.3f", val_auc, test_auc)
    logger.info("LogReg AUPRC — val: %.3f | test: %.3f", val_ap, test_ap)

    baseline_dir = artifacts_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    model_path = baseline_dir / "logistic_regression.joblib"
    model.save(model_path.as_posix())
    metadata = {
        "type": "logistic_regression",
        "feature_columns": feature_cols,
        "target": schema.target,
        "time_idx": schema.time_idx,
    }
    (baseline_dir / "logistic_regression.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_gru(
    cfg: dict[str, Any],
    schema: DataSchema,
    df: pd.DataFrame,
    artifacts_dir: Path,
    logger,
) -> None:
    feature_cols = cfg["baseline"]["feature_columns"]
    window = cfg["data"]["window_length"]
    horizon = cfg["data"]["prediction_horizon"]
    split = temporal_split(df, time_col=schema.time_idx, val_ratio=cfg["data"]["val_ratio"], test_ratio=cfg["data"]["test_ratio"])

    X_train, y_train = build_sequence_tensors(
        split.train,
        feature_cols,
        schema.target,
        schema.worker_id,
        window,
        horizon,
        time_col=schema.time_idx,
    )
    X_val, y_val = build_sequence_tensors(
        split.val,
        feature_cols,
        schema.target,
        schema.worker_id,
        window,
        horizon,
        time_col=schema.time_idx,
    )

    device = torch.device("cuda" if cfg["training"]["gpus"] > 0 and torch.cuda.is_available() else "cpu")
    model = SequenceGRUClassifier(
        input_size=X_train.shape[-1],
        hidden_size=cfg["baseline"]["gru_hidden_size"],
        num_layers=cfg["baseline"]["gru_layers"],
        dropout=cfg["baseline"]["gru_dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["baseline"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg["training"]["batch_size"])

    best_val_auc = 0.0
    best_state = None
    for epoch in range(cfg["baseline"]["epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds.append(torch.sigmoid(logits).cpu())
                labels.append(batch_y)
        if preds:
            val_probs = torch.cat(preds).numpy()
            val_labels = torch.cat(labels).numpy()
            val_auc = roc_auc_score(val_labels, val_probs)
            logger.info("Epoch %d | GRU val AUROC: %.3f", epoch + 1, val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    baseline_dir = artifacts_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    model_path = baseline_dir / "gru_baseline.pt"
    torch.save(best_state, model_path)
    metadata = {
        "type": "gru",
        "feature_columns": feature_cols,
        "target": schema.target,
        "worker_id": schema.worker_id,
        "window_length": window,
        "prediction_horizon": horizon,
        "hidden_size": cfg["baseline"]["gru_hidden_size"],
        "num_layers": cfg["baseline"]["gru_layers"],
        "dropout": cfg["baseline"]["gru_dropout"],
    }
    (baseline_dir / "gru_baseline.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved GRU baseline to %s", model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression and GRU baselines.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml")
    parser.add_argument(
        "--model",
        type=str,
        choices=("logistic", "gru", "both"),
        default="both",
        help="Choose which baseline to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["training"]["seed"])
    logger = configure_logging("train_baseline")

    processed_path = Path(cfg["paths"]["processed_train_file"])
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}. Run preprocessing first.")
    df = pd.read_csv(processed_path)
    schema = DataSchema.from_config(cfg)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])

    if args.model in ("logistic", "both"):
        train_logistic(cfg, schema, df, artifacts_dir, logger)
    if args.model in ("gru", "both"):
        train_gru(cfg, schema, df, artifacts_dir, logger)


if __name__ == "__main__":
    main()
