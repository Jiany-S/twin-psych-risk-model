"""Run full MultiPhysio training with deterministic subject-holdout split construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_or_generate  # noqa: E402
from src.data.preprocess import preprocess_dataframe  # noqa: E402
from src.data.schema import DataSchema  # noqa: E402
from src.run_experiment import run_experiment  # noqa: E402
from src.utils.io import load_merged_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full MultiPhysio training.")
    parser.add_argument("--config", type=str, default="src/config/multiphysio_full.yaml")
    parser.add_argument(
        "--resolved-config-out",
        type=str,
        default="experiments/tmp/multiphysio_full_resolved.yaml",
    )
    return parser.parse_args()


def _round_robin_fill(
    candidates: list[str], groups: dict[str, list[str]], target_counts: dict[str, int], rng: np.random.Generator
) -> None:
    if not candidates:
        return
    pool = candidates.copy()
    rng.shuffle(pool)
    for worker in pool:
        # fill the most under-target split first
        deficits = {
            name: target_counts[name] - len(groups[name])
            for name in ("train", "val", "test")
        }
        order = sorted(deficits, key=lambda n: deficits[n], reverse=True)
        groups[order[0]].append(worker)


def _build_subject_holdout(frame, schema: DataSchema, seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    stats = (
        frame.groupby(schema.worker_id, observed=True)[schema.stress_target]
        .agg(["size", "sum"])
        .reset_index()
    )
    stats[schema.worker_id] = stats[schema.worker_id].astype(str)
    positives = sorted(stats.loc[stats["sum"] > 0, schema.worker_id].tolist())
    negatives = sorted(stats.loc[stats["sum"] <= 0, schema.worker_id].tolist())
    workers = sorted(stats[schema.worker_id].tolist())
    n_workers = len(workers)
    if n_workers < 9:
        raise ValueError("Not enough workers for stable 3-way holdout split (need >= 9).")

    rng = np.random.default_rng(seed)
    target_train = max(3, int(round(n_workers * train_ratio)))
    target_val = max(2, int(round(n_workers * val_ratio)))
    target_test = max(2, n_workers - target_train - target_val)
    # Fix rounding drift.
    while target_train + target_val + target_test > n_workers:
        target_train = max(3, target_train - 1)
    while target_train + target_val + target_test < n_workers:
        target_train += 1

    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_names = ["train", "val", "test"]
    rng.shuffle(split_names)

    # Guarantee at least one positive and one negative per split when possible.
    if len(positives) >= 3:
        pos = positives.copy()
        rng.shuffle(pos)
        for idx, name in enumerate(split_names):
            groups[name].append(pos[idx])
        positives = [w for w in pos[3:]]
    if len(negatives) >= 3:
        neg = negatives.copy()
        rng.shuffle(neg)
        for idx, name in enumerate(split_names):
            groups[name].append(neg[idx])
        negatives = [w for w in neg[3:]]

    targets = {"train": target_train, "val": target_val, "test": target_test}
    _round_robin_fill(positives, groups, targets, rng)
    _round_robin_fill(negatives, groups, targets, rng)

    # Final deterministic normalization.
    for name in groups:
        groups[name] = sorted(set(groups[name]))
    overlap = (set(groups["train"]) & set(groups["val"])) | (set(groups["train"]) & set(groups["test"])) | (set(groups["val"]) & set(groups["test"]))
    if overlap:
        raise RuntimeError(f"Split construction produced overlap: {sorted(overlap)}")

    all_assigned = sorted(groups["train"] + groups["val"] + groups["test"])
    if sorted(workers) != all_assigned:
        missing = sorted(set(workers) - set(all_assigned))
        extra = sorted(set(all_assigned) - set(workers))
        raise RuntimeError(f"Split construction mismatch. Missing={missing}, Extra={extra}")
    return groups


def _assert_split_class_coverage(frame, schema: DataSchema, split_subjects: dict[str, list[str]]) -> None:
    for name in ("train", "val", "test"):
        ids = set(split_subjects[name])
        part = frame[frame[schema.worker_id].astype(str).isin(ids)]
        labels = set(part[schema.stress_target].astype(float).unique().tolist())
        if 0.0 not in labels or 1.0 not in labels:
            raise ValueError(f"{name} split is single-class under current assignment: labels={sorted(labels)}")


def main() -> None:
    args = parse_args()
    cfg = load_merged_yaml(ROOT / "src" / "config" / "default.yaml", ROOT / args.config)
    if str(cfg.get("dataset", {}).get("name", "")).lower() != "multiphysio":
        raise ValueError("run_multiphysio_full.py expects dataset.name = multiphysio")

    schema = DataSchema.from_config(cfg)
    frame = preprocess_dataframe(cfg, load_or_generate(cfg, schema), schema)
    split_cfg = cfg.get("split", {})
    seed = int(cfg.get("reproducibility", {}).get("seed", split_cfg.get("seed", 42)))
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))

    subjects = _build_subject_holdout(frame, schema, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    _assert_split_class_coverage(frame, schema, subjects)

    cfg["dataset"]["subjects"] = sorted(frame[schema.worker_id].astype(str).unique().tolist())
    cfg["split"]["mode"] = "subject_holdout"
    cfg["split"]["train_subjects"] = subjects["train"]
    cfg["split"]["val_subjects"] = subjects["val"]
    cfg["split"]["test_subjects"] = subjects["test"]

    resolved_path = Path(args.resolved_config_out)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    run_dir = run_experiment(str(resolved_path))
    metrics_path = Path(run_dir) / "metrics.json"
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    for model_key in ("xgboost", "tft"):
        block = metrics.get(model_key, {})
        if isinstance(block, dict) and "error" in block:
            raise RuntimeError(f"{model_key} failed: {block['error']}")

    test_balance = metrics.get("class_balance", {}).get("test", {})
    if "0.0" not in test_balance or "1.0" not in test_balance:
        raise RuntimeError(f"Full run completed but test split is single-class: {test_balance}")

    print(f"Full MultiPhysio run completed: {run_dir}")
    print(f"Resolved config: {resolved_path.resolve()}")
    print(f"Split subjects: train={len(subjects['train'])} val={len(subjects['val'])} test={len(subjects['test'])}")
    print(f"Test class balance: {test_balance}")
    print(f"XGBoost stress AUROC: {metrics.get('xgboost', {}).get('stress', {}).get('auroc')}")
    print(f"TFT stress AUROC: {metrics.get('tft', {}).get('stress', {}).get('auroc')}")


if __name__ == "__main__":
    main()
