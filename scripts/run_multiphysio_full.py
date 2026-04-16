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


def _build_subject_holdout(
    frame,
    schema: DataSchema,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    min_val_positive_windows: int,
    min_test_positive_windows: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    stats = (
        frame.groupby(schema.worker_id, observed=True)[schema.stress_target]
        .agg(["size", "sum"])
        .reset_index()
    )
    stats[schema.worker_id] = stats[schema.worker_id].astype(str)
    stats["sum"] = stats["sum"].astype(int)
    stats["size"] = stats["size"].astype(int)
    stats["neg"] = stats["size"] - stats["sum"]
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

    total_pos = int(stats["sum"].sum())
    total_rows = int(stats["size"].sum())
    target_pos = {
        "train": max(1, int(round(total_pos * train_ratio))),
        "val": max(min_val_positive_windows, int(round(total_pos * val_ratio))),
        "test": max(min_test_positive_windows, total_pos - int(round(total_pos * train_ratio)) - int(round(total_pos * val_ratio))),
    }
    # Keep positives feasible.
    if target_pos["train"] + target_pos["val"] + target_pos["test"] > total_pos:
        overflow = target_pos["train"] + target_pos["val"] + target_pos["test"] - total_pos
        target_pos["train"] = max(1, target_pos["train"] - overflow)

    target_rows = {
        "train": max(1, int(round(total_rows * train_ratio))),
        "val": max(1, int(round(total_rows * val_ratio))),
        "test": max(1, total_rows - int(round(total_rows * train_ratio)) - int(round(total_rows * val_ratio))),
    }
    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    assigned = {"train": {"pos": 0, "rows": 0}, "val": {"pos": 0, "rows": 0}, "test": {"pos": 0, "rows": 0}}
    targets = {"train": target_train, "val": target_val, "test": target_test}
    worker_meta = {
        str(row[schema.worker_id]): {"pos": int(row["sum"]), "rows": int(row["size"])}
        for _, row in stats.iterrows()
    }

    # Sort by descending positives, then rows, to place high-signal workers first.
    ordered = sorted(
        workers,
        key=lambda w: (worker_meta[w]["pos"], worker_meta[w]["rows"]),
        reverse=True,
    )
    for w in ordered:
        candidates = [s for s in ("train", "val", "test") if len(groups[s]) < targets[s]]
        if not candidates:
            candidates = ["train", "val", "test"]
        wp = worker_meta[w]["pos"]
        wr = worker_meta[w]["rows"]

        def score(split: str) -> float:
            after_pos = assigned[split]["pos"] + wp
            after_rows = assigned[split]["rows"] + wr
            after_n = len(groups[split]) + 1
            pos_term = abs(after_pos - target_pos[split]) / max(1, target_pos[split])
            rows_term = abs(after_rows - target_rows[split]) / max(1, target_rows[split])
            n_term = abs(after_n - targets[split]) / max(1, targets[split])
            # bias val/test to avoid starving positives in calibration/evaluation splits
            boost = -0.15 if split in {"val", "test"} and wp > 0 else 0.0
            return 1.0 * pos_term + 0.5 * rows_term + 0.5 * n_term + boost

        best = sorted(candidates, key=score)[0]
        groups[best].append(w)
        assigned[best]["pos"] += wp
        assigned[best]["rows"] += wr

    # Repair loop to enforce minimum positive windows in val/test.
    def split_pos(name: str) -> int:
        return int(sum(worker_meta[w]["pos"] for w in groups[name]))

    def split_rows(name: str) -> int:
        return int(sum(worker_meta[w]["rows"] for w in groups[name]))

    for target_split, min_pos in (("val", min_val_positive_windows), ("test", min_test_positive_windows)):
        guard = 0
        while split_pos(target_split) < min_pos and guard < 200:
            guard += 1
            donor_candidates = sorted(groups["train"], key=lambda w: worker_meta[w]["pos"], reverse=True)
            receiver_candidates = sorted(groups[target_split], key=lambda w: worker_meta[w]["pos"])
            if not donor_candidates or not receiver_candidates:
                break
            donor = donor_candidates[0]
            receiver = receiver_candidates[0]
            if worker_meta[donor]["pos"] <= worker_meta[receiver]["pos"]:
                break
            groups["train"].remove(donor)
            groups[target_split].append(donor)
            groups[target_split].remove(receiver)
            groups["train"].append(receiver)

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
    split_stats = {
        name: {
            "subjects": int(len(groups[name])),
            "rows": split_rows(name),
            "positives": split_pos(name),
        }
        for name in ("train", "val", "test")
    }
    return groups, split_stats


def _assert_split_class_coverage(
    frame,
    schema: DataSchema,
    split_subjects: dict[str, list[str]],
    min_val_positive_windows: int,
    min_test_positive_windows: int,
) -> None:
    for name in ("train", "val", "test"):
        ids = set(split_subjects[name])
        part = frame[frame[schema.worker_id].astype(str).isin(ids)]
        labels = set(part[schema.stress_target].astype(float).unique().tolist())
        if 0.0 not in labels or 1.0 not in labels:
            raise ValueError(f"{name} split is single-class under current assignment: labels={sorted(labels)}")
        positives = int(part[schema.stress_target].sum())
        if name == "val" and positives < min_val_positive_windows:
            raise ValueError(f"val split positives too low: {positives} < {min_val_positive_windows}")
        if name == "test" and positives < min_test_positive_windows:
            raise ValueError(f"test split positives too low: {positives} < {min_test_positive_windows}")


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
    min_val_positive_windows = int(split_cfg.get("min_val_positive_windows", 20))
    min_test_positive_windows = int(split_cfg.get("min_test_positive_windows", 20))

    subjects, split_stats = _build_subject_holdout(
        frame,
        schema,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        min_val_positive_windows=min_val_positive_windows,
        min_test_positive_windows=min_test_positive_windows,
    )
    _assert_split_class_coverage(
        frame,
        schema,
        subjects,
        min_val_positive_windows=min_val_positive_windows,
        min_test_positive_windows=min_test_positive_windows,
    )

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
    print(f"Split stats (rows/positives): {split_stats}")
    print(f"Test class balance: {test_balance}")
    print(f"XGBoost stress AUROC: {metrics.get('xgboost', {}).get('stress', {}).get('auroc')}")
    print(f"TFT stress AUROC: {metrics.get('tft', {}).get('stress', {}).get('auroc')}")


if __name__ == "__main__":
    main()
