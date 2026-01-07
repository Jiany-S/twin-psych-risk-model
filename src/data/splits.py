"""Dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "time_idx",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Split dataframe into train/val/test partitions using time-based thresholds."""
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    times = df[time_col]
    min_time, max_time = times.min(), times.max()
    val_cut = min_time + (max_time - min_time) * (1 - test_ratio - val_ratio)
    test_cut = min_time + (max_time - min_time) * (1 - test_ratio)

    train = df[times <= val_cut]
    val = df[(times > val_cut) & (times <= test_cut)]
    test = df[times > test_cut]

    return SplitResult(train=train.copy(), val=val.copy(), test=test.copy())


def worker_split(
    df: pd.DataFrame,
    worker_col: str = "worker_id",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Split dataframe by worker ids."""
    workers = sorted(df[worker_col].unique())
    n_workers = len(workers)
    val_size = max(1, int(n_workers * val_ratio))
    test_size = max(1, int(n_workers * test_ratio))

    rng = np.random.default_rng(42)
    shuffled = workers.copy()
    rng.shuffle(shuffled)

    test_workers = set(shuffled[:test_size])
    val_workers = set(shuffled[test_size : test_size + val_size])

    train = df[~df[worker_col].isin(test_workers | val_workers)]
    val = df[df[worker_col].isin(val_workers)]
    test = df[df[worker_col].isin(test_workers)]

    return SplitResult(train=train.copy(), val=val.copy(), test=test.copy())
