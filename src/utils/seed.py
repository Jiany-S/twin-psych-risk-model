"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass
    try:
        from lightning.pytorch import seed_everything as lightning_seed_everything

        lightning_seed_everything(seed, workers=True)
    except Exception:
        try:
            from pytorch_lightning import seed_everything as lightning_seed_everything

            lightning_seed_everything(seed, workers=True)
        except Exception:
            pass
