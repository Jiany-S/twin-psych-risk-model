"""Run artifact provenance helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def latest_run(root: str | Path, prefix: str) -> Path:
    root_path = Path(root)
    matches = sorted([p for p in root_path.glob(f"{prefix}*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No run directory matching {prefix}* under {root_path}")
    return matches[0]

