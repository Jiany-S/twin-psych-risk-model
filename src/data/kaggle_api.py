"""Helpers to fetch datasets using the official Kaggle API client."""

from __future__ import annotations

from pathlib import Path


def fetch_kaggle_dataset(
    dataset_ref: str,
    target_dir: str | Path,
    force_download: bool = False,
    quiet: bool = True,
) -> Path:
    """Download and extract a Kaggle dataset into target_dir.

    Requires Kaggle credentials via ~/.kaggle/kaggle.json or env vars
    KAGGLE_USERNAME/KAGGLE_KEY.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if any(target.iterdir()) and not force_download:
        return target

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kaggle package is required for dataset.source='kaggle_api'. Install with `pip install kaggle`."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Kaggle authentication failed. Configure ~/.kaggle/kaggle.json or "
            "set KAGGLE_USERNAME/KAGGLE_KEY."
        ) from exc

    api.dataset_download_files(
        dataset=dataset_ref,
        path=str(target),
        force=force_download,
        quiet=quiet,
        unzip=True,
    )
    if not any(target.iterdir()):
        raise RuntimeError(f"Kaggle dataset download produced an empty directory: {target}")
    return target

