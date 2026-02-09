"""WESAD dataset loader supporting native pickle and CSV exports."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import DataSchema


LABEL_MAP_STRESS = {
    1: 0.0,  # baseline
    2: 1.0,  # stress
    3: np.nan,  # amusement excluded from stress binary by default
}
LABEL_MAP_COMFORT = {
    1: 0.9,  # baseline
    2: 0.2,  # stress
    3: 0.7,  # amusement
}
PROTOCOL_MAP = {1: "baseline", 2: "stress", 3: "amusement"}


def _discover_wesad_pickles(path: Path) -> list[Path]:
    return sorted(path.glob("S*/S*.pkl"))


def _normalize_subject_id(raw: str) -> str:
    txt = str(raw).strip()
    if txt.upper().startswith("S"):
        suffix = txt[1:]
    else:
        suffix = txt
    return f"S{suffix}"


def _subject_id_from_path(path: Path) -> str:
    stem = path.stem
    return _normalize_subject_id(stem)


def _extract_signal(signal_dict: dict[str, Any], key: str) -> np.ndarray | None:
    value = signal_dict.get(key)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim > 1:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.mean(axis=1)
    return arr.astype(float)


def _load_wesad_subject_pickle(pkl_path: Path, schema: DataSchema) -> pd.DataFrame:
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh, encoding="latin1")

    label = np.asarray(data.get("label", []), dtype=int)
    signal = data.get("signal", {})
    chest = signal.get("chest", {})
    wrist = signal.get("wrist", {})

    ecg = _extract_signal(chest, "ECG")
    eda = _extract_signal(chest, "EDA")
    temp = _extract_signal(chest, "Temp")
    resp = _extract_signal(chest, "Resp")
    acc = _extract_signal(chest, "ACC")

    if ecg is None:
        # fallback to wrist BVP proxy when ECG is unavailable
        ecg = _extract_signal(wrist, "BVP")
    if eda is None:
        eda = _extract_signal(wrist, "EDA")
    if temp is None:
        temp = _extract_signal(wrist, "TEMP")
    if acc is None:
        acc = _extract_signal(wrist, "ACC")

    if ecg is None or eda is None or temp is None or label.size == 0:
        raise ValueError(f"Incomplete WESAD subject file: {pkl_path}")

    lengths = [len(label), len(ecg), len(eda), len(temp)]
    if resp is not None:
        lengths.append(len(resp))
    if acc is not None:
        lengths.append(len(acc))
    n = min(lengths)

    label = label[:n]
    ecg = ecg[:n]
    eda = eda[:n]
    temp = temp[:n]
    resp = resp[:n] if resp is not None else np.full(n, np.nan)
    acc = acc[:n] if acc is not None else np.full(n, np.nan)

    df = pd.DataFrame(
        {
            schema.worker_id: _subject_id_from_path(pkl_path),
            schema.timestamp: np.arange(n, dtype=float),
            schema.time_idx: np.arange(n, dtype=int),
            "ecg": ecg,
            "eda": eda,
            "temp": temp,
            "resp": resp,
            "accel": acc,
            schema.protocol_label: pd.Series(label).map(PROTOCOL_MAP).fillna("other"),
            schema.stress_target: pd.Series(label).map(LABEL_MAP_STRESS),
            schema.comfort_target: pd.Series(label).map(LABEL_MAP_COMFORT),
        }
    )
    df = df[df[schema.comfort_target].notna()].copy()
    # For stress target we drop unknown protocol rows (e.g., meditations/transitions).
    df = df[df[schema.stress_target].notna()].copy()
    return df.reset_index(drop=True)


def _load_wesad_csvs(path: Path, schema: DataSchema) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}.")
    frames: list[pd.DataFrame] = []
    for csv_file in files:
        frame = pd.read_csv(csv_file)
        if schema.worker_id not in frame.columns:
            frame[schema.worker_id] = csv_file.stem
        frame[schema.worker_id] = frame[schema.worker_id].astype(str).map(_normalize_subject_id)
        if schema.timestamp not in frame.columns:
            if schema.time_idx in frame.columns:
                frame[schema.timestamp] = frame[schema.time_idx]
            else:
                frame[schema.timestamp] = np.arange(len(frame), dtype=float)
        if schema.time_idx not in frame.columns:
            frame[schema.time_idx] = np.arange(len(frame), dtype=int)
        if schema.protocol_label in frame.columns:
            protocol_raw = frame[schema.protocol_label]
            protocol_num = pd.to_numeric(protocol_raw, errors="coerce")
            if protocol_num.notna().any():
                frame[schema.stress_target] = protocol_num.map(LABEL_MAP_STRESS)
                frame[schema.comfort_target] = protocol_num.map(LABEL_MAP_COMFORT)
                frame[schema.protocol_label] = protocol_num.map(PROTOCOL_MAP).fillna("other")
            else:
                text = protocol_raw.astype(str).str.lower()
                frame[schema.protocol_label] = text
                frame[schema.stress_target] = text.map({"baseline": 0.0, "stress": 1.0, "amusement": np.nan})
                frame[schema.comfort_target] = text.map({"baseline": 0.9, "stress": 0.2, "amusement": 0.7})
        else:
            if schema.stress_target not in frame.columns or schema.comfort_target not in frame.columns:
                raise ValueError(
                    f"{csv_file} must include either '{schema.protocol_label}' or both "
                    f"'{schema.stress_target}' and '{schema.comfort_target}'."
                )
            frame[schema.protocol_label] = frame.get(schema.protocol_label, "unknown")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df[schema.stress_target].notna() & df[schema.comfort_target].notna()].copy()
    return df


def load_wesad_dataset(
    dataset_path: str | Path,
    schema: DataSchema,
    data_format: str = "auto",
    subjects: list[str] | None = None,
    max_rows_per_subject: int | None = None,
    downsample_factor: int | None = None,
) -> pd.DataFrame:
    """Load WESAD from native pickle folders or CSV exports."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"WESAD dataset path does not exist: {path}")

    pickles = _discover_wesad_pickles(path)
    use_pickles = data_format in {"auto", "wesad_pickle"} and len(pickles) > 0
    use_csv = data_format in {"auto", "csv"} and not use_pickles

    if use_pickles:
        frames = []
        for pkl_path in pickles:
            subject_df = _load_wesad_subject_pickle(pkl_path, schema)
            if max_rows_per_subject and max_rows_per_subject > 0:
                subject_df = subject_df.iloc[:max_rows_per_subject].copy()
            if downsample_factor and downsample_factor > 1:
                subject_df = subject_df.iloc[::downsample_factor].copy()
            frames.append(subject_df)
        if not frames:
            raise FileNotFoundError(f"No valid WESAD subject pickle files found under {path}")
        df = pd.concat(frames, ignore_index=True)
    elif use_csv:
        df = _load_wesad_csvs(path, schema)
        if max_rows_per_subject and max_rows_per_subject > 0:
            df = (
                df.sort_values([schema.worker_id, schema.time_idx])
                .groupby(schema.worker_id, observed=True)
                .head(max_rows_per_subject)
                .reset_index(drop=True)
            )
        if downsample_factor and downsample_factor > 1:
            df = (
                df.sort_values([schema.worker_id, schema.time_idx])
                .groupby(schema.worker_id, observed=True)
                .apply(lambda g: g.iloc[::downsample_factor])
                .reset_index(drop=True)
            )
    else:
        raise ValueError(
            f"Unsupported WESAD structure at {path}. "
            "Expected S*/S*.pkl files or CSV files, and dataset.format in {auto,wesad_pickle,csv}."
        )

    for missing_col in schema.robot_context:
        if missing_col not in df.columns:
            df[missing_col] = 0.0
    if schema.hazard_zone not in df.columns:
        df[schema.hazard_zone] = 0
    if schema.task_phase not in df.columns:
        df[schema.task_phase] = "default"

    if subjects:
        normalized = {_normalize_subject_id(s) for s in subjects}
        df = df[df[schema.worker_id].astype(str).map(_normalize_subject_id).isin(normalized)].copy()
        if df.empty:
            raise ValueError(
                f"No rows remain after subject filtering. Requested subjects: {sorted(normalized)}"
            )
        df[schema.worker_id] = df[schema.worker_id].astype(str).map(_normalize_subject_id)

    return df.sort_values([schema.worker_id, schema.time_idx]).reset_index(drop=True)
