"""WESAD dataset loader supporting native pickle and CSV exports."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .schema import DataSchema


LABEL_MAP_STRESS_DEFAULT = {
    1: 0.0,  # baseline
    2: 1.0,  # stress
    3: np.nan,  # amusement excluded by default
}
LABEL_MAP_STRESS_INCLUDE_AMUSEMENT = {
    1: 0.0,
    2: 1.0,
    3: 0.0,  # treat amusement as non-stress
}
LABEL_MAP_COMFORT = {
    1: 0.9,  # baseline
    2: 0.2,  # stress
    3: 0.7,  # amusement
}
PROTOCOL_MAP = {1: "baseline", 2: "stress", 3: "amusement"}
WESAD_CHEST_RATE_HZ = 700.0
WESAD_WRIST_RATES_HZ = {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}


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


def _resample_numeric(values: np.ndarray, source_rate_hz: float, target_ts: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full(len(target_ts), np.nan)
    source_ts = np.arange(len(values), dtype=float) / float(source_rate_hz)
    if len(source_ts) == 1:
        return np.full(len(target_ts), float(values[0]))
    interpolator = interp1d(source_ts, values.astype(float), bounds_error=False, fill_value=(values[0], values[-1]))
    return interpolator(target_ts).astype(float)


def _sample_labels(label: np.ndarray, source_rate_hz: float, target_ts: np.ndarray) -> np.ndarray:
    source_idx = np.clip(np.round(target_ts * float(source_rate_hz)).astype(int), 0, len(label) - 1)
    return label[source_idx]


def _load_wesad_subject_pickle(
    pkl_path: Path,
    schema: DataSchema,
    target_sampling_rate_hz: float,
    stress_include_amusement: bool = False,
) -> pd.DataFrame:
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
    ecg_rate = eda_rate = temp_rate = resp_rate = acc_rate = WESAD_CHEST_RATE_HZ

    if ecg is None:
        # fallback to wrist BVP proxy when ECG is unavailable
        ecg = _extract_signal(wrist, "BVP")
        ecg_rate = WESAD_WRIST_RATES_HZ["BVP"]
    if eda is None:
        eda = _extract_signal(wrist, "EDA")
        eda_rate = WESAD_WRIST_RATES_HZ["EDA"]
    if temp is None:
        temp = _extract_signal(wrist, "TEMP")
        temp_rate = WESAD_WRIST_RATES_HZ["TEMP"]
    if acc is None:
        acc = _extract_signal(wrist, "ACC")
        acc_rate = WESAD_WRIST_RATES_HZ["ACC"]

    if ecg is None or eda is None or temp is None or label.size == 0:
        raise ValueError(f"Incomplete WESAD subject file: {pkl_path}")

    if target_sampling_rate_hz <= 0:
        raise ValueError("target_sampling_rate_hz must be positive for WESAD resampling.")
    duration_seconds = len(label) / WESAD_CHEST_RATE_HZ
    n = int(np.floor(duration_seconds * target_sampling_rate_hz))
    if n < 2:
        raise ValueError(f"WESAD subject {pkl_path} is too short after resampling.")
    target_ts = np.arange(n, dtype=float) / target_sampling_rate_hz
    label_resampled = _sample_labels(label, WESAD_CHEST_RATE_HZ, target_ts)
    ecg = _resample_numeric(ecg, ecg_rate, target_ts)
    eda = _resample_numeric(eda, eda_rate, target_ts)
    temp = _resample_numeric(temp, temp_rate, target_ts)
    resp = _resample_numeric(resp, resp_rate, target_ts) if resp is not None else np.full(n, np.nan)
    acc = _resample_numeric(acc, acc_rate, target_ts) if acc is not None else np.full(n, np.nan)

    label_map_stress = LABEL_MAP_STRESS_INCLUDE_AMUSEMENT if stress_include_amusement else LABEL_MAP_STRESS_DEFAULT
    df = pd.DataFrame(
        {
            schema.worker_id: _subject_id_from_path(pkl_path),
            schema.timestamp: target_ts,
            schema.time_idx: np.arange(n, dtype=int),
            "ecg": ecg,
            "eda": eda,
            "temp": temp,
            "resp": resp,
            "accel": acc,
            schema.protocol_label: pd.Series(label_resampled).map(PROTOCOL_MAP).fillna("other"),
            schema.stress_target: pd.Series(label_resampled).map(label_map_stress),
            schema.comfort_target: pd.Series(label_resampled).map(LABEL_MAP_COMFORT),
        }
    )
    df = df[df[schema.comfort_target].notna()].copy()
    # For stress target we drop unknown protocol rows (e.g., meditations/transitions).
    df = df[df[schema.stress_target].notna()].copy()
    return df.reset_index(drop=True)


def _load_wesad_csvs(path: Path, schema: DataSchema, stress_include_amusement: bool = False) -> pd.DataFrame:
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
                label_map_stress = (
                    LABEL_MAP_STRESS_INCLUDE_AMUSEMENT if stress_include_amusement else LABEL_MAP_STRESS_DEFAULT
                )
                frame[schema.stress_target] = protocol_num.map(label_map_stress)
                frame[schema.comfort_target] = protocol_num.map(LABEL_MAP_COMFORT)
                frame[schema.protocol_label] = protocol_num.map(PROTOCOL_MAP).fillna("other")
            else:
                text = protocol_raw.astype(str).str.lower()
                frame[schema.protocol_label] = text
                if stress_include_amusement:
                    frame[schema.stress_target] = text.map({"baseline": 0.0, "stress": 1.0, "amusement": 0.0})
                else:
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
    target_sampling_rate_hz: float = 4.0,
    stress_include_amusement: bool = False,
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
            subject_df = _load_wesad_subject_pickle(
                pkl_path,
                schema,
                target_sampling_rate_hz=target_sampling_rate_hz,
                stress_include_amusement=stress_include_amusement,
            )
            if downsample_factor and downsample_factor > 1:
                raise ValueError(
                    "dataset.downsample_factor is deprecated for WESAD pickle loading. "
                    "Use stream.target_sampling_rate_hz for explicit resampling."
                )
            if max_rows_per_subject and max_rows_per_subject > 0 and len(subject_df) > max_rows_per_subject:
                subject_df = subject_df.iloc[:max_rows_per_subject].copy()
            frames.append(subject_df)
        if not frames:
            raise FileNotFoundError(f"No valid WESAD subject pickle files found under {path}")
        df = pd.concat(frames, ignore_index=True)
    elif use_csv:
        df = _load_wesad_csvs(path, schema, stress_include_amusement=stress_include_amusement)
        df = df.sort_values([schema.worker_id, schema.time_idx])
        if downsample_factor and downsample_factor > 1:
            raise ValueError("dataset.downsample_factor is deprecated. Use stream.target_sampling_rate_hz or pre-resample CSVs.")
        if max_rows_per_subject and max_rows_per_subject > 0:
            def _sample(g: pd.DataFrame) -> pd.DataFrame:
                if len(g) <= max_rows_per_subject:
                    return g
                return g.iloc[:max_rows_per_subject]

            df = df.groupby(schema.worker_id, observed=True).apply(_sample).reset_index(drop=True)
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

    df = df.sort_values([schema.worker_id, schema.timestamp]).reset_index(drop=True)
    df[schema.time_idx] = df.groupby(schema.worker_id, observed=True).cumcount().astype(int)
    df.attrs["time_metadata"] = {
        "representation": "raw_signal",
        "source_sampling_rate_hz": WESAD_CHEST_RATE_HZ if use_pickles else None,
        "target_sampling_rate_hz": float(target_sampling_rate_hz) if use_pickles else None,
        "effective_sampling_rate_hz": float(target_sampling_rate_hz) if use_pickles else None,
        "resampling": "linear_interpolation_per_signal" if use_pickles else "csv_provided_timestamps",
        "max_rows_per_subject": max_rows_per_subject,
    }
    return df
