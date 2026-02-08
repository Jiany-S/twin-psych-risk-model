"""Signal feature extraction for windowed physiology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import find_peaks, welch


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum((x - x_mean) * (values - values.mean())) / denom)


def _hrv_time_domain(ecg_window: np.ndarray, sampling_rate_hz: float) -> dict[str, float]:
    if len(ecg_window) < 3:
        return {"meanHR": np.nan, "RMSSD": np.nan, "SDNN": np.nan, "pNN50": np.nan}
    # Approximate beat intervals from peaks in ECG-like signal.
    peaks, _ = find_peaks(ecg_window, distance=max(1, int(0.3 * sampling_rate_hz)))
    if len(peaks) < 3:
        return {"meanHR": np.nan, "RMSSD": np.nan, "SDNN": np.nan, "pNN50": np.nan}
    rr = np.diff(peaks) / max(sampling_rate_hz, 1e-6)
    rr_ms = rr * 1000.0
    mean_hr = 60.0 / max(np.mean(rr), 1e-6)
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else np.nan
    sdnn = np.std(rr_ms)
    pnn50 = np.mean(np.abs(diff_rr) > 50.0) if len(diff_rr) > 0 else np.nan
    return {"meanHR": float(mean_hr), "RMSSD": float(rmssd), "SDNN": float(sdnn), "pNN50": float(pnn50)}


def _hrv_freq_domain(rr_seconds: np.ndarray) -> dict[str, float]:
    if len(rr_seconds) < 8:
        return {"LF": np.nan, "HF": np.nan, "LF_HF": np.nan}
    freqs, psd = welch(rr_seconds, fs=4.0, nperseg=min(64, len(rr_seconds)))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else np.nan
    hf = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else np.nan
    ratio = lf / hf if hf is not None and hf > 0 else np.nan
    return {"LF": float(lf), "HF": float(hf), "LF_HF": float(ratio)}


def extract_window_features(
    ecg_window: np.ndarray,
    eda_window: np.ndarray,
    temp_window: np.ndarray,
    sampling_rate_hz: float,
    include_freq_domain: bool,
    scr_threshold: float,
    min_scr_distance: int,
) -> dict[str, float]:
    features: dict[str, float] = {}

    # ECG / HRV features
    hrv_td = _hrv_time_domain(ecg_window, sampling_rate_hz)
    features.update(hrv_td)
    if include_freq_domain:
        peaks, _ = find_peaks(ecg_window, distance=max(1, int(0.3 * sampling_rate_hz)))
        if len(peaks) >= 3:
            rr = np.diff(peaks) / max(sampling_rate_hz, 1e-6)
            features.update(_hrv_freq_domain(rr))
        else:
            features.update({"LF": np.nan, "HF": np.nan, "LF_HF": np.nan})

    # EDA features (simple tonic/phasic decomposition)
    if len(eda_window) > 3:
        kernel = max(3, int(min(len(eda_window) - 1, sampling_rate_hz * 2)))
        tonic = np.convolve(eda_window, np.ones(kernel) / kernel, mode="same")
    else:
        tonic = eda_window
    phasic = eda_window - tonic
    scr_peaks, props = find_peaks(phasic, height=scr_threshold, distance=max(1, min_scr_distance))
    scr_amp = props["peak_heights"] if "peak_heights" in props else np.array([], dtype=float)
    features["scl_mean"] = float(np.mean(tonic))
    features["scl_var"] = float(np.var(tonic))
    features["scr_count"] = float(len(scr_peaks))
    features["scr_mean_amp"] = float(np.mean(scr_amp)) if len(scr_amp) > 0 else 0.0

    # Temperature features
    features["temp_mean"] = float(np.mean(temp_window))
    features["temp_std"] = float(np.std(temp_window))
    features["temp_slope"] = _slope(temp_window)
    return features


def impute_with_train_statistics(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = X_train.copy()
    val = X_val.copy()
    test = X_test.copy()
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for arr in (train, val, test):
        mask = np.isnan(arr)
        if np.any(mask):
            arr[mask] = np.take(med, np.where(mask)[1])
    return train, val, test
