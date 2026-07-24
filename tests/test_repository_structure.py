from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from src.utils.io import load_config_with_base


ACTIVE_CONFIGS = [
    "default.yaml",
    "fast_wesad.yaml",
    "slow_tft.yaml",
    "multiphysio_cv.yaml",
    "multiphysio_ablation_global.yaml",
    "multiphysio_ablation_calibration.yaml",
    "multiphysio_ablation_calibration_features.yaml",
    "multiphysio_ablation_calibration_metadata.yaml",
    "streaming_multirate.yaml",
    "streaming_physiological.yaml",
    "streaming_physiological_oracle.yaml",
    "streaming_synthetic_physical.yaml",
    "synthetic_debug.yaml",
]

DEPRECATED_TIMING_FIELDS = {"window_length", "horizon_steps", "window_step", "downsample_factor"}


def test_active_configs_load_and_avoid_deprecated_timing_fields():
    for name in ACTIVE_CONFIGS:
        cfg = load_config_with_base(Path("src/config") / name)
        assert isinstance(cfg, dict)
        assert cfg.get("targets")
        text = repr(cfg)
        for field in DEPRECATED_TIMING_FIELDS:
            assert field not in text
        stream = cfg.get("stream", {})
        if stream:
            assert stream.get("representation") in {"raw_signal", "precomputed_features"}
        if cfg.get("replay", {}).get("probability_source") == "saved_predictions":
            assert cfg["replay"].get("manifest_path")


def test_documented_cli_entrypoints_expose_help():
    commands = [
        [sys.executable, "-m", "src.run_experiment", "--help"],
        [sys.executable, "scripts/run_multiphysio_cv.py", "--help"],
        [sys.executable, "-m", "src.training.fast_train", "--help"],
        [sys.executable, "-m", "src.training.tft_train", "--help"],
        [sys.executable, "-m", "src.streaming.replay", "--help"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()


def test_unit_tests_do_not_require_generated_run_directories():
    generated_run_path = "experiments" + "/runs"
    offenders = []
    for path in Path("tests").glob("test_*.py"):
        if path.name != "test_repository_structure.py" and generated_run_path in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []
