# Configuration Matrix

| Config | Purpose | Dataset | Target | Entry command | Scientific status |
| --- | --- | --- | --- | --- | --- |
| `default.yaml` | Base synthetic benchmark | Synthetic | Synthetic `stress` primary target | `python -m src.run_experiment --config src/config/default.yaml` | Development smoke only |
| `synthetic_debug.yaml` | Small synthetic debug run | Synthetic | Synthetic `stress` | `python -m src.run_experiment --config src/config/synthetic_debug.yaml` | Debug only |
| `multiphysio_cv.yaml` | Grouped subject CV benchmark | MultiPhysio 60-second precomputed features | Configured questionnaire targets, including STAI stress and NASA workload | `python scripts/run_multiphysio_cv.py --config src/config/multiphysio_cv.yaml` | Current MultiPhysio benchmark |
| `multiphysio_debug.yaml` | MultiPhysio loader diagnostics | MultiPhysio 60-second precomputed features | STAI-derived `y_stress` | `python scripts/debug_multiphysio_dataset.py --config src/config/multiphysio_debug.yaml` | Diagnostic only |
| `multiphysio_ablation_global.yaml` | Profile ablation: global normalization | MultiPhysio | STAI-derived `y_stress` | `python -m src.run_experiment --config src/config/multiphysio_ablation_global.yaml` | Documented ablation |
| `multiphysio_ablation_calibration.yaml` | Profile ablation: calibration normalization | MultiPhysio | STAI-derived `y_stress` | `python -m src.run_experiment --config src/config/multiphysio_ablation_calibration.yaml` | Documented ablation |
| `multiphysio_ablation_calibration_features.yaml` | Profile ablation: calibration features | MultiPhysio | STAI-derived `y_stress` | `python -m src.run_experiment --config src/config/multiphysio_ablation_calibration_features.yaml` | Documented ablation |
| `multiphysio_ablation_calibration_metadata.yaml` | Profile ablation: calibration plus real metadata | MultiPhysio | STAI-derived `y_stress` | `python -m src.run_experiment --config src/config/multiphysio_ablation_calibration_metadata.yaml` | Documented ablation |
| `fast_wesad.yaml` | Fast causal detector smoke/benchmark | WESAD raw physiology | Protocol stress, class 1 = stress | `python -m src.training.fast_train --config src/config/fast_wesad.yaml` | Current WESAD fast benchmark |
| `slow_tft.yaml` | Slow multi-horizon forecaster smoke/benchmark | WESAD raw physiology | Protocol stress forecast | `python -m src.training.tft_train --config src/config/slow_tft.yaml` | Current WESAD slow benchmark |
| `streaming_physiological.yaml` | Real-model physiological replay | Saved WESAD fast/slow prediction artifacts | Protocol stress predictions from explicit manifest | `python -m src.streaming.replay --config src/config/streaming_physiological.yaml` | Engineering replay of held-out model artifacts |
| `streaming_physiological_oracle.yaml` | Oracle-label policy simulation | WESAD labels | Protocol stress labels converted to oracle probabilities | `python -m src.streaming.replay --config src/config/streaming_physiological_oracle.yaml` | Policy-only simulation, not model performance |
| `streaming_synthetic_physical.yaml` | Synthetic physical safety simulation | Synthetic physical scenarios | No physiological target | `python -m src.streaming.replay --config src/config/streaming_synthetic_physical.yaml` | Software invariant validation only |
| `streaming_multirate.yaml` | Combined synthetic integration simulation | Synthetic physical and ML scenarios | Synthetic probabilities | `python -m src.streaming.replay --config src/config/streaming_multirate.yaml` | Integration simulation only |
| `replay_manifest_example.yaml` | Manifest schema example | N/A | N/A | Referenced by docs only | Example, not an executable config |

Historical/debug configs not listed here are preserved for provenance but are not current supported benchmark entrypoints.
