# Final Cleanup Report

## Scope

Prompt 11 focused on release-quality cleanup without adding datasets, model architectures, hardware integration, APIs, or new safety features.

## Files Deleted

No files were permanently deleted. Unsupported historical files were archived instead of removed.

## Files Archived

| Destination | Contents | Evidence |
| --- | --- | --- |
| `docs/archive/configs/` | 9 historical/debug configs: MultiPhysio full/smoke and older WESAD debug/pilot/paper/tiny configs | Not referenced by current README, CI, tests, or final supported commands |
| `docs/archive/scripts/` | 10 unsupported wrapper/summary/debug scripts | Superseded by supported CLIs in `docs/configuration_matrix.md` |
| `docs/archive/notebooks/` | 2 exploratory notebooks | Not part of CI or supported commands |
| `docs/archive/weekly_reports/` | 3 weekly report files | Historical provenance only |
| `docs/archive/experimentation_workflow.md` | Older workflow doc | Referenced archived scripts/configs |

Archived Markdown files include a historical-source banner where applicable. `docs/archive/README.md` identifies the archive as non-current.

## Modules Consolidated

| Area | Change |
| --- | --- |
| YAML `_base_` loading | Moved recursive config loading to `src/utils/io.py::load_config_with_base`; replay now reuses it instead of a local duplicate merge helper. |
| Prediction artifact pairing | Kept parsing/validation in `src/streaming/prediction_artifacts.py`; added manifest validation there rather than adding a parallel framework. |

## Duplicate Implementations Removed

- Removed replay-local recursive config merge logic.
- Removed implicit latest-run fast/slow artifact discovery from physiological replay.

## Deprecated Config Fields Removed

Active configs under `src/config/` do not contain:

- `window_length`
- `horizon_steps`
- `window_step`
- `downsample_factor`

This is enforced by `tests/test_repository_structure.py`.

## Active Config Count

| Point | Count |
| --- | ---: |
| Before cleanup | 24 YAML files under `src/config/` |
| After cleanup | 15 YAML files under `src/config/`, including `replay_manifest_example.yaml` |

## Python Line Count

| Area | Before | After |
| --- | ---: | ---: |
| `src/` | 6798 | 6946 |
| `scripts/` | 1393 | 881 |
| `tests/` | 1138 | 1349 |

The active script surface shrank substantially. `src/` and `tests/` increased because the real-model replay manifest contract and compatibility tests were added.

## Test Count And Duration

Final full-suite result:

```text
python -m pytest -q
58 passed in 56.92s
```

Earlier diagnostic run with durations:

```text
47 passed in 25.67s
```

The prior full-suite timeout was not reproducible after cleanup; `python -m pytest -q` now completes as one command. `pytest-timeout` is declared as a dev dependency and configured with a 60-second per-test timeout.

## Ruff Result

```text
ruff check src scripts tests
All checks passed!
```

Ruff remains focused on undefined-name correctness checks (`F821`, `F822`, `F823`). Broader import-order/style modernization remains technical debt.

## Pip Check Result

```text
python -m pip check
No broken requirements found.
```

## CLI Smoke Results

All required help commands returned exit code 0:

- `python -m src.run_experiment --help`
- `python scripts/run_multiphysio_cv.py --help`
- `python -m src.training.fast_train --help`
- `python -m src.training.tft_train --help`
- `python -m src.streaming.replay --help`

## Smoke Experiment Results

| Command | Result artifact |
| --- | --- |
| `python scripts/run_multiphysio_cv.py --config src/config/multiphysio_cv.yaml` | `experiments/runs/multiphysio_cv_20260724_113527` |
| `python -m src.training.fast_train --config src/config/fast_wesad.yaml` | `experiments/runs/fast_wesad_20260724_114207` |
| `python -m src.training.tft_train --config src/config/slow_tft.yaml` | `experiments/runs/slow_tft_20260724_114303` |
| `python -m src.streaming.replay --config src/config/streaming_physiological.yaml` | `experiments/runs/streaming_real_model_physiological_replay_20260724_114606` |
| `python -m src.streaming.replay --config src/config/streaming_physiological_oracle.yaml` | `experiments/runs/streaming_oracle_label_policy_simulation_20260724_114636` |
| `python -m src.streaming.replay --config src/config/streaming_synthetic_physical.yaml` | `experiments/runs/streaming_synthetic_physical_safety_simulation_20260724_114636` |
| `python -m src.streaming.replay --config src/config/streaming_multirate.yaml` | `experiments/runs/streaming_combined_integration_simulation_20260724_114636` |

## Artifact-Pairing Strategy

Real-model physiological replay now requires `replay.manifest_path`.

The active WESAD fast and slow training configs write exact manifest fragments to:

```text
experiments/runs/replay_manifest_wesad_smoke.yaml
```

Replay validates:

- manifest version and required sections;
- metadata file existence;
- target name;
- positive-class orientation;
- shared held-out subjects;
- timestamp unit;
- stream representation;
- config hashes;
- monotonic timestamps;
- duplicate prediction keys;
- exact prediction-timestamp overlap;
- target agreement where fast/slow horizons overlap.

The final accepted report was written to:

```text
experiments/runs/streaming_real_model_physiological_replay_20260724_114606/artifact_pairing_report.json
```

It accepted `S8` and `S9`, aligned 3988 prediction origins, dropped 13570 fast-only origins, and used `exact_prediction_timestamp`.

## Remaining Compatibility Aliases

- `WindowedData.y_stress` and related `windows_y_stress.npy` naming remain for legacy `src.run_experiment` compatibility.
- XGBoost internals still use stress-specific names where the historical API is stress-focused.
- `replay_manifest_example.yaml` is an example schema, not an executable config.

## Known Remaining Technical Debt

- `src/streaming/replay.py`, `src/run_experiment.py`, `src/training/tft_train.py`, `src/training/fast_train.py`, and `scripts/run_multiphysio_cv.py` remain large modules.
- Broad Ruff style checks are not enabled yet.
- Current WESAD slow smoke metrics changed after aligning held-out subjects with fast replay; they should not be compared directly to earlier slow smoke numbers.
- Archived scripts/configs are preserved for provenance but are not maintained.
- Optional heavy dependencies for TFT and Kaggle access are still environment-sensitive.

## Behavior Changes

- Real-model physiological replay no longer discovers latest fast/slow runs.
- `slow_tft.yaml` now uses the same held-out subject split as `fast_wesad.yaml` for replay compatibility: train `S2-S5`, validation `S6-S7`, test `S8-S9`.
- `streaming_physiological.yaml` requires the explicit manifest generated by fast/slow training.
- Unsupported wrappers and historical configs were moved to `docs/archive/`.

## Metrics Changed Due To Correctness Bug

The slow WESAD smoke metrics changed because the previous slow config used test subjects `S5/S6`, while fast used `S8/S9`. That incompatibility made real-model physiological replay scientifically unsafe. The corrected slow config now aligns held-out replay subjects with the fast detector.

## Final Supported Commands

```bash
python -m pytest -q
ruff check src scripts tests
python scripts/sanity_check.py
python scripts/run_multiphysio_cv.py --config src/config/multiphysio_cv.yaml
python -m src.training.fast_train --config src/config/fast_wesad.yaml
python -m src.training.tft_train --config src/config/slow_tft.yaml
python -m src.streaming.replay --config src/config/streaming_physiological.yaml
python -m src.streaming.replay --config src/config/streaming_physiological_oracle.yaml
python -m src.streaming.replay --config src/config/streaming_synthetic_physical.yaml
python -m src.streaming.replay --config src/config/streaming_multirate.yaml
```
