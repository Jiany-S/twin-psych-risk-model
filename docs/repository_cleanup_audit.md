# Repository Cleanup Audit

Freeze point for Prompt 11 final cleanup.

## Freeze Metadata

| Item | Value |
| --- | --- |
| Git commit | `bae5e6322b064e6f1a2806156022a5949b2afcfb` |
| Worktree status | Clean at audit start |
| Python | `Python 3.13.5` |
| Selected installed packages | `matplotlib==3.10.0`, `pytorch-forecasting==1.6.1`, `pytorch-lightning==2.6.0`, `ruff==0.16.0`, `torch==2.10.0`, `xgboost==3.1.3` |
| Python line count before cleanup | `src`: 6798, `scripts`: 1393, `tests`: 1138 |
| Test files collected | 8 files, 47 tests |
| Tracked experiment artifacts | None under `experiments/` |
| Tracked data artifacts | MultiPhysio dataset docs/metadata, `.gitkeep` files, synthetic replay scenario manifest |

## Active CLI Entrypoints

| Entrypoint | Default config | Status |
| --- | --- | --- |
| `python -m src.run_experiment` | `src/config/default.yaml` | Supported legacy benchmark runner |
| `python scripts/run_multiphysio_cv.py` | `src/config/multiphysio_cv.yaml` | Supported MultiPhysio grouped CV runner |
| `python -m src.training.fast_train` | `src/config/fast_wesad.yaml` | Supported WESAD fast detector runner |
| `python -m src.training.tft_train` | `src/config/slow_tft.yaml` | Supported slow multi-horizon runner |
| `python -m src.streaming.replay` | `src/config/streaming_multirate.yaml` | Supported replay runner |
| `python scripts/debug_multiphysio_dataset.py` | `src/config/multiphysio_debug.yaml` | Diagnostic utility |
| `python scripts/debug_tft_dataset.py` | `src/config/wesad_debug_tft.yaml` | Diagnostic utility |
| `python scripts/run_multiphysio_full.py` | `src/config/multiphysio_full.yaml` | Historical/heavier runner; candidate to archive |
| `python scripts/run_multiphysio_smoke.py` | `src/config/multiphysio_smoke.yaml` | Historical smoke wrapper; candidate to consolidate |
| `python scripts/make_paper_summary.py` | run directory input | Historical summary helper; uses deprecated row timing fields |
| `python scripts/prepare_wesad_subset.py` | data output arguments | Dataset preparation utility |

## Config Inheritance

Only replay configs currently use `_base_`:

| Config | Base |
| --- | --- |
| `streaming_physiological.yaml` | `streaming_multirate.yaml` |
| `streaming_physiological_oracle.yaml` | `streaming_physiological.yaml` |
| `streaming_synthetic_physical.yaml` | `streaming_multirate.yaml` |

All other YAML configs are standalone overrides or legacy full configs.

## Cleanup Decision Table

| Path | Purpose | Active callers | Decision | Evidence |
| --- | --- | --- | --- | --- |
| `src/run_experiment.py` | Legacy end-to-end XGBoost/TFT benchmark orchestration | README, default CLI, WESAD pilot scripts | Keep, small cleanup only | Active documented command; large module at 38 KB |
| `src/data/` | Dataset loaders, schema, preprocessing, timing/window utilities | All training runners and tests | Keep, consolidate config/timing validation only | Active imports across `src/run_experiment.py`, fast/slow training, CV |
| `src/data/windowing.py` | Legacy single-target window builder | `src/run_experiment.py`, tests, sanity | Keep with compatibility alias | Active tests use `y_stress`; generic rename can be gradual |
| `src/data/time_semantics.py` | Duration-to-samples temporal contract | `src/run_experiment.py`, tests | Keep | Centralized timing conversion already tested |
| `src/models/` | TFT, TCN, and XGB model definitions | Training runners | Keep | Active imports in fast/slow/XGB runners |
| `src/profiles/worker_profile.py` | Global/calibration/profile feature handling | `src/run_experiment.py`, tests | Keep | Prompt 4 tests validate behavior |
| `src/training/baseline_train.py` | Dummy/logistic/RF baseline suite | `src/run_experiment.py`, tests | Keep | Active baseline tests |
| `src/training/xgb_train.py` | Strong boosted-tree benchmark | `src/run_experiment.py` | Keep, rename internals later if low risk | Still uses `y_stress` names for primary target |
| `src/training/fast_train.py` | WESAD fast causal detector smoke/benchmark | README, replay artifact source | Keep | Active CLI and replay manifest input |
| `src/training/tft_train.py` | Slow multi-horizon forecaster smoke/benchmark | README, replay artifact source | Keep | Active CLI and replay manifest input |
| `src/training/metrics.py` | Shared classification/regression metrics and thresholding | Multiple training runners/tests | Keep | Correct shared home |
| `src/training/plotting.py` | Shared ROC/PR/calibration/comparison plotting | `src/run_experiment.py` | Keep | Single caller but useful boundary for plot code |
| `src/streaming/replay.py` | Replay event generation, execution, metrics, artifact writing | README, tests | Consolidate only where needed | Largest module at 41 KB; artifact pairing is risky |
| `src/streaming/prediction_artifacts.py` | Replay prediction parsing/validation | `src/streaming/replay.py`, tests | Keep and extend for manifest | Existing artifact utility home |
| `src/streaming/fast_detector.py` | Causal streaming detector primitives | `fast_train`, tests | Keep | Active tests cover buffering/serialization |
| `src/safety/` | Physical kernel, adapters, state machine | Replay and safety tests | Keep | Safety invariants actively tested |
| `src/utils/io.py` | YAML load/merge and JSON save | Main runner and scripts | Consolidate config loading here | Replay currently has duplicate `_load_config` |
| `src/utils/artifacts.py` | Hashing/git/latest-run helpers | Replay artifact parsing | Keep, remove unsafe latest-run replay use | `latest_run` acceptable for diagnostics, not real-model replay pairing |
| `src/config/default.yaml` | Base synthetic/default benchmark config | `src.run_experiment`, scripts load base | Keep as base/default |
| `src/config/fast_wesad.yaml` | Fast WESAD smoke/benchmark | README, final gate | Keep |
| `src/config/slow_tft.yaml` | Slow WESAD smoke/benchmark | README, final gate | Keep |
| `src/config/multiphysio_cv.yaml` | MultiPhysio grouped CV benchmark | README, final gate | Keep |
| `src/config/streaming_*.yaml` | Replay configs | README, final gate | Keep, add explicit manifest for real-model replay |
| `src/config/multiphysio_ablation_*.yaml` | Prompt 4 profile ablations | `docs/profile_ablation_report.md` | Keep as documented ablations |
| `src/config/multiphysio_debug.yaml` | Loader diagnostics | `debug_multiphysio_dataset.py` | Keep if diagnostic CLI remains |
| `src/config/multiphysio_full.yaml` | Heavier historical MultiPhysio run | `run_multiphysio_full.py`, weekly report | Archive candidate | Not part of final gate; may be historical |
| `src/config/multiphysio_smoke.yaml` | Historical smoke runner config | smoke script | Archive/consolidate candidate | Final gate uses CV runner instead |
| `src/config/wesad_*pilot*`, `wesad_paper*`, `wesad_kaggle_api.yaml`, `wesad_debug_tft.yaml`, `wesad_tiny.yaml` | Historical/debug WESAD configs | scripts/docs | Archive candidates, preserve reports | Some docs mark timing unreliable |
| `scripts/run_multiphysio_cv.py` | Active MultiPhysio grouped CV | README, final gate | Keep |
| `scripts/run_multiphysio_full.*` | Full MultiPhysio wrapper | weekly report only | Archive candidate | Not final supported command |
| `scripts/run_multiphysio_smoke.*` | Historical smoke wrapper | script docs only | Archive candidate | Superseded by capped CV final command |
| `scripts/run_wesad_pilot.*` | Historical pilot wrappers | AGENTS/docs | Keep or archive with WESAD historical configs | Not final gate but useful history |
| `scripts/debug_*.py` | Diagnostics | Manual use | Keep if config validation passes |
| `scripts/make_paper_summary.py` | Historical report helper | Manual use | Archive candidate | Still reads deprecated row-based timing keys |
| `data/synthetic_replay/scenario_manifest.csv` | Synthetic replay manifest fixture | Replay validator/tests | Keep | Small intentional fixture |
| `data/multiphysio/*` | Dataset docs and small metadata | MultiPhysio loader/docs | Keep | No raw feature table directory tracked |
| `docs/current_pipeline_audit.md`, `experiment_semantics.md`, `time_semantics.md`, `profile_ablation_report.md` | Audits/methodology | README/docs | Keep under audits/methodology |
| `docs/fast_*`, `slow_forecaster.md`, `multiphysio_benchmark.md`, `multirate_architecture.md` | Phase reports | README/docs | Keep, potentially move older smoke reports to archive |
| `docs/final_architecture.md`, `model_comparison.md`, `dataset_limitations.md`, `validated_claims.md`, `reproducibility_checklist.md`, `research_roadmap.md` | Current docs | README | Keep as current source of truth |
| `docs/weekly_reports/` | Historical weekly reports | none active | Archive/history | Preserve with historical banner |
| `notebooks/*.ipynb` | Exploratory prototypes | none detected in active docs/CI | Archive candidate | Not tested; may be stale |
| `.github/workflows/ci.yml` | CI validation | GitHub Actions | Keep/update | Must run pytest, Ruff, sanity |
| `.gitignore` | Generated-file guardrails | Git | Keep/update | Already ignores `experiments/`; missing model extension patterns |
| `requirements.txt` | Runtime dependencies | CI/users | Keep/update | Contains runtime stack |
| `requirements-dev.txt` | Test/lint dependencies | CI | Keep/update | Add timeout if used |
| `pyproject.toml` | Ruff config | CI/local lint | Keep/update | Current focused Ruff gate passes |

## Generated Files Tracked

No files under `experiments/` are tracked. No model checkpoint or cache files are tracked. Intentional small data fixtures/docs are tracked under `data/`.

## Initial Risks

1. Real-model physiological replay still depends on implicit fast/slow artifact discovery and can pair incompatible runs.
2. Full `python -m pytest -q` previously timed out in the tool wrapper despite grouped test success; root cause not yet established at this inventory stage.
3. Many historical configs still contain deprecated or dataset-specific semantics and are documented as unreliable in earlier audits.
4. Several large modules mix orchestration, metric calculation, artifact writing, and CLI behavior.
5. `ruff check src scripts tests` passes only under focused correctness rules; broad style debt remains intentionally outside the current gate.
