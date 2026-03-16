# Experimentation Workflow

This document defines the recommended workflow for contributors and coding agents when running or modifying experiments.

## Standard Workflow
1. Run a fast regression check:
```bash
python scripts/sanity_check.py
```
2. Run one targeted experiment config:
```bash
python -m src.run_experiment --config src/config/default.yaml
```
3. Review generated artifacts in `experiments/runs/<timestamp>/`.
4. Record conclusions in `docs/weekly_reports/YYYY-MM-DD.md`.

## Validation Gates
Use the smallest gate that validates your change.

### Utility/metrics logic changes
```bash
python scripts/sanity_check.py
```

### Pipeline/orchestration changes
```bash
python -m src.run_experiment --config src/config/default.yaml
```

### TFT dataset/windowing changes
```bash
python scripts/debug_tft_dataset.py --config src/config/wesad_debug_tft.yaml
python -m src.run_experiment --config src/config/wesad_debug_tft.yaml
```

## Required Handoff Information
- What changed and why.
- Exact commands executed.
- Pass/fail status.
- Output run directory path(s).
- Skipped checks and reason.

## Improvement Loop
Use this loop after meaningful runs.

1. Observe
- Read `metrics.json`, `results.md`, and the three diagnostic stats files.
- Check AUROC, AUPRC, F1, ECE/Brier, class balance, and window counts.

2. Diagnose
- Pick one likely bottleneck category:
  - data quality
  - split strategy
  - feature engineering
  - model capacity/regularization
  - threshold/calibration policy

3. Plan One Change
- Define one high-impact, low-risk experiment.
- Write a falsifiable expectation (example: "ECE improves by >= 0.02 with AUROC drop <= 0.01").

4. Execute
- Use explicit config deltas and fixed seeds.
- Save outputs in the standard run directory structure.

5. Evaluate
- Compare against baseline:
  - stress: AUROC, AUPRC, F1, ECE
  - comfort (if enabled): RMSE
  - robustness: NaN/Inf diagnostics and class balance sanity

6. Record
- Append/update a dated report in `docs/weekly_reports/` with:
  - hypothesis
  - config delta
  - metric deltas
  - decision (`adopt`, `reject`, `follow-up`)

## Quick Paper Summary
For an existing run:
```bash
python scripts/make_paper_summary.py --run_dir experiments/runs/<timestamp>
```
