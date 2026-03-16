# AGENTS.md

## Purpose
This file defines how coding agents should operate in `twin-psych-risk-model`.
The goal is fast, safe iteration on cognitive risk forecasting experiments without data leakage or reproducibility drift.

## Scope And Priority
1. Follow explicit user instructions first.
2. Follow this file for repo-specific behavior.
3. Keep changes minimal, testable, and reversible.
4. Prefer deterministic workflows and logged artifacts.

## Repository Snapshot
- Main entrypoint: `python -m src.run_experiment --config <yaml>`
- Core package: `src/`
- Configs: `src/config/*.yaml`
- Utility scripts: `scripts/`
- Outputs: `experiments/runs/<timestamp>/`
- Key run artifacts:
  - `metrics.json`
  - `results.md`
  - `config_resolved.yaml`
  - `profile_feature_stats.json`
  - `numeric_feature_stats.json`
  - `engineered_feature_stats.json`
  - `processed/*.npy`, `processed/*.csv`
  - `plots/*.png`

## Agent Operating Rules

### 1) Start With Fast Context
- Read only files relevant to the task before editing.
- For model/data changes, inspect at minimum:
  - `src/run_experiment.py`
  - related module(s) in `src/data`, `src/training`, `src/models`
  - active config in `src/config/`
- Avoid broad refactors unless requested.

### 2) Protect Scientific Validity
- Do not introduce target leakage.
- Keep chronological split guarantees intact.
- Keep subject-holdout split disjoint when enabled.
- Preserve train-only profile fitting behavior unless task explicitly changes methodology.

### 3) Reproducibility Is Required
- Keep seeded behavior stable unless user requests changes.
- Do not silently change default config semantics.
- If changing defaults, explain expected metric impact and migration path.

### 4) Data Safety
- Never commit credentials, tokens, or local secrets.
- Kaggle credentials must remain external (`~/.kaggle/kaggle.json` or env vars).
- Do not add large raw data files to git.

### 5) Performance And Cost Discipline
- Use synthetic/small configs for fast verification first.
- Run heavier WESAD jobs only when required.
- If a task can be validated with one targeted script, do not run full pipelines by default.

## Standard Execution Playbooks

### Quick Sanity (preferred first)
```bash
python scripts/sanity_check.py
```

### End-To-End Run
```bash
python -m src.run_experiment --config src/config/default.yaml
```

### TFT Dataset Debug
```bash
python scripts/debug_tft_dataset.py --config src/config/wesad_debug_tft.yaml
```

### WESAD Pilot (Profiles On/Off)
```bash
# Linux/macOS
bash scripts/run_wesad_pilot.sh

# Windows
scripts\run_wesad_pilot.bat
```

### Paper Summary From Existing Run
```bash
python scripts/make_paper_summary.py --run_dir experiments/runs/<timestamp>
```

## Validation Gates Before Handoff
Run the smallest gate that proves the change:
1. For utility or metric logic changes: run `python scripts/sanity_check.py`.
2. For pipeline/orchestration changes: run one config via `src.run_experiment`.
3. For TFT dataset/windowing changes: run `scripts/debug_tft_dataset.py` plus one experiment run.

Handoff must include:
- What changed and why.
- Exact command(s) executed.
- Pass/fail status.
- Output artifact path(s).
- Any skipped checks and reason.

## Editing Guidelines
- Preserve module boundaries (`data`, `training`, `models`, `profiles`, `utils`).
- Keep functions focused and typed when practical.
- Add short comments only for non-obvious logic.
- Do not rewrite unrelated files.
- Keep public CLI behavior backward compatible unless requested.

## Failure Handling
If a run fails:
1. Capture the exact failing command and stack trace location.
2. Reproduce with smallest config/input.
3. Patch minimally.
4. Re-run the relevant validation gate.
5. Record residual risk if not fully resolved.

## Continuous Improvement Loop
Use this loop after every meaningful run or merged change.

### Loop Steps
1. Observe
- Read `metrics.json`, `results.md`, and diagnostic stats JSON files.
- Check class balance, AUROC/AUPRC, calibration (Brier/ECE), and window counts.

2. Diagnose
- Identify the primary bottleneck category:
  - data quality
  - split quality
  - feature engineering
  - model capacity/regularization
  - threshold/calibration policy

3. Plan One Change
- Define one high-impact, low-risk next experiment.
- State a falsifiable expectation (example: "ECE decreases by >=0.02 without AUROC drop >0.01").

4. Execute
- Run with an explicit config diff and fixed seed.
- Store outputs under normal run directories.

5. Evaluate
- Compare against previous best on:
  - stress: AUROC, AUPRC, F1, ECE
  - comfort (if enabled): RMSE
  - robustness: class balance sanity, NaN/Inf diagnostics

6. Record
- Add a short dated note in `docs/weekly_reports/YYYY-MM-DD.md` with:
  - hypothesis
  - config delta
  - key metrics deltas
  - decision: adopt/reject/needs follow-up

### Loop Guardrails
- Change one variable family at a time when possible.
- Do not claim wins from single unstable runs with tiny test splits.
- Favor improvements that preserve interpretability and leakage safety.

## Definition Of Done
A task is done when all are true:
- Requested code/docs updates are implemented.
- Relevant validation gate(s) passed, or failure is explicitly documented.
- No unintended changes in unrelated files.
- Results and next-step risks are clearly communicated.

## Reference Formats Consulted
This structure was aligned to current public agent-instruction patterns:
- OpenAI Codex repository `AGENTS.md`: https://github.com/openai/codex/blob/main/AGENTS.md
- `agents.md` open format specification: https://github.com/agentsmd/agents.md
- Cursor project rules documentation: https://docs.cursor.com/context/rules-for-ai
- Anthropic Claude Code guidance (project memory/instructions): https://www.anthropic.com/engineering/claude-code-best-practices
