# Reproducibility Checklist

1. Install dependencies with `pip install -r requirements-dev.txt`.
2. Run `python -m pytest -q`. Last Prompt 11 full run: 58 tests passed in 56.12 seconds on CPU with Python 3.13.5.
3. Run `python scripts/sanity_check.py`.
4. Run capped MultiPhysio CV: `python scripts/run_multiphysio_cv.py --config src/config/multiphysio_cv.yaml`.
5. Run fast WESAD benchmark: `python -m src.training.fast_train --config src/config/fast_wesad.yaml`.
6. Run slow WESAD forecaster: `python -m src.training.tft_train --config src/config/slow_tft.yaml`.
7. Run saved-prediction physiological replay: `python -m src.streaming.replay --config src/config/streaming_physiological.yaml`. This requires the explicit manifest written by steps 5 and 6.
8. Run oracle-label policy simulation: `python -m src.streaming.replay --config src/config/streaming_physiological_oracle.yaml`.
9. Run synthetic physical replay: `python -m src.streaming.replay --config src/config/streaming_synthetic_physical.yaml`.
10. Run combined integration replay: `python -m src.streaming.replay --config src/config/streaming_multirate.yaml`.

Ruff and pytest-timeout are configured in `pyproject.toml`; run `ruff check src scripts tests` when Ruff is installed.
