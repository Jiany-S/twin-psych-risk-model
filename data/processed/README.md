# Processed Data

This directory contains preprocessed and normalized data ready for model training.

## Files

After running the preprocessing pipeline, you will find:

- `train.npy`: Training data windows
- `val.npy`: Validation data windows
- `test.npy`: Test data windows
- `train_normalized.npy`: Normalized training data
- `val_normalized.npy`: Normalized validation data
- `test_normalized.npy`: Normalized test data
- `normalizer.pkl`: Fitted z-score normalizer
- `worker_profiles.pkl`: Worker-specific EMA statistics
- `worker_profiles_summary.csv`: Summary of worker statistics

## Data Format

The `.npy` files contain numpy arrays with shape:
- `(n_samples, window_size + forecast_horizon, n_features)`

where samples combine both the input window and the target forecast window.
