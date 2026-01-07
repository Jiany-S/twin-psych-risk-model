# Twin Psychology Risk Model

A Python repository for risk forecasting using Temporal Fusion Transformers (TFT) with worker-specific profiles based on Ecological Momentary Assessment (EMA) data.

## Overview

This project implements a risk forecasting system that:
- Preprocesses time series data with windowing and train/val/test splits
- Computes worker-specific EMA statistics (mean μ and standard deviation σ)
- Normalizes data using z-score normalization
- Trains Temporal Fusion Transformer models for risk prediction
- Evaluates models using AUROC and Expected Calibration Error (ECE)

## Project Structure

```
twin-psych-risk-model/
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Time series windowing and data splits
│   │   └── normalization.py    # Z-score normalization
│   ├── profiles/
│   │   └── worker_profile.py   # Worker-specific EMA statistics (μ/σ)
│   ├── models/                 # Model architectures (TFT)
│   ├── training/
│   │   └── train_tft.py        # TFT training script
│   ├── utils/                  # Utility functions
│   ├── config/                 # Configuration management
│   └── evaluate.py             # Model evaluation (AUROC/ECE)
├── notebooks/                  # Jupyter notebooks for exploration
├── docs/
│   └── weekly_reports/         # Weekly progress reports
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Preprocessed data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jiany-S/twin-psych-risk-model.git
cd twin-psych-risk-model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Preprocess raw time series data with windowing and splits:

```bash
python src/data/preprocess.py \
    --input data/raw/ema_data.csv \
    --output-dir data/processed \
    --window-size 30 \
    --forecast-horizon 7 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**Arguments:**
- `--input`: Path to raw input data (CSV format)
- `--output-dir`: Directory to save processed data
- `--window-size`: Number of time steps in each window (lookback period)
- `--forecast-horizon`: Number of time steps to forecast ahead
- `--train-ratio`, `--val-ratio`, `--test-ratio`: Data split proportions

### 2. Worker Profile Computation

Compute worker-specific EMA statistics (mean and standard deviation):

```bash
python src/profiles/worker_profile.py \
    --input data/raw/ema_data.csv \
    --output data/processed/worker_profiles.pkl \
    --summary-output data/processed/worker_profiles_summary.csv \
    --worker-id-column worker_id
```

**Arguments:**
- `--input`: Path to EMA data (CSV format)
- `--output`: Path to save worker profiles (pickle format)
- `--summary-output`: Path to save summary statistics (CSV)
- `--worker-id-column`: Name of the column containing worker IDs
- `--feature-columns`: Optional list of feature columns to compute stats for

### 3. Data Normalization

Apply z-score normalization to datasets:

```bash
python src/data/normalization.py \
    --train data/processed/train.npy \
    --val data/processed/val.npy \
    --test data/processed/test.npy \
    --output-dir data/processed \
    --normalizer-path data/processed/normalizer.pkl
```

**Arguments:**
- `--train`, `--val`, `--test`: Paths to data files
- `--output-dir`: Directory to save normalized data
- `--normalizer-path`: Path to save the fitted normalizer
- `--epsilon`: Small constant to avoid division by zero (default: 1e-8)

### 4. Model Training

Train the Temporal Fusion Transformer model:

```bash
python src/training/train_tft.py \
    --train-data data/processed/train_normalized.npy \
    --val-data data/processed/val_normalized.npy \
    --checkpoint-dir checkpoints \
    --hidden-size 64 \
    --num-attention-heads 4 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --max-epochs 100 \
    --early-stopping-patience 10
```

**Arguments:**
- `--train-data`, `--val-data`: Paths to training and validation data
- `--checkpoint-dir`: Directory to save model checkpoints
- `--hidden-size`: Size of hidden layers
- `--num-attention-heads`: Number of attention heads
- `--dropout`: Dropout probability
- `--num-encoder-layers`, `--num-decoder-layers`: Number of layers
- `--learning-rate`: Learning rate for optimizer
- `--batch-size`: Training batch size
- `--max-epochs`: Maximum number of training epochs
- `--early-stopping-patience`: Patience for early stopping
- `--device`: Device to use ('cuda' or 'cpu')

### 5. Model Evaluation

Evaluate model predictions using AUROC and ECE metrics:

```bash
python src/evaluate.py \
    --predictions predictions.npy \
    --labels true_labels.npy \
    --output results/evaluation.json \
    --n-bins 10
```

**Arguments:**
- `--predictions`: Path to predicted probabilities (.npy file)
- `--labels`: Path to true labels (.npy file)
- `--output`: Path to save evaluation results (JSON file)
- `--n-bins`: Number of bins for ECE calculation

## Metrics

### AUROC (Area Under ROC Curve)
Measures the model's ability to distinguish between positive and negative classes. Higher values (closer to 1.0) indicate better performance.

### ECE (Expected Calibration Error)
Measures how well the predicted probabilities match the actual outcomes. Lower values (closer to 0.0) indicate better calibration.

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure Guidelines

- Place data processing code in `src/data/`
- Place model definitions in `src/models/`
- Place training scripts in `src/training/`
- Place utility functions in `src/utils/`
- Place configuration files in `src/config/`
- Use notebooks for exploration and visualization
- Document weekly progress in `docs/weekly_reports/`

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run tests to ensure everything works
4. Submit a pull request

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{twin_psych_risk_model,
  title = {Twin Psychology Risk Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/Jiany-S/twin-psych-risk-model}
}
```