# Model Training and Evaluation

This notebook demonstrates the full training pipeline for the TFT model.

## Sections

1. Data Preprocessing
2. Worker Profile Computation
3. Data Normalization
4. Model Training
5. Evaluation Metrics
6. Results Visualization

## Setup

```python
import sys
sys.path.append('../src')

from data.preprocess import TimeSeriesPreprocessor
from data.normalization import ZScoreNormalizer
from profiles.worker_profile import WorkerProfileManager
from training.train_tft import TFTTrainer, TFTConfig
from evaluate import ModelEvaluator
```

## TODO

- [ ] Preprocess sample data
- [ ] Train TFT model
- [ ] Generate predictions
- [ ] Evaluate with AUROC/ECE
- [ ] Visualize calibration curve
