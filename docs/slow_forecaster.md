# Slow Multi-Horizon Forecaster

Dataset: WESAD raw physiology. Target: WESAD protocol stress state, not construction risk.
Context seconds: `30.0`. Inference stride seconds: `1.0`. Horizons seconds: `[5.0, 30.0]`.
Loss: BCE with logits for neural classification models; sigmoid is applied once to logits before calibration.

| model         |   horizon_seconds |    auroc |    auprc |        f1 |   fixed_0_5_f1 |   balanced_accuracy |    brier |        ece |   latency_p95_ms |
|:--------------|------------------:|---------:|---------:|----------:|---------------:|--------------------:|---------:|-----------:|-----------------:|
| dummy         |                 5 | 0.5      | 0.30015  | 0         |      0         |            0.5      | 0.210094 | 0.00579344 |          0       |
| dummy         |                30 | 0.5      | 0.30015  | 0         |      0         |            0.5      | 0.210094 | 0.00579344 |          0       |
| logistic      |                 5 | 0.766675 | 0.692944 | 0.0344828 |      0.0344828 |            0.508772 | 0.281514 | 0.287702   |          1.40236 |
| logistic      |                30 | 0.766675 | 0.692944 | 0.0344828 |      0.0344828 |            0.508772 | 0.281514 | 0.287702   |          1.40236 |
| random_forest |                 5 | 0.593578 | 0.425204 | 0         |      0         |            0.5      | 0.299184 | 0.299006   |         43.2428  |
| random_forest |                30 | 0.593578 | 0.425204 | 0         |      0         |            0.5      | 0.299184 | 0.299006   |         43.2428  |
| xgboost       |                 5 | 0.538355 | 0.385893 | 0.0214876 |      0         |            0.50543  | 0.296753 | 0.297388   |          2.48061 |
| xgboost       |                30 | 0.538355 | 0.385893 | 0.0214876 |      0         |            0.50543  | 0.296753 | 0.297388   |          2.48061 |
| tcn           |                 5 | 0.626837 | 0.329377 | 0         |      0         |            0.5      | 0.287263 | 0.280105   |          1.30652 |
| tcn           |                30 | 0.611645 | 0.319686 | 0         |      0         |            0.5      | 0.291477 | 0.287127   |          1.30652 |
| tft           |                 5 | 0.6301   | 0.366873 | 0         |      0         |            0.5      | 0.288664 | 0.282258   |          3.16768 |
| tft           |                30 | 0.6301   | 0.366873 | 0         |      0         |            0.5      | 0.288664 | 0.28226    |          3.16768 |

Attention-style outputs are not produced by this smoke runner; attention, if later extracted from a full TFT implementation, must not be interpreted as causal explanation.