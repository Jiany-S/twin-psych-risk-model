# Slow Multi-Horizon Forecaster

Dataset: WESAD raw physiology. Target: WESAD protocol stress state, not construction risk.
Context seconds: `30.0`. Inference stride seconds: `1.0`. Horizons seconds: `[5.0, 30.0]`.
Loss: BCE with logits for neural classification models; sigmoid is applied once to logits before calibration.

| model         |   horizon_seconds |    auroc |    auprc |       f1 |   fixed_0_5_f1 |   balanced_accuracy |    brier |         ece |   latency_p95_ms |
|:--------------|------------------:|---------:|---------:|---------:|---------------:|--------------------:|---------:|------------:|-----------------:|
| dummy         |                 5 | 0.5      | 0.294176 | 0        |      0         |            0.5      | 0.207637 | 0.000361155 |          0       |
| dummy         |                30 | 0.5      | 0.294176 | 0        |      0         |            0.5      | 0.207637 | 0.000361155 |          0       |
| logistic      |                 5 | 0.796285 | 0.577908 | 0.502157 |      0.52218   |            0.648266 | 0.202742 | 0.178589    |          1.04622 |
| logistic      |                30 | 0.796285 | 0.577908 | 0.502157 |      0.52218   |            0.648266 | 0.202742 | 0.178589    |          1.04622 |
| random_forest |                 5 | 0.964729 | 0.918663 | 0.78519  |      0.799864  |            0.885977 | 0.141761 | 0.160305    |         30.0628  |
| random_forest |                30 | 0.964729 | 0.918663 | 0.78519  |      0.799864  |            0.885977 | 0.141761 | 0.160305    |         30.0628  |
| xgboost       |                 5 | 0.963447 | 0.925767 | 0.786445 |      0.799481  |            0.877274 | 0.100927 | 0.149802    |          1.4965  |
| xgboost       |                30 | 0.963447 | 0.925767 | 0.786445 |      0.799481  |            0.877274 | 0.100927 | 0.149802    |          1.4965  |
| tcn           |                 5 | 0.917362 | 0.82436  | 0.454616 |      0.514198  |            0.5      | 0.487786 | 0.560619    |          2.05054 |
| tcn           |                30 | 0.190232 | 0.192499 | 0.345087 |      0.0335878 |            0.357514 | 0.260769 | 0.336431    |          2.05054 |
| tft           |                 5 | 0.977114 | 0.952016 | 0.454616 |      0.518616  |            0.5      | 0.246005 | 0.415963    |          3.85793 |
| tft           |                30 | 0.977114 | 0.952016 | 0.454616 |      0.52674   |            0.5      | 0.245186 | 0.417929    |          3.85793 |

Attention-style outputs are not produced by this smoke runner; attention, if later extracted from a full TFT implementation, must not be interpreted as causal explanation.