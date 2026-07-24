# Fast WESAD Diagnostics

Run directory: `experiments\runs\fast_wesad_20260723_200607`

Baseline maps to class 0 and stress maps to class 1 in `src/data/load_wesad.py`.
Amusement handling: `stress_include_amusement=True`.
Scikit-learn probabilities are selected by locating class `1` in `classes_`; TCN probabilities are `sigmoid(logit)` for class 1.

## Split Quality

| split      | worker_id   |   row_count |   window_count |   positive_count |   negative_count |   prevalence |   window_positive_count |   window_negative_count |   window_prevalence | protocol_composition                                  |   first_timestamp |   last_timestamp |
|:-----------|:------------|------------:|---------------:|-----------------:|-----------------:|-------------:|------------------------:|------------------------:|--------------------:|:------------------------------------------------------|------------------:|-----------------:|
| test       | S8          |        8836 |           8803 |             2680 |             6156 |     0.303305 |                    2669 |                    6134 |            0.303192 | {"amusement": 1480, "baseline": 4676, "stress": 2680} |            158.75 |          3962.5  |
| test       | S9          |        8788 |           8755 |             2580 |             6208 |     0.293582 |                    2569 |                    6186 |            0.293432 | {"amusement": 1488, "baseline": 4720, "stress": 2580} |             75.25 |          4430    |
| train      | S2          |        8484 |           8451 |             2460 |             6024 |     0.289958 |                    2449 |                    6002 |            0.289788 | {"amusement": 1448, "baseline": 4576, "stress": 2460} |            306.75 |          5125.5  |
| train      | S3          |        8620 |           8587 |             2560 |             6060 |     0.296984 |                    2549 |                    6038 |            0.296844 | {"amusement": 1500, "baseline": 4560, "stress": 2560} |            353    |          5378.75 |
| train      | S4          |        8660 |           8627 |             2540 |             6120 |     0.293303 |                    2529 |                    6098 |            0.293149 | {"amusement": 1488, "baseline": 4632, "stress": 2540} |            285.5  |          4248.25 |
| train      | S5          |        8868 |           8835 |             2580 |             6288 |     0.290934 |                    2569 |                    6266 |            0.290775 | {"amusement": 1496, "baseline": 4792, "stress": 2580} |            279.5  |          4247.25 |
| validation | S6          |        8808 |           8775 |             2600 |             6208 |     0.295186 |                    2589 |                    6186 |            0.295043 | {"amusement": 1488, "baseline": 4720, "stress": 2600} |            642.75 |          5922.5  |
| validation | S7          |        8792 |           8759 |             2560 |             6232 |     0.291174 |                    2549 |                    6210 |            0.291015 | {"amusement": 1488, "baseline": 4744, "stress": 2560} |            100.75 |          3736.5  |

## Orientation

| model    |    auroc |    auprc | orientation_diagnostics                                                                              | model_class_order   |
|:---------|---------:|---------:|:-----------------------------------------------------------------------------------------------------|:--------------------|
| dummy    | 0.5      | 0.298326 | {'auroc_p': 0.5, 'auroc_one_minus_p': 0.5, 'orientation_warning': ''}                                | [0, 1]              |
| logistic | 0.558855 | 0.437723 | {'auroc_p': 0.5588551971606095, 'auroc_one_minus_p': 0.44114480283939045, 'orientation_warning': ''} | [0, 1]              |
| tcn      | 0.622576 | 0.324546 | {'auroc_p': 0.6225759760714658, 'auroc_one_minus_p': 0.37742400068431, 'orientation_warning': ''}    | [0, 1]              |

## Temporal Alignment

`diagnostic_predictions.csv` includes worker ID, protocol label, window start/end timestamps, prediction timestamp, target timestamp, target value, and predicted probability. Windows crossing protocol boundaries or timestamp gaps are excluded.
