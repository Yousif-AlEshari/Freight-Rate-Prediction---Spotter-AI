# Temporal Model Selection

Reproduce the initial comparison from the repository root:

```bash
python -m src.train --iterations 600
```

## Validation design

Each fold trains on all prior rows and validates on the next complete month:

| Fold | Training period | Validation period |
|---|---|---|
| 1 | January-June | July |
| 2 | January-July | August |
| 3 | January-August | September |
| 4 | January-September | October |

Preprocessing is refit within every training fold. No validation rows contribute to
imputation or model fitting.

## Result

`catboost_rpm_mae_no_quote` is the leading candidate. A subsequent iteration
sensitivity check selected 400 boosting iterations:

| Metric | Result |
|---|---:|
| Mean MAE | $114.20 |
| Worst-month MAE | $148.13 |
| Mean RMSE | $627.62 |
| Mean median absolute error | $42.41 |
| Mean WAPE | 4.78% |
| Mean bias | -$21.33 |

The 400-iteration model improved mean MAE by 17.7% over the median-RPM business
baseline. It also narrowly improved mean and worst-month MAE over 600 and 800
iterations while reducing runtime.

## Decision

The selected model predicts rate per mile with CatBoost's MAE objective, weights
training rows by distance to align the objective with dollar MAE, and reconstructs
the final dollar rate by multiplying predicted RPM by distance.

The route feature is retained. In three October cold-start simulations that removed
eight cities entirely from historical training, retaining route produced $146.70
mean MAE versus $149.35 without route. Coordinates and distance remain available as
fallbacks for unseen categories.

`quote_signal` is excluded. Adding it increased mean MAE from $114.36 to $118.82
and increased August MAE from $83.69 to $103.03. This agrees with the EDA finding
that its relationship with rate per mile changes by month and becomes effectively
uncorrelated in August.

The direct RMSE model performed worst because approximately 1.4% of labels are
extreme, apparently unpredictable target outliers. On regular validation targets,
the selected model's mean MAE was $61.20; on outlier targets it was $3,879.17.
The large outlier error is shared by every candidate and supports a robust objective
instead of fitting their magnitude.

Detailed outputs are stored in:

- `artifacts/temporal_cv_folds.csv`
- `artifacts/model_comparison.csv`
- `artifacts/robustness/iteration_sensitivity_summary.csv`
- `artifacts/robustness/city_cold_start_summary.csv`
