# Freight Rate Prediction

This repository contains an end-to-end, leakage-aware machine learning workflow for predicting freight rates in a realistic temporal forecasting setting. The project is designed around the Spotter freight-rate assessment: it uses historical load data, time-based validation, robust preprocessing, model comparison, and final inference to generate predictions for future market conditions.

> Note: the underlying data is not committed to the repository for confidentiality reasons.

## Overview

The objective is to estimate `posted_rate` for future freight loads using historical data from January through October 2025 and generate predictions for November and December 2025. The key challenge is not simply building a strong regression model, but doing so without leakage and under realistic temporal drift.

This project emphasizes:

- time-aware model selection instead of random validation splits
- rate-per-mile modeling with business-relevant feature engineering
- geographic and route robustness checks
- reproducible final model training and inference
- contract validation and reporting artifacts

## Repository structure

```text
data/                       Historical and validation inputs
notebooks/eda.ipynb         Exploratory data analysis
src/preprocessing.py        Domain cleaning and feature transformations
src/modeling.py             Model training, metrics, and temporal evaluation
src/train.py                Initial multi-model comparison workflow
src/robustness.py           Sensitivity and cold-start robustness checks
src/predict.py              Final model fitting and prediction generation
src/report.py               Report generation from saved artifacts
score.py                    Local validation/scoring entry point
tests/                      Unit and contract tests
artifacts/                  Model outputs, evaluation metrics, and metadata
scorer_results/             Scorer-generated output charts
reports/                    DOCX/PDF reporting outputs
```

## Project workflow

### 1. Data and validation design

The labeled development data covers January through October 2025, while the unlabeled validation period covers November and December 2025. Random cross-validation would mix time regimes and distort the business problem, so the repository uses expanding-window validation:

| Fold | Train | Validate |
| --- | --- | --- |
| 1 | January-June | July |
| 2 | January-July | August |
| 3 | January-August | September |
| 4 | January-September | October |

All preprocessing parameters are learned only on the training portion of each fold.

### 2. Exploratory analysis

The notebook provides a detailed view of the data, including:

- missingness patterns
- invalid or unrealistic weights
- temporal drift
- target outliers
- categorical coverage
- quote-signal instability
- preprocessing verification

To open the analysis notebook:

```powershell
.\.venv\Scripts\jupyter-lab.exe notebooks\eda.ipynb
```

### 3. Model selection

The initial model comparison is run with:

```powershell
.\.venv\Scripts\python.exe -m src.train --iterations 600
```

The selected model is a CatBoost regressor trained on rate-per-mile with an MAE objective and distance-based sample weights. `quote_signal` is excluded because its behavior changes materially over time.

Robustness checks are run with:

```powershell
.\.venv\Scripts\python.exe -m src.robustness
```

The analysis includes iteration sensitivity and geographic cold-start simulations. Detailed results are documented in `artifacts/model_selection.md`.

### 4. Final prediction generation

After selecting the final configuration, the repository fits the model on all labeled rows and produces the requested prediction files:

```powershell
.\.venv\Scripts\python.exe -m src.predict --iterations 400
```

This generates:

- `validation_predictions.csv`
- `december_predictions.csv`
- `artifacts/final_model.joblib`
- `artifacts/final_model_metadata.json`

The December scenario includes rows without coordinates or market features, so city coordinates and market indices are recovered from the labeled data and validation covariates.

### 5. Scoring and report generation

The repository includes a contract validator and scorer:

```powershell
.\.venv\Scripts\python.exe score.py `
  --predictions validation_predictions.csv `
  --december-predictions december_predictions.csv
```

This validates the output schema and creates the chart in `scorer_results/`.

To generate the final reporting artifacts:

```powershell
.\.venv\Scripts\python.exe -m src.report `
  --title "Freight Rate Machine Learning Report"
```

This produces:

- `reports/freight_rate_machine_learning_report.docx`
- `reports/freight_rate_machine_learning_report.pdf`

## Setup

Python 3.12 was used for the solution run.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows, the project can be run directly through the virtual environment without activation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Selected validation result

The temporal backtests used for model selection showed the following average monthly performance across the four folds:

| Metric | Result |
| --- | ---: |
| Mean MAE | $114.20 |
| Worst-month MAE | $148.13 |
| Mean median absolute error | $42.41 |
| Mean WAPE | 4.78% |
| Mean RMSE | $627.62 |

These values reflect historical time-based validation rather than hidden future labels and are intended to estimate expected model quality under realistic market drift.

## Key technical notes

- The workflow is built to avoid leakage by fitting preprocessing and feature logic inside each training fold.
- Geographic and route cold-start scenarios are explicitly tested to understand robustness under unseen markets and lane patterns.
- The selected model is a strong, operationally defensible default for freight rate estimation in this setting.

## Submission artifacts

This repository is structured to support a submission-ready workflow, including:

- versioned source code
- reproducible model training and inference
- validation outputs
- scorer-generated visuals
- final reporting artifacts
- contract-valid prediction files

## License

This repository is intended for internal assessment and project use. Please check the project or organization policy for licensing and distribution terms before public reuse.
