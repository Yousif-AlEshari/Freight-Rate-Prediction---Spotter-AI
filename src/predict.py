"""Fit the selected model and create final assessment prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn

from .modeling import selected_model_spec
from .preprocessing import REQUIRED_FEATURE_COLUMNS, split_features_target


PREDICTION_COLUMNS = ("load_id", "predicted_rate")
DECEMBER_COLUMNS = (
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
)


def build_city_coordinate_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one validated latitude/longitude pair per observed city."""
    required = {
        "pickup",
        "pickup_lat",
        "pickup_lon",
        "delivery",
        "delivery_lat",
        "delivery_lon",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing coordinate columns: {sorted(missing)}")

    pickup = frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "latitude", "pickup_lon": "longitude"}
    )
    delivery = frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={
            "delivery": "city",
            "delivery_lat": "latitude",
            "delivery_lon": "longitude",
        }
    )
    coordinates = pd.concat([pickup, delivery], ignore_index=True)
    coordinate_counts = coordinates.groupby("city")[["latitude", "longitude"]].nunique()
    inconsistent = coordinate_counts.gt(1).any(axis=1)
    if inconsistent.any():
        cities = coordinate_counts.index[inconsistent].tolist()
        raise ValueError(f"Cities have inconsistent coordinates: {cities}")
    return coordinates.drop_duplicates("city").set_index("city").sort_index()


def build_december_features(
    december_inputs: pd.DataFrame,
    development: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich fixed December rows with reproducible model inputs."""
    required = {"pickup", "delivery", "distance", "equipment", "weight", "date"}
    missing = required - set(december_inputs.columns)
    if missing:
        raise ValueError(f"December inputs are missing columns: {sorted(missing)}")

    result = december_inputs.loc[:, list(required)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("December inputs contain invalid dates.")

    coordinates = build_city_coordinate_lookup(development)
    for endpoint in ("pickup", "delivery"):
        unknown = sorted(set(result[endpoint]) - set(coordinates.index))
        if unknown:
            raise ValueError(f"No development coordinates found for {endpoint}: {unknown}")
        result[f"{endpoint}_lat"] = result[endpoint].map(coordinates["latitude"])
        result[f"{endpoint}_lon"] = result[endpoint].map(coordinates["longitude"])

    validation_dates = pd.to_datetime(validation["date"], errors="coerce")
    validation_market = pd.to_numeric(validation["market_index"], errors="coerce")
    daily_market = validation_market.groupby(validation_dates).mean()
    result["market_index"] = result["date"].map(daily_market)
    if result["market_index"].isna().any():
        missing_dates = result.loc[result["market_index"].isna(), "date"].dt.date.tolist()
        raise ValueError(f"No validation market index available for dates: {missing_dates}")

    # The selected model excludes quote_signal, but preprocessing keeps a stable schema.
    result["quote_signal"] = np.nan
    return result.loc[:, REQUIRED_FEATURE_COLUMNS]


def create_validation_predictions(
    validation: pd.DataFrame,
    template: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Align finite positive predictions to the authoritative template IDs."""
    if list(template.columns) != list(PREDICTION_COLUMNS):
        raise ValueError("Prediction template must contain load_id,predicted_rate in that order.")
    if "load_id" not in validation:
        raise ValueError("Validation data is missing load_id.")
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("Validation and template IDs must be unique.")

    numeric_predictions = _positive_predictions(predictions, len(validation))
    prediction_by_id = pd.Series(
        numeric_predictions,
        index=validation["load_id"].astype(str),
    )
    template_ids = template["load_id"].astype(str)
    if set(prediction_by_id.index) != set(template_ids):
        raise ValueError("Validation IDs do not match the prediction template.")

    result = template.copy()
    result["predicted_rate"] = template_ids.map(prediction_by_id).round(2)
    if result["predicted_rate"].isna().any():
        raise ValueError("Could not align every validation prediction to the template.")
    return result


def create_december_predictions(
    december_inputs: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Fill December rates while preserving the scorer's required schema."""
    original_columns = [column for column in DECEMBER_COLUMNS if column != "predicted_rate"]
    missing = set(original_columns) - set(december_inputs.columns)
    if missing:
        raise ValueError(f"December inputs are missing columns: {sorted(missing)}")
    numeric_predictions = _positive_predictions(predictions, len(december_inputs))
    result = december_inputs.loc[:, original_columns].copy()
    result["predicted_rate"] = np.round(numeric_predictions, 2)
    return result.loc[:, DECEMBER_COLUMNS]


def _positive_predictions(predictions: np.ndarray, expected_rows: int) -> np.ndarray:
    numeric = np.asarray(predictions, dtype=float)
    if numeric.shape != (expected_rows,):
        raise ValueError(
            f"Expected {expected_rows} one-dimensional predictions, received {numeric.shape}."
        )
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Predictions must be finite and strictly positive.")
    return numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the selected model and create validation and December predictions."
    )
    parser.add_argument("--development", default="data/train-test.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument(
        "--template",
        default="data/validation-predictions-template.csv",
    )
    parser.add_argument("--december-inputs", default="data/december-chart-inputs.csv")
    parser.add_argument("--validation-output", default="validation_predictions.csv")
    parser.add_argument("--december-output", default="december_predictions.csv")
    parser.add_argument("--model-output", default="artifacts/final_model.joblib")
    parser.add_argument("--metadata-output", default="artifacts/final_model_metadata.json")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--exclude-route", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    input_paths = {
        "development": Path(args.development),
        "validation": Path(args.validation),
        "template": Path(args.template),
        "December inputs": Path(args.december_inputs),
    }
    missing_files = [f"{label}: {path}" for label, path in input_paths.items() if not path.is_file()]
    if missing_files:
        raise SystemExit("Missing input files: " + "; ".join(missing_files))

    development = pd.read_csv(input_paths["development"], parse_dates=["date"])
    validation = pd.read_csv(input_paths["validation"], parse_dates=["date"])
    template = pd.read_csv(input_paths["template"])
    december_inputs = pd.read_csv(input_paths["December inputs"])

    X_train, y_train = split_features_target(development)
    spec = selected_model_spec(
        iterations=args.iterations,
        random_seed=args.random_seed,
        include_route=not args.exclude_route,
    )
    model = spec.build()
    print(f"Fitting {spec.name} on {len(X_train):,} development rows...")
    model.fit(X_train, y_train)

    validation_rates = model.predict(validation)
    validation_predictions = create_validation_predictions(
        validation,
        template,
        validation_rates,
    )

    december_features = build_december_features(december_inputs, development, validation)
    december_rates = model.predict(december_features)
    december_predictions = create_december_predictions(december_inputs, december_rates)

    validation_output = Path(args.validation_output)
    december_output = Path(args.december_output)
    model_output = Path(args.model_output)
    metadata_output = Path(args.metadata_output)
    for output in (validation_output, december_output, model_output, metadata_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    validation_predictions.to_csv(validation_output, index=False)
    december_predictions.to_csv(december_output, index=False)
    joblib.dump(model, model_output)

    metadata = {
        "model": spec.name,
        "parameters": spec.parameters,
        "training_rows": len(X_train),
        "training_date_min": str(pd.to_datetime(X_train["date"]).min().date()),
        "training_date_max": str(pd.to_datetime(X_train["date"]).max().date()),
        "model_features": model.model_feature_names_.tolist(),
        "december_market_source": "daily mean from unlabeled validation.csv",
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Created {validation_output} with {len(validation_predictions):,} predictions.")
    print(f"Created {december_output} with {len(december_predictions):,} predictions.")
    print(f"Saved fitted model: {model_output}")
    print(f"Saved model metadata: {metadata_output}")


if __name__ == "__main__":
    main()
