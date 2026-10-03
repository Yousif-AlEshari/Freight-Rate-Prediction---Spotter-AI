"""Fit/transform-safe preprocessing for freight-rate modeling.

The transformer in this module performs domain cleaning and deterministic feature
engineering only. Encoding, scaling, feature selection, and model fitting are
intentionally left for the modeling pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"

REQUIRED_FEATURE_COLUMNS = (
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
)

CATEGORICAL_FEATURES = (
    "pickup",
    "delivery",
    "equipment",
    "route",
)

NUMERIC_FEATURES = (
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "log_distance",
    "weight",
    "weight_missing",
    "weight_was_negative",
    "market_index",
    "market_index_missing",
    "quote_signal",
    "quote_signal_missing",
    "latitude_delta",
    "longitude_delta",
    "haversine_miles",
    "bearing_sin",
    "bearing_cos",
    "days_since_reference",
    "month",
    "quarter",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)


def _require_dataframe(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, received {type(frame).__name__}.")


def _validate_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_features_target(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return independent feature and numeric target copies from labeled data."""
    _require_dataframe(frame)
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is not present.")

    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any() or not np.isfinite(target).all():
        raise ValueError(f"Target column {target_column!r} contains invalid values.")
    if (target <= 0).any():
        raise ValueError(f"Target column {target_column!r} must be strictly positive.")

    return frame.drop(columns=[target_column]).copy(), target.astype(float).copy()


class FreightFeatureEngineer(TransformerMixin, BaseEstimator):
    """Clean freight attributes and create model-ready domain features.

    Parameters
    ----------
    reference_date:
        Stable origin used for the linear time feature. Keeping this fixed avoids
        changing feature values across temporal cross-validation folds.
    missing_category:
        Replacement for missing categorical values.
    """

    def __init__(
        self,
        reference_date: str = "2025-01-01",
        missing_category: str = "__missing__",
    ) -> None:
        self.reference_date = reference_date
        self.missing_category = missing_category

    def fit(self, X: pd.DataFrame, y: object = None) -> FreightFeatureEngineer:
        """Learn fold-local imputation values without using the target."""
        del y
        frame = self._validated_copy(X)
        equipment = self._clean_category(frame["equipment"])
        weight = pd.to_numeric(frame["weight"], errors="coerce").abs()

        self.weight_median_ = float(weight.median())
        if not np.isfinite(self.weight_median_):
            raise ValueError("Cannot fit because weight has no valid numeric values.")

        grouped = pd.DataFrame({"equipment": equipment, "weight": weight}).groupby(
            "equipment", observed=True
        )["weight"]
        self.weight_median_by_equipment_ = grouped.median().to_dict()
        self.market_index_median_ = self._finite_median(frame["market_index"], "market_index")
        self.quote_signal_median_ = self._finite_median(frame["quote_signal"], "quote_signal")
        self.reference_date_ = pd.Timestamp(self.reference_date)
        self.feature_names_out_ = np.asarray(
            (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES), dtype=object
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned cleaning and deterministic feature engineering."""
        check_is_fitted(
            self,
            attributes=[
                "weight_median_",
                "weight_median_by_equipment_",
                "market_index_median_",
                "quote_signal_median_",
                "reference_date_",
                "feature_names_out_",
            ],
        )
        frame = self._validated_copy(X)
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            bad_rows = frame.index[dates.isna()].tolist()[:5]
            raise ValueError(f"Invalid date values at row indices: {bad_rows}")

        result = pd.DataFrame(index=frame.index)
        for column in ("pickup", "delivery", "equipment"):
            result[column] = self._clean_category(frame[column])
        result["route"] = result["pickup"] + " -> " + result["delivery"]

        coordinate_columns = (
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
        )
        for column in coordinate_columns:
            result[column] = self._required_numeric(frame[column], column)

        distance = self._required_numeric(frame["distance"], "distance")
        if (distance <= 0).any():
            raise ValueError("distance must be strictly positive.")
        result["distance"] = distance
        result["log_distance"] = np.log1p(distance)

        raw_weight = pd.to_numeric(frame["weight"], errors="coerce")
        result["weight_missing"] = raw_weight.isna().astype("int8")
        result["weight_was_negative"] = raw_weight.lt(0).fillna(False).astype("int8")
        clean_weight = raw_weight.abs()
        equipment_medians = result["equipment"].map(self.weight_median_by_equipment_)
        result["weight"] = clean_weight.fillna(equipment_medians).fillna(self.weight_median_)

        result["market_index"], result["market_index_missing"] = self._impute_numeric(
            frame["market_index"], self.market_index_median_
        )
        result["quote_signal"], result["quote_signal_missing"] = self._impute_numeric(
            frame["quote_signal"], self.quote_signal_median_
        )

        self._add_geographic_features(result)
        self._add_calendar_features(result, dates)

        return result.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output names for sklearn pipeline introspection."""
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()

    def _validated_copy(self, X: pd.DataFrame) -> pd.DataFrame:
        _require_dataframe(X)
        _validate_columns(X, REQUIRED_FEATURE_COLUMNS)
        return X.loc[:, REQUIRED_FEATURE_COLUMNS].copy()

    def _clean_category(self, values: pd.Series) -> pd.Series:
        cleaned = values.astype("string").str.strip()
        return cleaned.fillna(self.missing_category).replace("", self.missing_category)

    @staticmethod
    def _finite_median(values: pd.Series, column: str) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            raise ValueError(f"Cannot fit because {column} has no valid numeric values.")
        return float(finite.median())

    @staticmethod
    def _required_numeric(values: pd.Series, column: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        invalid = numeric.isna() | ~np.isfinite(numeric)
        if invalid.any():
            bad_rows = values.index[invalid].tolist()[:5]
            raise ValueError(f"Invalid {column} values at row indices: {bad_rows}")
        return numeric

    @staticmethod
    def _impute_numeric(values: pd.Series, median: float) -> tuple[pd.Series, pd.Series]:
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        missing = (numeric.isna() | ~np.isfinite(numeric)).astype("int8")
        numeric = numeric.where(np.isfinite(numeric), np.nan).fillna(median)
        return numeric, missing

    @staticmethod
    def _add_geographic_features(result: pd.DataFrame) -> None:
        lat1 = np.radians(result["pickup_lat"].to_numpy())
        lon1 = np.radians(result["pickup_lon"].to_numpy())
        lat2 = np.radians(result["delivery_lat"].to_numpy())
        lon2 = np.radians(result["delivery_lon"].to_numpy())
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        haversine = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        )
        central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
        bearing = np.arctan2(
            np.sin(delta_lon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon),
        )

        result["latitude_delta"] = result["delivery_lat"] - result["pickup_lat"]
        result["longitude_delta"] = result["delivery_lon"] - result["pickup_lon"]
        result["haversine_miles"] = 3_958.7613 * central_angle
        result["bearing_sin"] = np.sin(bearing)
        result["bearing_cos"] = np.cos(bearing)

    def _add_calendar_features(self, result: pd.DataFrame, dates: pd.Series) -> None:
        day_of_week = dates.dt.dayofweek
        day_of_year = dates.dt.dayofyear

        result["days_since_reference"] = (dates - self.reference_date_).dt.days.astype(float)
        result["month"] = dates.dt.month.astype("int8")
        result["quarter"] = dates.dt.quarter.astype("int8")
        result["day_of_week"] = day_of_week.astype("int8")
        result["day_of_year"] = day_of_year.astype("int16")
        result["is_weekend"] = day_of_week.ge(5).astype("int8")
        result["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        result["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
        result["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
        result["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)


def build_preprocessing_pipeline(
    reference_date: str = "2025-01-01",
) -> Pipeline:
    """Build the domain-preprocessing stage for extension with a later model."""
    return Pipeline(
        steps=[
            (
                "features",
                FreightFeatureEngineer(reference_date=reference_date),
            )
        ]
    )
