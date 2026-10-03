"""Models and temporal evaluation utilities for freight-rate prediction."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .preprocessing import (
    CATEGORICAL_FEATURES,
    FreightFeatureEngineer,
    split_features_target,
)


DEFAULT_VALIDATION_MONTHS = ("2025-07", "2025-08", "2025-09", "2025-10")
VALID_TARGET_MODES = {"direct", "rpm"}


@dataclass(frozen=True)
class ModelSpec:
    """Named model factory configuration used by temporal evaluation."""

    name: str
    estimator_class: type[BaseEstimator]
    parameters: dict[str, Any]

    def build(self) -> BaseEstimator:
        return self.estimator_class(**self.parameters)


class MedianRatePerMileRegressor(RegressorMixin, BaseEstimator):
    """Business baseline using median RPM by equipment and distance band."""

    def __init__(self, n_distance_bins: int = 6, minimum_prediction: float = 1.0) -> None:
        self.n_distance_bins = n_distance_bins
        self.minimum_prediction = minimum_prediction

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> MedianRatePerMileRegressor:
        self._validate_frame(X)
        if self.n_distance_bins < 2:
            raise ValueError("n_distance_bins must be at least 2.")

        distance = self._positive_numeric(X["distance"], "distance")
        target = self._positive_array(y, "target")
        if len(distance) != len(target):
            raise ValueError("X and y must contain the same number of rows.")

        rpm = target / distance.to_numpy()
        quantiles = np.linspace(0.0, 1.0, self.n_distance_bins + 1)
        interior_edges = np.unique(np.quantile(distance, quantiles))[1:-1]
        self.distance_edges_ = np.concatenate(([-np.inf], interior_edges, [np.inf]))

        equipment = X["equipment"].astype("string").fillna("__missing__")
        distance_bin = pd.cut(distance, bins=self.distance_edges_, labels=False)
        lookup_frame = pd.DataFrame(
            {"equipment": equipment.to_numpy(), "distance_bin": distance_bin, "rpm": rpm}
        )
        grouped = lookup_frame.groupby(
            ["equipment", "distance_bin"], observed=True
        )["rpm"].median()
        self.group_median_rpm_ = grouped.to_dict()
        self.equipment_median_rpm_ = lookup_frame.groupby(
            "equipment", observed=True
        )["rpm"].median().to_dict()
        self.global_median_rpm_ = float(np.median(rpm))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=[
                "distance_edges_",
                "group_median_rpm_",
                "equipment_median_rpm_",
                "global_median_rpm_",
            ],
        )
        self._validate_frame(X)
        distance = self._positive_numeric(X["distance"], "distance")
        equipment = X["equipment"].astype("string").fillna("__missing__")
        distance_bin = pd.cut(distance, bins=self.distance_edges_, labels=False)

        rpm_predictions = []
        for equipment_value, bin_value in zip(equipment, distance_bin, strict=True):
            group_key = (equipment_value, bin_value)
            rpm_predictions.append(
                self.group_median_rpm_.get(
                    group_key,
                    self.equipment_median_rpm_.get(
                        equipment_value,
                        self.global_median_rpm_,
                    ),
                )
            )

        predictions = np.asarray(rpm_predictions) * distance.to_numpy()
        return np.maximum(predictions, self.minimum_prediction)

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        missing = {"distance", "equipment"} - set(X.columns)
        if missing:
            raise ValueError(f"Missing required baseline columns: {sorted(missing)}")

    @staticmethod
    def _positive_numeric(values: pd.Series, label: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ValueError(f"{label} must contain finite, strictly positive values.")
        return numeric

    @staticmethod
    def _positive_array(values: pd.Series | np.ndarray, label: str) -> np.ndarray:
        numeric = np.asarray(values, dtype=float)
        if numeric.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional.")
        if not np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ValueError(f"{label} must contain finite, strictly positive values.")
        return numeric


class CatBoostFreightRegressor(RegressorMixin, BaseEstimator):
    """CatBoost model with fold-local freight preprocessing and target handling."""

    def __init__(
        self,
        target_mode: str = "direct",
        loss_function: str = "MAE",
        include_quote_signal: bool = False,
        include_route: bool = True,
        distance_weighted_rpm: bool = True,
        iterations: int = 600,
        depth: int = 7,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 5.0,
        random_seed: int = 42,
        thread_count: int = -1,
        minimum_prediction: float = 1.0,
    ) -> None:
        self.target_mode = target_mode
        self.loss_function = loss_function
        self.include_quote_signal = include_quote_signal
        self.include_route = include_route
        self.distance_weighted_rpm = distance_weighted_rpm
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.minimum_prediction = minimum_prediction

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> CatBoostFreightRegressor:
        if self.target_mode not in VALID_TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {sorted(VALID_TARGET_MODES)}, "
                f"received {self.target_mode!r}."
            )
        target = MedianRatePerMileRegressor._positive_array(y, "target")
        if len(X) != len(target):
            raise ValueError("X and y must contain the same number of rows.")

        self.preprocessor_ = FreightFeatureEngineer()
        prepared = self.preprocessor_.fit_transform(X)
        prepared = self._select_model_features(prepared)
        distance = prepared["distance"].to_numpy(dtype=float)

        if self.target_mode == "rpm":
            model_target = target / distance
            sample_weight = distance if self.distance_weighted_rpm else None
        else:
            model_target = target
            sample_weight = None

        self.model_ = CatBoostRegressor(
            loss_function=self.loss_function,
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            allow_writing_files=False,
            verbose=False,
        )
        self.model_.fit(
            prepared,
            model_target,
            cat_features=self.categorical_features_,
            sample_weight=sample_weight,
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.model_feature_names_ = np.asarray(prepared.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=["preprocessor_", "model_", "model_feature_names_"],
        )
        prepared = self.preprocessor_.transform(X)
        prepared = self._select_model_features(prepared)
        model_prediction = np.asarray(self.model_.predict(prepared), dtype=float)
        if self.target_mode == "rpm":
            model_prediction = model_prediction * prepared["distance"].to_numpy(dtype=float)
        return np.maximum(model_prediction, self.minimum_prediction)

    def _select_model_features(self, prepared: pd.DataFrame) -> pd.DataFrame:
        selected = prepared.copy()
        if not self.include_quote_signal:
            selected = selected.drop(columns=["quote_signal", "quote_signal_missing"])
        if not self.include_route:
            selected = selected.drop(columns=["route"])
        self.categorical_features_ = [
            column for column in CATEGORICAL_FEATURES if column in selected.columns
        ]
        for column in self.categorical_features_:
            selected[column] = selected[column].astype(str)
        return selected


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate scale, percentage, central, and bias diagnostics."""
    actual = MedianRatePerMileRegressor._positive_array(y_true, "actual target")
    predicted = MedianRatePerMileRegressor._positive_array(y_predicted, "predictions")
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have equal length.")

    error = predicted - actual
    absolute_error = np.abs(error)
    squared_error = np.square(error)
    log_error = np.log1p(predicted) - np.log1p(actual)
    denominator = np.sum(np.abs(actual))

    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "median_ae": float(np.median(absolute_error)),
        "wape": float(np.sum(absolute_error) / denominator),
        "rmsle": float(np.sqrt(np.mean(np.square(log_error)))),
        "bias": float(np.mean(error)),
    }


def default_model_specs(iterations: int = 600, random_seed: int = 42) -> list[ModelSpec]:
    """Return the controlled baseline, objective, target, and quote ablations."""
    selected = selected_model_spec(iterations=iterations, random_seed=random_seed)
    common = {
        "iterations": iterations,
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": random_seed,
    }
    return [
        ModelSpec(
            name="median_rpm_baseline",
            estimator_class=MedianRatePerMileRegressor,
            parameters={"n_distance_bins": 6},
        ),
        ModelSpec(
            name="catboost_direct_rmse_no_quote",
            estimator_class=CatBoostFreightRegressor,
            parameters={
                **common,
                "target_mode": "direct",
                "loss_function": "RMSE",
                "include_quote_signal": False,
            },
        ),
        ModelSpec(
            name="catboost_direct_mae_no_quote",
            estimator_class=CatBoostFreightRegressor,
            parameters={
                **common,
                "target_mode": "direct",
                "loss_function": "MAE",
                "include_quote_signal": False,
            },
        ),
        selected,
        ModelSpec(
            name="catboost_rpm_mae_with_quote",
            estimator_class=CatBoostFreightRegressor,
            parameters={
                **common,
                "target_mode": "rpm",
                "loss_function": "MAE",
                "include_quote_signal": True,
                "distance_weighted_rpm": True,
            },
        ),
    ]


def selected_model_spec(
    iterations: int = 400,
    random_seed: int = 42,
    include_route: bool = True,
    name: str = "catboost_rpm_mae_no_quote",
) -> ModelSpec:
    """Build the selected robust RPM model configuration."""
    return ModelSpec(
        name=name,
        estimator_class=CatBoostFreightRegressor,
        parameters={
            "target_mode": "rpm",
            "loss_function": "MAE",
            "include_quote_signal": False,
            "include_route": include_route,
            "distance_weighted_rpm": True,
            "iterations": iterations,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "random_seed": random_seed,
        },
    )


def temporal_masks(
    dates: pd.Series,
    validation_month: str,
) -> tuple[pd.Series, pd.Series]:
    """Create an expanding training window and one-month future holdout."""
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("date contains invalid values.")
    period = pd.Period(validation_month, freq="M")
    validation_start = period.start_time
    validation_end = (period + 1).start_time
    train_mask = parsed_dates.lt(validation_start)
    validation_mask = parsed_dates.ge(validation_start) & parsed_dates.lt(validation_end)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"No rows available for temporal fold {validation_month}.")
    return train_mask, validation_mask


def evaluate_temporal_models(
    frame: pd.DataFrame,
    specs: list[ModelSpec] | None = None,
    validation_months: tuple[str, ...] = DEFAULT_VALIDATION_MONTHS,
    iterations: int = 600,
    random_seed: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model candidates under expanding-window monthly holdouts."""
    X, y = split_features_target(frame)
    dates = pd.to_datetime(X["date"], errors="coerce")
    model_specs = specs or default_model_specs(iterations, random_seed)
    records: list[dict[str, Any]] = []

    for validation_month in validation_months:
        train_mask, validation_mask = temporal_masks(dates, validation_month)
        X_train = X.loc[train_mask].copy()
        X_validation = X.loc[validation_mask].copy()
        y_train = y.loc[train_mask].copy()
        y_validation = y.loc[validation_mask].copy()

        train_routes = set(zip(X_train["pickup"], X_train["delivery"], strict=True))
        validation_routes = list(
            zip(X_validation["pickup"], X_validation["delivery"], strict=True)
        )
        seen_route = np.asarray([route in train_routes for route in validation_routes])

        training_rpm = y_train.to_numpy() / X_train["distance"].to_numpy()
        q1, q3 = np.quantile(training_rpm, [0.25, 0.75])
        rpm_iqr = q3 - q1
        validation_rpm = y_validation.to_numpy() / X_validation["distance"].to_numpy()
        regular_target = (validation_rpm >= q1 - 3 * rpm_iqr) & (
            validation_rpm <= q3 + 3 * rpm_iqr
        )

        for spec in model_specs:
            if verbose:
                print(
                    f"Fitting {spec.name} for {validation_month} "
                    f"({len(X_train):,} train / {len(X_validation):,} validation)"
                )
            estimator = spec.build()
            started = perf_counter()
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_validation)
            fit_seconds = perf_counter() - started

            row: dict[str, Any] = {
                "model": spec.name,
                "validation_month": validation_month,
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "seen_route_pct": float(np.mean(seen_route)),
                "target_outlier_pct": float(np.mean(~regular_target)),
                "fit_predict_seconds": fit_seconds,
                **regression_metrics(y_validation, predictions),
                "mae_seen_route": _segment_mae(y_validation, predictions, seen_route),
                "mae_unseen_route": _segment_mae(y_validation, predictions, ~seen_route),
                "mae_regular_target": _segment_mae(
                    y_validation, predictions, regular_target
                ),
                "mae_outlier_target": _segment_mae(
                    y_validation, predictions, ~regular_target
                ),
            }
            records.append(row)

    fold_results = pd.DataFrame.from_records(records)
    summary = summarize_results(fold_results)
    return fold_results, summary


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate temporal-fold metrics and rank candidates by mean MAE."""
    required = {"model", "mae", "rmse", "median_ae", "wape", "rmsle", "bias"}
    missing = required - set(fold_results.columns)
    if missing:
        raise ValueError(f"Fold results are missing columns: {sorted(missing)}")

    summary = fold_results.groupby("model", as_index=False).agg(
        folds=("validation_month", "count"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        worst_mae=("mae", "max"),
        mean_rmse=("rmse", "mean"),
        mean_median_ae=("median_ae", "mean"),
        mean_wape=("wape", "mean"),
        mean_rmsle=("rmsle", "mean"),
        mean_bias=("bias", "mean"),
        mean_mae_seen_route=("mae_seen_route", "mean"),
        mean_mae_unseen_route=("mae_unseen_route", "mean"),
        mean_mae_regular_target=("mae_regular_target", "mean"),
        mean_mae_outlier_target=("mae_outlier_target", "mean"),
        total_seconds=("fit_predict_seconds", "sum"),
    )
    summary["mae_rank"] = summary["mean_mae"].rank(method="min").astype(int)
    return summary.sort_values(["mae_rank", "mean_rmse"]).reset_index(drop=True)


def _segment_mae(
    y_true: pd.Series | np.ndarray,
    y_predicted: pd.Series | np.ndarray,
    mask: np.ndarray,
) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_predicted, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("nan")
    return float(np.mean(np.abs(predicted[selected] - actual[selected])))
