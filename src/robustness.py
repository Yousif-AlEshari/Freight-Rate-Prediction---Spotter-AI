"""Sensitivity and geographic cold-start checks for the selected model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .modeling import (
    DEFAULT_VALIDATION_MONTHS,
    evaluate_temporal_models,
    regression_metrics,
    selected_model_spec,
    temporal_masks,
)
from .preprocessing import split_features_target


@dataclass(frozen=True)
class CityColdStartSplit:
    """Masks and held-out cities for one geographic stress-test repeat."""

    repeat: int
    held_out_cities: tuple[str, ...]
    train_mask: pd.Series
    validation_mask: pd.Series


def evaluate_iteration_sensitivity(
    frame: pd.DataFrame,
    iteration_values: tuple[int, ...] = (400, 600, 800),
    validation_months: tuple[str, ...] = DEFAULT_VALIDATION_MONTHS,
    random_seed: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare only the selected model while varying boosting iterations."""
    invalid = [value for value in iteration_values if value <= 0]
    if invalid:
        raise ValueError(f"Iteration values must be positive: {invalid}")

    specs = [
        selected_model_spec(
            iterations=value,
            random_seed=random_seed,
            name=f"selected_rpm_mae_{value}_iterations",
        )
        for value in iteration_values
    ]
    fold_results, summary = evaluate_temporal_models(
        frame,
        specs=specs,
        validation_months=validation_months,
        random_seed=random_seed,
        verbose=verbose,
    )
    iterations_by_name = {spec.name: spec.parameters["iterations"] for spec in specs}
    fold_results.insert(
        1,
        "iterations",
        fold_results["model"].map(iterations_by_name).astype(int),
    )
    summary.insert(
        1,
        "iterations",
        summary["model"].map(iterations_by_name).astype(int),
    )
    return fold_results, summary


def build_city_cold_start_splits(
    frame: pd.DataFrame,
    validation_month: str = "2025-10",
    held_out_city_count: int = 8,
    repeats: int = 3,
    random_seed: int = 42,
) -> list[CityColdStartSplit]:
    """Remove selected cities from historical training and test them one month ahead."""
    required = {"pickup", "delivery", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing cold-start columns: {sorted(missing)}")
    if held_out_city_count <= 0 or repeats <= 0:
        raise ValueError("held_out_city_count and repeats must be positive.")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    base_train_mask, month_mask = temporal_masks(dates, validation_month)
    historical = frame.loc[base_train_mask]
    city_pool = sorted(set(historical["pickup"]) | set(historical["delivery"]))
    if held_out_city_count >= len(city_pool):
        raise ValueError("held_out_city_count must be smaller than the historical city count.")

    generator = np.random.default_rng(random_seed)
    splits = []
    for repeat in range(1, repeats + 1):
        held_out = tuple(
            sorted(generator.choice(city_pool, size=held_out_city_count, replace=False))
        )
        historical_has_city = frame["pickup"].isin(held_out) | frame["delivery"].isin(
            held_out
        )
        train_mask = base_train_mask & ~historical_has_city
        validation_mask = month_mask & historical_has_city
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"Cold-start repeat {repeat} produced an empty partition.")
        splits.append(
            CityColdStartSplit(
                repeat=repeat,
                held_out_cities=held_out,
                train_mask=train_mask,
                validation_mask=validation_mask,
            )
        )
    return splits


def evaluate_city_cold_start(
    frame: pd.DataFrame,
    iterations: int = 400,
    validation_month: str = "2025-10",
    held_out_city_count: int = 8,
    repeats: int = 3,
    random_seed: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate route inclusion when validation cities are entirely unseen."""
    X, y = split_features_target(frame)
    splits = build_city_cold_start_splits(
        X,
        validation_month=validation_month,
        held_out_city_count=held_out_city_count,
        repeats=repeats,
        random_seed=random_seed,
    )
    records = []

    for split in splits:
        X_train = X.loc[split.train_mask].copy()
        y_train = y.loc[split.train_mask].copy()
        X_validation = X.loc[split.validation_mask].copy()
        y_validation = y.loc[split.validation_mask].copy()

        for include_route in (True, False):
            label = "with_route" if include_route else "without_route"
            if verbose:
                print(
                    f"Fitting cold-start repeat {split.repeat} {label} "
                    f"({len(X_train):,} train / {len(X_validation):,} validation)"
                )
            estimator = selected_model_spec(
                iterations=iterations,
                random_seed=random_seed,
                include_route=include_route,
                name=f"selected_{label}",
            ).build()
            started = perf_counter()
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_validation)
            elapsed = perf_counter() - started
            records.append(
                {
                    "model": f"selected_{label}",
                    "include_route": include_route,
                    "repeat": split.repeat,
                    "validation_month": validation_month,
                    "held_out_cities": "|".join(split.held_out_cities),
                    "train_rows": len(X_train),
                    "validation_rows": len(X_validation),
                    "fit_predict_seconds": elapsed,
                    **regression_metrics(y_validation, predictions),
                }
            )

    fold_results = pd.DataFrame.from_records(records)
    summary = (
        fold_results.groupby(["model", "include_route"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            worst_mae=("mae", "max"),
            mean_rmse=("rmse", "mean"),
            mean_median_ae=("median_ae", "mean"),
            mean_wape=("wape", "mean"),
            mean_rmsle=("rmsle", "mean"),
            mean_bias=("bias", "mean"),
            total_seconds=("fit_predict_seconds", "sum"),
        )
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )
    return fold_results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected-model iteration and geographic robustness checks."
    )
    parser.add_argument("--data", default="data/train-test.csv")
    parser.add_argument("--output-dir", default="artifacts/robustness")
    parser.add_argument("--iteration-values", nargs="+", type=int, default=[400, 600, 800])
    parser.add_argument("--validation-months", nargs="+", default=list(DEFAULT_VALIDATION_MONTHS))
    parser.add_argument("--cold-start-iterations", type=int, default=400)
    parser.add_argument("--cold-start-month", default="2025-10")
    parser.add_argument("--held-out-city-count", type=int, default=8)
    parser.add_argument("--cold-start-repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-iterations", action="store_true")
    parser.add_argument("--skip-cold-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Development data not found: {data_path}")
    if args.skip_iterations and args.skip_cold_start:
        raise SystemExit("Both robustness checks were skipped.")

    frame = pd.read_csv(data_path, parse_dates=["date"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_iterations:
        folds, summary = evaluate_iteration_sensitivity(
            frame,
            iteration_values=tuple(args.iteration_values),
            validation_months=tuple(args.validation_months),
            random_seed=args.random_seed,
            verbose=True,
        )
        folds.to_csv(output_dir / "iteration_sensitivity_folds.csv", index=False)
        summary.to_csv(output_dir / "iteration_sensitivity_summary.csv", index=False)
        print("\nIteration sensitivity:")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    if not args.skip_cold_start:
        folds, summary = evaluate_city_cold_start(
            frame,
            iterations=args.cold_start_iterations,
            validation_month=args.cold_start_month,
            held_out_city_count=args.held_out_city_count,
            repeats=args.cold_start_repeats,
            random_seed=args.random_seed,
            verbose=True,
        )
        folds.to_csv(output_dir / "city_cold_start_folds.csv", index=False)
        summary.to_csv(output_dir / "city_cold_start_summary.csv", index=False)
        print("\nGeographic cold-start summary:")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    print(f"\nSaved robustness results under: {output_dir}")


if __name__ == "__main__":
    main()
