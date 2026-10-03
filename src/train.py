"""Command-line temporal model comparison for the freight-rate assessment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .modeling import DEFAULT_VALIDATION_MONTHS, default_model_specs, evaluate_temporal_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare freight-rate models with expanding-window validation."
    )
    parser.add_argument("--data", default="data/train-test.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--validation-months",
        nargs="+",
        default=list(DEFAULT_VALIDATION_MONTHS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model names to run; omit to evaluate every candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Development data not found: {data_path}")

    frame = pd.read_csv(data_path, parse_dates=["date"])
    specs = default_model_specs(args.iterations, args.random_seed)
    available = {spec.name: spec for spec in specs}
    if args.models:
        unknown = sorted(set(args.models) - set(available))
        if unknown:
            raise SystemExit(
                f"Unknown models: {unknown}. Available models: {sorted(available)}"
            )
        specs = [available[name] for name in args.models]

    fold_results, summary = evaluate_temporal_models(
        frame,
        specs=specs,
        validation_months=tuple(args.validation_months),
        iterations=args.iterations,
        random_seed=args.random_seed,
        verbose=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "temporal_cv_folds.csv"
    summary_path = output_dir / "model_comparison.csv"
    fold_results.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nTemporal validation summary (ranked by mean MAE):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSaved fold results: {fold_path}")
    print(f"Saved model comparison: {summary_path}")


if __name__ == "__main__":
    main()
