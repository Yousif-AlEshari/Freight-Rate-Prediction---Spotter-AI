from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.modeling import (
    CatBoostFreightRegressor,
    MedianRatePerMileRegressor,
    regression_metrics,
    temporal_masks,
)


def make_freight_frame(rows: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    index = np.arange(rows)
    distance = 100.0 + index * 8.0
    equipment = np.resize(np.array(["Dry Van", "Reefer", "Flatbed"]), rows)
    pickup = np.resize(np.array(["Austin", "Boston", "Chicago", "Denver"]), rows)
    delivery = np.resize(np.array(["Miami", "Seattle", "Dallas", "Atlanta"]), rows)
    weight = 20_000.0 + index * 200.0

    frame = pd.DataFrame(
        {
            "load_id": [f"TR-{number:06d}" for number in range(1, rows + 1)],
            "pickup": pickup,
            "delivery": delivery,
            "pickup_lat": 30.0 + index % 5,
            "pickup_lon": -100.0 + index % 7,
            "delivery_lat": 35.0 + index % 6,
            "delivery_lon": -90.0 + index % 8,
            "distance": distance,
            "equipment": equipment,
            "weight": weight,
            "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "market_index": 0.9 + (index % 10) / 100.0,
            "quote_signal": 2.0 + (index % 7) / 100.0,
        }
    )
    premium = pd.Series(equipment).map({"Dry Van": 0.0, "Flatbed": 0.15, "Reefer": 0.25})
    rpm = 1.8 + premium.to_numpy() + weight / 200_000.0
    target = pd.Series(distance * rpm, index=frame.index, name="posted_rate")
    return frame, target


class RegressionMetricsTests(unittest.TestCase):
    def test_metrics_have_expected_values(self) -> None:
        metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))

        self.assertAlmostEqual(metrics["mae"], 15.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(250.0))
        self.assertAlmostEqual(metrics["median_ae"], 15.0)
        self.assertAlmostEqual(metrics["wape"], 0.1)
        self.assertAlmostEqual(metrics["bias"], -5.0)

    def test_metrics_reject_non_positive_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            regression_metrics(np.array([100.0]), np.array([0.0]))


class TemporalSplitTests(unittest.TestCase):
    def test_temporal_masks_do_not_cross_validation_boundary(self) -> None:
        dates = pd.Series(pd.to_datetime(["2025-06-30", "2025-07-01", "2025-07-31", "2025-08-01"]))
        train_mask, validation_mask = temporal_masks(dates, "2025-07")

        self.assertEqual(train_mask.tolist(), [True, False, False, False])
        self.assertEqual(validation_mask.tolist(), [False, True, True, False])


class ModelTests(unittest.TestCase):
    def test_median_rpm_baseline_handles_unseen_equipment(self) -> None:
        frame, target = make_freight_frame(30)
        model = MedianRatePerMileRegressor(n_distance_bins=3).fit(frame, target)
        unseen = frame.iloc[:2].copy()
        unseen["equipment"] = "Unknown"

        predictions = model.predict(unseen)

        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions > 0).all())

    def test_catboost_rpm_model_predicts_positive_rates(self) -> None:
        frame, target = make_freight_frame(80)
        model = CatBoostFreightRegressor(
            target_mode="rpm",
            loss_function="MAE",
            include_quote_signal=False,
            iterations=20,
            depth=4,
            learning_rate=0.1,
            random_seed=7,
            thread_count=1,
        ).fit(frame.iloc[:60], target.iloc[:60])

        predictions = model.predict(frame.iloc[60:])

        self.assertEqual(predictions.shape, (20,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions > 0).all())
        self.assertNotIn("quote_signal", model.model_feature_names_)
        self.assertNotIn("quote_signal_missing", model.model_feature_names_)

    def test_catboost_can_exclude_route_feature(self) -> None:
        frame, target = make_freight_frame(50)
        model = CatBoostFreightRegressor(
            target_mode="rpm",
            loss_function="MAE",
            include_quote_signal=False,
            include_route=False,
            iterations=5,
            depth=3,
            thread_count=1,
        ).fit(frame.iloc[:40], target.iloc[:40])

        self.assertNotIn("route", model.model_feature_names_)
        self.assertNotIn("route", model.categorical_features_)


if __name__ == "__main__":
    unittest.main()
