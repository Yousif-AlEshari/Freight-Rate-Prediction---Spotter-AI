from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.predict import (
    DECEMBER_COLUMNS,
    build_december_features,
    create_december_predictions,
    create_validation_predictions,
)
from src.preprocessing import REQUIRED_FEATURE_COLUMNS


class PredictionOutputTests(unittest.TestCase):
    def test_validation_predictions_follow_template_order(self) -> None:
        validation = pd.DataFrame({"load_id": ["TE-000001", "TE-000002"]})
        template = pd.DataFrame(
            {
                "load_id": ["TE-000002", "TE-000001"],
                "predicted_rate": [np.nan, np.nan],
            }
        )

        result = create_validation_predictions(
            validation,
            template,
            np.array([100.126, 200.456]),
        )

        self.assertEqual(result["load_id"].tolist(), ["TE-000002", "TE-000001"])
        self.assertEqual(result["predicted_rate"].tolist(), [200.46, 100.13])

    def test_december_predictions_preserve_required_schema(self) -> None:
        inputs = pd.DataFrame(
            {
                "pickup": ["Lexington"],
                "delivery": ["Fort Wayne"],
                "distance": [360],
                "equipment": ["Dry Van"],
                "weight": [32_000],
                "date": ["2025-12-01"],
                "predicted_rate": [np.nan],
            }
        )

        result = create_december_predictions(inputs, np.array([812.346]))

        self.assertEqual(list(result.columns), list(DECEMBER_COLUMNS))
        self.assertEqual(result.loc[0, "predicted_rate"], 812.35)


class DecemberEnrichmentTests(unittest.TestCase):
    def test_builds_coordinates_and_daily_market_features(self) -> None:
        development = pd.DataFrame(
            {
                "pickup": ["Lexington", "Fort Wayne"],
                "pickup_lat": [36.99152, 41.31561],
                "pickup_lon": [-84.99876, -85.36206],
                "delivery": ["Fort Wayne", "Lexington"],
                "delivery_lat": [41.31561, 36.99152],
                "delivery_lon": [-85.36206, -84.99876],
            }
        )
        validation = pd.DataFrame(
            {
                "date": ["2025-12-01", "2025-12-01", "2025-12-02"],
                "market_index": [0.9, 1.0, 1.1],
            }
        )
        december = pd.DataFrame(
            {
                "pickup": ["Lexington", "Lexington"],
                "delivery": ["Fort Wayne", "Fort Wayne"],
                "distance": [360, 360],
                "equipment": ["Dry Van", "Dry Van"],
                "weight": [32_000, 32_000],
                "date": ["2025-12-01", "2025-12-02"],
            }
        )

        result = build_december_features(december, development, validation)

        self.assertEqual(list(result.columns), list(REQUIRED_FEATURE_COLUMNS))
        self.assertAlmostEqual(result.loc[0, "pickup_lat"], 36.99152)
        self.assertAlmostEqual(result.loc[0, "delivery_lon"], -85.36206)
        self.assertEqual(result["market_index"].tolist(), [0.95, 1.1])
        self.assertTrue(result["quote_signal"].isna().all())


if __name__ == "__main__":
    unittest.main()
