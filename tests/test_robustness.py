from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.robustness import build_city_cold_start_splits


class CityColdStartSplitTests(unittest.TestCase):
    def test_held_out_cities_are_absent_from_training(self) -> None:
        rows = 120
        cities = np.array(["A", "B", "C", "D", "E", "F"])
        frame = pd.DataFrame(
            {
                "pickup": np.resize(cities, rows),
                "delivery": np.resize(np.roll(cities, 1), rows),
                "date": pd.date_range("2025-09-01", periods=rows, freq="12h"),
            }
        )

        splits = build_city_cold_start_splits(
            frame,
            validation_month="2025-10",
            held_out_city_count=2,
            repeats=2,
            random_seed=7,
        )

        self.assertEqual(len(splits), 2)
        for split in splits:
            training = frame.loc[split.train_mask]
            validation = frame.loc[split.validation_mask]
            held_out = set(split.held_out_cities)
            self.assertTrue(held_out.isdisjoint(training["pickup"]))
            self.assertTrue(held_out.isdisjoint(training["delivery"]))
            self.assertTrue(
                (
                    validation["pickup"].isin(held_out)
                    | validation["delivery"].isin(held_out)
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
