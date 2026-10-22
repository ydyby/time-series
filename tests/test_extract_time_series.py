"""Regression tests for optional Savitzky-Golay smoothing.

Run with: python -m unittest discover -s tests -v
"""

from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.signal import savgol_filter

from scripts import extract_time_series as extraction


class SmoothingTests(unittest.TestCase):
    def compute(self, values, window, order):
        values = np.asarray(values, dtype=float)
        tifs = [
            extraction.TifInfo(
                Path(f"scene_{i}.tif"), datetime(2020, 1, 1) + timedelta(days=i),
                "ASC", "g1",
            )
            for i in range(len(values))
        ]
        samples = [extraction.RectSample(0, 1, 0, 0, 1, 1)]
        readers = []
        for value in values:
            reader = MagicMock()
            reader.__enter__.return_value.read.return_value = np.full(
                (len(extraction.BAND_NAMES), 1, 1), value, dtype=np.float32,
            )
            readers.append(reader)
        with patch.object(extraction.rasterio, "open", side_effect=readers):
            return extraction.compute_time_series(
                tifs, samples, ["VV_lin", "VH_lin"], window, order,
            )

    def assert_raw(self, frames, stats, values):
        for band, frame in frames.items():
            np.testing.assert_equal(frame.iloc[:, 0].to_numpy(), values)
            self.assertEqual(stats[band]["min"], np.nanmin(values))
            self.assertEqual(stats[band]["max"], np.nanmax(values))
            self.assertEqual(stats[band]["mean"], np.nanmean(values))

    def test_invalid_parameters_skip_filter_and_preserve_output(self):
        cases = [
            (3, 7, 2),    # reported short-series failure
            (4, 4, 2),    # rounding an even window up exceeds scene count
            (9, 3, 3),    # order equals window, with enough valid samples
            (9, 3, 4),    # order exceeds window
            (9, 5, -1),   # negative polynomial order
        ]
        for count, window, order in cases:
            with self.subTest(count=count, window=window, order=order):
                values = np.arange(count, dtype=float)
                with patch.object(
                    extraction, "savgol_filter", wraps=savgol_filter,
                ) as filtering:
                    with self.assertLogs(level="WARNING") as logs:
                        frames, stats = self.compute(values, window, order)
                    filtering.assert_not_called()
                self.assertIn("Skipping smoothing", logs.output[0])
                self.assert_raw(frames, stats, values)

    def test_valid_windows_use_real_filter_for_each_band(self):
        values = np.array([0., 2., 1., 5., 3., 8., 4.])
        for window in (5, 4, 7):
            with self.subTest(window=window):
                frames, stats = self.compute(values, window, 2)
                expected = savgol_filter(values, window + (window % 2 == 0), 2)
                self.assertFalse(np.allclose(expected, values))
                for band, frame in frames.items():
                    np.testing.assert_allclose(frame.iloc[:, 0], expected)
                    self.assertAlmostEqual(stats[band]["mean"], expected.mean())

    def test_disabled_smoothing_preserves_output(self):
        values = np.arange(5, dtype=float)
        for window, order in ((None, None), (0, 2), (2, 2), (5, None)):
            with self.subTest(window=window, order=order):
                with patch.object(extraction, "savgol_filter") as filtering:
                    frames, stats = self.compute(values, window, order)
                    filtering.assert_not_called()
                self.assert_raw(frames, stats, values)

    def test_insufficient_non_nan_samples_still_skip_smoothing(self):
        values = np.array([1., np.nan, 2., np.nan, 3.])
        with patch.object(extraction, "savgol_filter") as filtering:
            with self.assertLogs(level="WARNING"):
                frames, stats = self.compute(values, 5, 2)
            filtering.assert_not_called()
        self.assert_raw(frames, stats, values)


if __name__ == "__main__":
    unittest.main()
