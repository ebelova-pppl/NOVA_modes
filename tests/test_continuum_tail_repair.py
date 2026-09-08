"""Regression controls for shared terminal continuum repair."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cont_features import _repair_joint_monotonic_datcon_tail, load_datcon_for_mode


class ContinuumTailRepairTests(unittest.TestCase):
    def setUp(self):
        self.low = np.array([0.8, 0.85, 0.82, 0.84, 0.7, 5.2, 11.0, 17.0, np.nan])
        self.high = np.array([3.5, 3.6, 3.55, 3.7, 5.4, 10.1, 15.6, 21.0, np.nan])
        self.r = 0.85 + 0.005 * np.arange(len(self.low))

    def test_sustained_rise_backtracks_and_preserves_interior_and_missing_data(self):
        original = [self.low**2, self.high**2]
        repaired = _repair_joint_monotonic_datcon_tail(*original, self.r)
        for raw, fixed in zip(original, repaired):
            np.testing.assert_array_equal(fixed[:5], raw[:5])
            np.testing.assert_allclose(fixed[5:8], raw[4])
            np.testing.assert_array_equal(np.isnan(fixed), np.isnan(raw))
        np.testing.assert_array_equal(original[0], self.low**2)

    def test_gentle_single_boundary_and_reversing_tails_are_not_repaired(self):
        cases = [
            (self.low / 100, self.high / 100),
            (np.ones_like(self.low), self.high),
            (np.r_[self.low[:7], 0.5, np.nan], np.r_[self.high[:7], 4.0, np.nan]),
        ]
        for low, high in cases:
            with self.subTest(low=low):
                original = (low**2, high**2)
                repaired = _repair_joint_monotonic_datcon_tail(*original, self.r)
                for raw, fixed in zip(original, repaired):
                    np.testing.assert_array_equal(fixed, raw)

    def test_reference_break_and_single_step_are_not_sustained_rises(self):
        for index in (3, 6):
            low, high = self.low.copy(), self.high.copy()
            low[index] = high[index] = np.nan
            original = (low**2, high**2)
            repaired = _repair_joint_monotonic_datcon_tail(*original, self.r)
            for raw, fixed in zip(original, repaired):
                np.testing.assert_array_equal(fixed, raw)

    def test_native_radius_scaling_and_contiguous_header_offset(self):
        for nr in (201, 401):
            with self.subTest(nr=nr), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "N10"
                directory.mkdir()
                i1 = int(0.85 * (nr - 1)) + 1
                values = np.column_stack((self.low**2, self.high**2))
                values[-1] = 1000.0
                path = directory / "datcon10"
                with path.open("w") as handle:
                    handle.write(f"{i1} {i1 + len(values) - 1}\n")
                    np.savetxt(handle, values)
                original_bytes = path.read_bytes()
                low, high, first, last = load_datcon_for_mode(
                    str(directory / "egn10w.test"), nr
                )
                self.assertEqual((first, last), (i1, i1 + len(values) - 1))
                for fixed, raw in ((low, self.low), (high, self.high)):
                    np.testing.assert_allclose(fixed[i1 + 4 : i1 + 7], raw[4] ** 2)
                    self.assertTrue(np.isnan(fixed[: i1 - 1]).all())
                    self.assertTrue(np.isnan(fixed[i1 + 7 :]).all())
                self.assertEqual(path.read_bytes(), original_bytes)

        # The same frequency steps on a coarse grid are not steep enough.
        original = self.low**2, self.high**2
        coarse = np.arange(len(self.low)) / 8
        repaired = _repair_joint_monotonic_datcon_tail(*original, coarse)
        for raw, fixed in zip(original, repaired):
            np.testing.assert_array_equal(fixed, raw)

    def test_invalid_datcon_radial_bounds_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "N1"
            directory.mkdir()
            (directory / "datcon1").write_text("0 2\n1 2\n1 2\n1 2\n")
            with self.assertRaisesRegex(ValueError, "Bad datcon radial bounds"):
                load_datcon_for_mode(str(directory / "egn01w.test"), 201)


if __name__ == "__main__":
    unittest.main()
