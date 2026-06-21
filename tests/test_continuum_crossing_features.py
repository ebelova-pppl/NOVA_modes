import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cont_features import (  # noqa: E402
    CROSSING_FEATURE_DEFAULTS,
    continuum_crossing_features,
    continuum_crossing_records,
)
from mode_features import compute_features_for_mode, get_feature_names  # noqa: E402


class ContinuumCrossingFeatureTests(unittest.TestCase):
    def test_known_lower_and_upper_crossings(self):
        r = np.linspace(0.0, 1.0, 5)
        mode = np.sqrt(np.array([[0.0, 1.0, 4.0, 1.0, 0.0]]))
        low2 = np.array([0.0, 0.5, 1.5, 1.5, 1.5])
        high2 = np.array([2.0, 2.0, 2.0, 1.0, 0.0])

        records = continuum_crossing_records(
            mode, 1.0, low2, high2, r=r, r_shear0=0.2
        )
        self.assertEqual([record["boundary"] for record in records], ["low", "high"])
        self.assertAlmostEqual(records[0]["r_cross"], 0.375)
        self.assertAlmostEqual(records[1]["r_cross"], 0.75)

        features = continuum_crossing_features(
            mode, 1.0, low2, high2, r=r, r_shear0=0.2
        )
        self.assertEqual(features["n_cross"], 2)
        self.assertAlmostEqual(features["r_star_max"], 0.375)
        self.assertAlmostEqual(features["W_star_max"], 0.625)
        self.assertAlmostEqual(features["W_star_sum"], 0.875)
        self.assertAlmostEqual(features["r_star_high_shear"], 0.75)
        self.assertAlmostEqual(features["W_star_high_shear"], 0.25 * 0.55**2)

    def test_multiple_crossings_are_counted_and_summed(self):
        r = np.linspace(0.0, 1.0, 5)
        mode = np.ones((2, 5))
        low2 = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
        high2 = np.full(5, 3.0)

        features = continuum_crossing_features(mode, 1.0, low2, high2, r=r)
        self.assertEqual(features["n_cross"], 4)
        self.assertAlmostEqual(features["W_star_sum"], 4.0)
        self.assertAlmostEqual(features["r_star_max"], 0.875)

    def test_exact_grid_crossing_is_not_double_counted(self):
        r = np.array([0.0, 0.5, 1.0])
        mode = np.ones((1, 3))
        low2 = np.array([0.0, 1.0, 2.0])
        high2 = np.full(3, 3.0)

        records = continuum_crossing_records(mode, 1.0, low2, high2, r=r)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["r_cross"], 0.5)

    def test_consecutive_zero_run_uses_midpoint(self):
        r = np.array([0.0, 0.25, 0.75, 1.0])
        mode = np.ones((1, 4))
        low2 = np.array([0.0, 1.0, 1.0, 2.0])
        high2 = np.full(4, 3.0)

        records = continuum_crossing_records(mode, 1.0, low2, high2, r=r)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["r_cross"], 0.5)

    def test_nan_gap_is_not_bridged(self):
        r = np.array([0.0, 0.5, 1.0])
        mode = np.ones((1, 3))
        low2 = np.array([0.0, np.nan, 2.0])
        high2 = np.full(3, 3.0)

        self.assertEqual(
            continuum_crossing_records(mode, 1.0, low2, high2, r=r),
            [],
        )

    def test_no_crossing_and_zero_mode_return_safe_defaults(self):
        mode = np.zeros((2, 5))
        low2 = np.zeros(5)
        high2 = np.full(5, 2.0)

        features = continuum_crossing_features(mode, 1.0, low2, high2)
        self.assertEqual(features, CROSSING_FEATURE_DEFAULTS)

    def test_equal_amplitude_tie_uses_largest_radius(self):
        mode = np.ones((1, 3))
        low2 = np.array([0.0, 2.0, 0.0])
        high2 = np.full(3, 3.0)

        features = continuum_crossing_features(mode, 1.0, low2, high2)
        self.assertEqual(features["n_cross"], 2)
        self.assertAlmostEqual(features["r_star_max"], 0.75)

    def test_malformed_shapes_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "radial dimension"):
            continuum_crossing_features(
                np.ones((2, 5)),
                1.0,
                np.ones(4),
                np.ones(5),
            )


class RFFeatureSchemaTests(unittest.TestCase):
    def setUp(self):
        self.mode = np.arange(15, dtype=float).reshape(3, 5) / 14.0
        self.extra = {"omega": 1.0, "gamma_d": 0.01, "ntor": 3}

    def test_production_and_experimental_schema_lengths(self):
        production = compute_features_for_mode(self.mode, self.extra)
        experimental = compute_features_for_mode(
            self.mode,
            self.extra,
            include_crossing_features=True,
        )

        self.assertEqual(len(get_feature_names(False)), 22)
        self.assertEqual(len(get_feature_names(True)), 28)
        self.assertNotIn("omega", get_feature_names(False))
        self.assertEqual(get_feature_names(False)[-1], "W_star_max")
        self.assertEqual(production.size, 22)
        self.assertEqual(experimental.size, 28)
        np.testing.assert_allclose(experimental[:22], production)
        np.testing.assert_allclose(experimental[22:], 0.0)

    def test_active_checkpoint_accepts_production_vector(self):
        model_path = REPO_ROOT / "models" / "nova_mode_classifier.joblib"
        if not model_path.is_file():
            self.skipTest(f"Active RF checkpoint not found: {model_path}")

        import joblib

        clf = joblib.load(model_path)
        production = compute_features_for_mode(self.mode, self.extra).reshape(1, -1)
        self.assertEqual(getattr(clf, "n_features_in_", None), 22)
        self.assertEqual(
            getattr(clf, "nova_feature_names_", None),
            get_feature_names(False),
        )
        probability = clf.predict_proba(production)
        self.assertEqual(probability.shape, (1, 2))

    def test_experimental_features_follow_documented_order(self):
        r = np.linspace(0.0, 1.0, 5)
        low2 = np.array([0.0, 0.5, 1.5, 1.5, 1.5])
        high2 = np.array([2.0, 2.0, 2.0, 1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            datcon = n_dir / "datcon3"
            lines = ["1 5"]
            lines.extend(f"{low} {high}" for low, high in zip(low2, high2))
            datcon.write_text("\n".join(lines) + "\n")

            extra = dict(self.extra)
            extra["path"] = str(n_dir / "egn03w.test")
            features = compute_features_for_mode(
                self.mode,
                extra,
                include_crossing_features=True,
            )

        expected = continuum_crossing_features(
            self.mode,
            self.extra["omega"],
            low2,
            high2,
            r=r,
        )
        production_names = get_feature_names(False)
        self.assertEqual(
            features[production_names.index("W_star_max")],
            expected["W_star_max"],
        )
        crossing_names = get_feature_names(True)[22:]
        np.testing.assert_allclose(
            features[22:],
            [expected[name] for name in crossing_names],
        )


if __name__ == "__main__":
    unittest.main()
