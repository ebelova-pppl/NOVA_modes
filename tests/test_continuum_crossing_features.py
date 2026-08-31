import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cont_features import (  # noqa: E402
    CROSSING_FEATURE_DEFAULTS,
    EXTREMUM_FEATURE_DEFAULTS,
    continuum_crossing_features,
    continuum_crossing_records,
    continuum_extremum_features,
    continuum_scalars,
    load_datcon_for_mode,
)
from mode_features import (  # noqa: E402
    compute_features_for_mode,
    get_feature_names,
    get_feature_schema_version,
)
from rf_train_classify import (  # noqa: E402
    attach_feature_metadata,
    validate_model_feature_schema,
)
from cnn_infer_common import (  # noqa: E402
    ContinuumBranchCNN,
    build_continuum_branch_array,
    build_continuum_channel_array,
    build_raw_preprocess_metadata,
    resolve_raw_preprocess_metadata,
)


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


class ContinuumExtremumFeatureTests(unittest.TestCase):
    def setUp(self):
        self.r = np.linspace(0.0, 1.0, 201)

    def test_upper_minimum_uses_W_peak_and_gap_side_sign(self):
        mode = np.exp(-((self.r - 0.20) / 0.025) ** 2)[None, :]
        lower = np.full_like(self.r, 0.5)
        upper = 1.2 + 4.0 * (self.r - 0.20) ** 2

        features = continuum_extremum_features(
            mode, 1.19, lower**2, upper**2, r=self.r
        )

        self.assertAlmostEqual(features["ext_dr"], 0.0)
        self.assertAlmostEqual(features["ext_df_gap"], (1.2 - 1.19) / 1.19)
        self.assertGreater(features["ext_energy_frac"], 0.95)

    def test_lower_maximum_has_same_positive_gap_side_sign(self):
        mode = np.exp(-((self.r - 0.25) / 0.025) ** 2)[None, :]
        lower = 0.8 - 2.0 * (self.r - 0.25) ** 2
        upper = np.full_like(self.r, 1.4)

        features = continuum_extremum_features(
            mode, 0.81, lower**2, upper**2, r=self.r
        )

        self.assertAlmostEqual(features["ext_dr"], 0.0)
        self.assertAlmostEqual(features["ext_df_gap"], (0.81 - 0.8) / 0.81)
        self.assertGreater(features["ext_energy_frac"], 0.95)

    def test_energy_fraction_is_small_for_mode_away_from_extremum(self):
        mode = np.exp(-((self.r - 0.45) / 0.025) ** 2)[None, :]
        lower = np.full_like(self.r, 0.5)
        upper = 1.2 + 4.0 * (self.r - 0.20) ** 2

        features = continuum_extremum_features(
            mode, 1.19, lower**2, upper**2, r=self.r
        )

        self.assertAlmostEqual(features["ext_dr"], 0.25)
        self.assertLess(features["ext_energy_frac"], 1e-6)

    def test_energy_fraction_uses_total_mode_energy(self):
        centered = np.exp(-((self.r - 0.20) / 0.02) ** 2)
        remote = np.exp(-((self.r - 0.60) / 0.02) ** 2)
        mode = np.stack([centered, remote])
        lower = np.full_like(self.r, 0.5)
        upper = 1.2 + 4.0 * (self.r - 0.20) ** 2

        features = continuum_extremum_features(
            mode, 1.19, lower**2, upper**2, r=self.r
        )

        self.assertGreater(features["ext_energy_frac"], 0.45)
        self.assertLess(features["ext_energy_frac"], 0.51)

    def test_zero_mode_returns_safe_defaults(self):
        lower = 0.8 - 2.0 * (self.r - 0.25) ** 2
        upper = np.full_like(self.r, 1.4)
        self.assertEqual(
            continuum_extremum_features(
                np.zeros((2, self.r.size)), 0.81, lower**2, upper**2, r=self.r
            ),
            EXTREMUM_FEATURE_DEFAULTS,
        )

    def test_no_inner_extremum_returns_safe_defaults(self):
        mode = np.ones((2, self.r.size))
        lower2 = np.full_like(self.r, 0.5**2)
        upper2 = np.full_like(self.r, 1.5**2)

        self.assertEqual(
            continuum_extremum_features(mode, 1.0, lower2, upper2, r=self.r),
            EXTREMUM_FEATURE_DEFAULTS,
        )

    def test_rule_specific_search_extends_past_point_four_without_changing_legacy(self):
        mode = np.exp(-((self.r - 0.45) / 0.025) ** 2)[None, :]
        lower = np.full_like(self.r, 0.5)
        upper = 1.2 + 4.0 * (self.r - 0.45) ** 2

        legacy, legacy_match = continuum_extremum_features(
            mode,
            1.19,
            lower**2,
            upper**2,
            r=self.r,
            return_match_status=True,
        )
        extended, extended_match = continuum_extremum_features(
            mode,
            1.19,
            lower**2,
            upper**2,
            r=self.r,
            r_min=0.03,
            r_max=0.5,
            return_match_status=True,
            filter_candidates_after_detection=True,
        )

        self.assertFalse(legacy_match)
        self.assertEqual(legacy, EXTREMUM_FEATURE_DEFAULTS)
        self.assertTrue(extended_match)
        self.assertAlmostEqual(extended["ext_dr"], 0.0)
        self.assertAlmostEqual(extended["ext_df_gap"], (1.2 - 1.19) / 1.19)

    def test_rule_specific_extremum_search_includes_point_five_boundary_only(self):
        lower = np.full_like(self.r, 0.5)
        at_boundary_mode = np.exp(-((self.r - 0.5) / 0.025) ** 2)[None, :]
        at_boundary_upper = 1.2 + 4.0 * (self.r - 0.5) ** 2
        boundary, boundary_match = continuum_extremum_features(
            at_boundary_mode,
            1.19,
            lower**2,
            at_boundary_upper**2,
            r=self.r,
            r_min=0.03,
            r_max=0.5,
            return_match_status=True,
            filter_candidates_after_detection=True,
        )

        outside_mode = np.exp(-((self.r - 0.55) / 0.025) ** 2)[None, :]
        outside_upper = 1.2 + 4.0 * (self.r - 0.55) ** 2
        outside, outside_match = continuum_extremum_features(
            outside_mode,
            1.19,
            lower**2,
            outside_upper**2,
            r=self.r,
            r_min=0.03,
            r_max=0.5,
            return_match_status=True,
            filter_candidates_after_detection=True,
        )

        self.assertTrue(boundary_match)
        self.assertAlmostEqual(boundary["ext_dr"], 0.0)
        self.assertFalse(outside_match)
        self.assertEqual(outside, EXTREMUM_FEATURE_DEFAULTS)


class ContinuumScalarTests(unittest.TestCase):
    def test_datcon_loader_repairs_joint_outer_tail_spike(self):
        lower = np.array([1.00, 1.04, 1.02, 1.06, 1.03, 14.0, 18.0, 22.0])
        upper = np.array([2.00, 2.05, 2.02, 2.08, 2.04, 20.0, 25.0, 30.0])
        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            lines = ["1 8"]
            lines.extend(f"{low**2} {high**2}" for low, high in zip(lower, upper))
            (n_dir / "datcon3").write_text("\n".join(lines) + "\n")

            low2, high2, *_ = load_datcon_for_mode(str(n_dir / "egn03w.test"), n_r=8)

        expected_lower = np.mean(lower[1:5])
        expected_upper = np.mean(upper[1:5])
        np.testing.assert_allclose(np.sqrt(low2[:5]), lower[:5])
        np.testing.assert_allclose(np.sqrt(high2[:5]), upper[:5])
        np.testing.assert_allclose(np.sqrt(low2[5:]), expected_lower)
        np.testing.assert_allclose(np.sqrt(high2[5:]), expected_upper)

    def test_datcon_loader_leaves_normal_outer_tail_unchanged(self):
        lower = np.array([1.00, 1.04, 1.08, 1.12, 1.18, 1.24, 1.31, 1.39])
        upper = np.array([2.00, 2.05, 2.11, 2.18, 2.26, 2.35, 2.45, 2.56])
        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            lines = ["1 8"]
            lines.extend(f"{low**2} {high**2}" for low, high in zip(lower, upper))
            (n_dir / "datcon3").write_text("\n".join(lines) + "\n")

            low2, high2, *_ = load_datcon_for_mode(str(n_dir / "egn03w.test"), n_r=8)

        np.testing.assert_allclose(np.sqrt(low2), lower)
        np.testing.assert_allclose(np.sqrt(high2), upper)

    def test_cnn_continuum_channels_signs_and_broadcast(self):
        lower = np.array([0.8, 0.8, 0.8])
        upper = np.array([1.2, 1.2, 1.2])
        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            lines = ["1 3"]
            lines.extend(f"{low**2} {high**2}" for low, high in zip(lower, upper))
            (n_dir / "datcon3").write_text("\n".join(lines) + "\n")
            path = str(n_dir / "egn03w.test")

            inside = build_continuum_channel_array(
                path, 1.0, n_r=3, M_target=2, R_target=3
            )
            above = build_continuum_channel_array(
                path, 1.3, n_r=3, M_target=2, R_target=3
            )
            below = build_continuum_channel_array(
                path, 0.7, n_r=3, M_target=2, R_target=3
            )

        self.assertEqual(inside.shape, (2, 2, 3))
        np.testing.assert_allclose(inside[0], 0.2, rtol=1e-6)
        np.testing.assert_allclose(inside[1], 0.2, rtol=1e-6)
        self.assertTrue(np.all(above[0] < 0.0))
        self.assertTrue(np.all(above[1] > 0.0))
        self.assertTrue(np.all(below[0] > 0.0))
        self.assertTrue(np.all(below[1] < 0.0))

    def test_cnn_continuum_channels_clip_and_missing_fallback(self):
        lower = np.array([0.1, 0.1, 0.1])
        upper = np.array([10.0, 10.0, 10.0])
        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            lines = ["1 3"]
            lines.extend(f"{low**2} {high**2}" for low, high in zip(lower, upper))
            (n_dir / "datcon3").write_text("\n".join(lines) + "\n")

            clipped = build_continuum_channel_array(
                str(n_dir / "egn03w.test"),
                1.0,
                n_r=3,
                M_target=2,
                R_target=3,
                clip=2.0,
            )
            missing = build_continuum_channel_array(
                str(n_dir / "missing" / "egn03w.test"),
                1.0,
                n_r=3,
                M_target=2,
                R_target=3,
            )

        self.assertLessEqual(float(np.max(clipped)), 2.0)
        self.assertGreaterEqual(float(np.min(clipped)), -2.0)
        np.testing.assert_allclose(missing, 0.0)

    def test_cnn_continuum_branch_contains_W_du_dl_and_mask(self):
        lower = np.array([0.8, 0.8, 0.8])
        upper = np.array([1.2, 1.2, 1.2])
        mode = np.array(
            [
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            lines = ["2 4"]
            lines.extend(f"{low**2} {high**2}" for low, high in zip(lower, upper))
            (n_dir / "datcon3").write_text("\n".join(lines) + "\n")
            branch = build_continuum_branch_array(
                str(n_dir / "egn03w.test"),
                mode,
                1.0,
                R_target=5,
            )

        self.assertEqual(branch.shape, (4, 5))
        np.testing.assert_allclose(branch[0], [0.0, 1.0, 0.8, 0.2, 0.0])
        np.testing.assert_allclose(branch[1, 1:4], 0.2, rtol=1e-6)
        np.testing.assert_allclose(branch[2, 1:4], 0.2, rtol=1e-6)
        np.testing.assert_allclose(branch[1:3, [0, 4]], 0.0)
        np.testing.assert_array_equal(branch[3], [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_cnn_continuum_branch_missing_datcon_keeps_W(self):
        mode = np.array([[0.0, 1.0, 2.0, 1.0, 0.0]])
        branch = build_continuum_branch_array(
            "/missing/N3/egn03w.test",
            mode,
            1.0,
            R_target=5,
        )

        np.testing.assert_allclose(branch[0], [0.0, 0.25, 1.0, 0.25, 0.0])
        np.testing.assert_allclose(branch[1:], 0.0)

    def test_cnn_continuum_branch_zero_input_control(self):
        branch = build_continuum_branch_array(
            "/missing/N3/egn03w.test",
            np.array([[1.0]]),
            1.0,
            R_target=7,
            zero_inputs=True,
        )
        metadata = build_raw_preprocess_metadata(
            R_target=7,
            M_target=100,
            continuum_branch=True,
            continuum_branch_zero_inputs=True,
        )
        resolved = resolve_raw_preprocess_metadata({"preprocess": metadata})

        self.assertEqual(branch.shape, (4, 7))
        self.assertEqual(branch.dtype, np.float32)
        np.testing.assert_array_equal(branch, 0.0)
        self.assertTrue(resolved["continuum_branch_zero_inputs"])

    def test_cnn_continuum_branch_model_output_shape(self):
        model = ContinuumBranchCNN()
        x_img = torch.zeros((2, 1, 100, 201), dtype=torch.float32)
        x_continuum = torch.zeros((2, 4, 201), dtype=torch.float32)

        logits = model(x_img, x_continuum)

        self.assertEqual(tuple(logits.shape), (2,))

    def test_energy_tie_selects_largest_radius_with_maximum_W(self):
        r = np.linspace(0.0, 1.0, 5)
        mode = np.sqrt(np.array([[0.0, 1.0, 4.0, 4.0, 0.0]]))
        low2 = np.zeros(5)
        high2 = np.full(5, 2.0)

        legacy = continuum_scalars(mode, 1.0, low2, high2, r=r)
        energy_tie = continuum_scalars(
            mode,
            1.0,
            low2,
            high2,
            r=r,
            r_star_energy_tie=True,
        )

        self.assertAlmostEqual(legacy["r_star"], 0.0)
        self.assertAlmostEqual(energy_tie["r_star"], 0.75)
        self.assertEqual(legacy["delta2_eff"], energy_tie["delta2_eff"])
        self.assertGreater(energy_tie["W_star"], legacy["W_star"])


class RFFeatureSchemaTests(unittest.TestCase):
    def setUp(self):
        self.mode = np.arange(15, dtype=float).reshape(3, 5) / 14.0
        self.extra = {"omega": 1.0, "gamma_d": 0.01, "ntor": 3}

    def test_energy_tie_has_distinct_schema_version(self):
        legacy = get_feature_schema_version()
        energy_tie = get_feature_schema_version(r_star_energy_tie=True)
        combined = get_feature_schema_version(
            include_extremum_features=True,
            r_star_energy_tie=True,
        )

        self.assertNotEqual(legacy, energy_tie)
        self.assertTrue(energy_tie.endswith("_rstar_energy_tie_v1"))
        self.assertTrue(combined.endswith("_rstar_energy_tie_v1"))

    def test_energy_tie_checkpoint_metadata_is_validated(self):
        clf = type("DummyRF", (), {"n_features_in_": 22})()
        names = get_feature_names()
        attach_feature_metadata(
            clf,
            names,
            r_star_energy_tie=True,
        )

        validate_model_feature_schema(
            clf,
            names,
            r_star_energy_tie=True,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_model_feature_schema(clf, names)

    def test_feature_builder_propagates_energy_tie_option(self):
        low2 = np.zeros(5)
        high2 = np.full(5, 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            n_dir = Path(tmp) / "N3"
            n_dir.mkdir()
            lines = ["1 5"]
            lines.extend(f"{low} {high}" for low, high in zip(low2, high2))
            (n_dir / "datcon3").write_text("\n".join(lines) + "\n")
            extra = dict(self.extra, path=str(n_dir / "egn03w.test"))

            legacy = compute_features_for_mode(self.mode, extra)
            energy_tie = compute_features_for_mode(
                self.mode,
                extra,
                r_star_energy_tie=True,
            )

        r_star_index = get_feature_names().index("r_star")
        self.assertAlmostEqual(legacy[r_star_index], 0.0)
        self.assertAlmostEqual(energy_tie[r_star_index], 1.0)

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

    def test_extremum_schema_lengths_and_fallback_order(self):
        extremum = compute_features_for_mode(
            self.mode,
            self.extra,
            include_extremum_features=True,
        )
        combined = compute_features_for_mode(
            self.mode,
            self.extra,
            include_crossing_features=True,
            include_extremum_features=True,
        )

        extremum_names = get_feature_names(include_extremum_features=True)
        combined_names = get_feature_names(
            include_crossing_features=True,
            include_extremum_features=True,
        )
        self.assertEqual(len(extremum_names), 25)
        self.assertEqual(len(combined_names), 31)
        self.assertEqual(
            extremum_names[-3:],
            ["ext_dr", "ext_df_gap", "ext_energy_frac"],
        )
        self.assertEqual(extremum.size, 25)
        self.assertEqual(combined.size, 31)
        np.testing.assert_allclose(extremum[-3:], [1.0, 1.0, 0.0])
        np.testing.assert_allclose(combined[-3:], [1.0, 1.0, 0.0])

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
