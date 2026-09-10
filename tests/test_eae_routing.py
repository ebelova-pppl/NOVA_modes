"""Regression checks for the shared v6 upper-gap energy routing policy."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from make_tae_like_list import classify_gap_region as preprocess_route, preprocess_shot
from sort_shot_mixed import classify_gap_region as mixed_route, parse_args
from split_tae_eae import classify_gap_region as split_route
from tae_eae_features import classify_gap_region
from tae_rule_config import load_rule_run_configuration, PRODUCTION_RULE_CONFIG_NAME
from tae_rule_io import sha256_file


class EaeRoutingTests(unittest.TestCase):
    def test_shared_entrypoints_and_strict_boundaries(self):
        self.assertIs(preprocess_route, classify_gap_region)
        self.assertIs(mixed_route, classify_gap_region)
        for fraction, delta, expected in [
            (0.1, 0.9, "eae_like"),
            (np.nextafter(0.2, 0), 1.0, "eae_like"),
            (0.2, 1.0, "mixed"),
            (np.nextafter(0.2, 1), 1.0, "mixed"),
            (0.2, -0.2, "eae_like"),
            (0.3, -0.1, "mixed"),
            (0.3, np.nextafter(-0.1, -1), "eae_like"),
            (0.4, -0.2, "mixed"),
            (0.5, 1.0, "mixed"),
            (np.nextafter(0.5, 1), 1.0, "tae_like"),
        ]:
            with self.subTest(fraction=fraction, delta=delta):
                self.assertEqual(classify_gap_region(delta, fraction), expected)
                legacy_region, group = split_route(
                    delta,
                    fraction,
                    signed_delta_threshold=-0.1,
                    fraction_threshold=0.5,
                    eae_fraction_threshold=0.4,
                )
                self.assertEqual(
                    legacy_region,
                    {
                        "eae_like": "above_upper2",
                        "tae_like": "below_upper2",
                        "mixed": "mixed",
                    }[expected],
                )
                self.assertEqual(group, "above" if expected == "eae_like" else "below")

    def test_frozen_v5_preserves_old_routing_and_gates(self):
        v5_path = REPO / "configs/rules/tae_rules_production_v5.yaml"
        self.assertEqual(
            sha256_file(v5_path),
            "982cc0ba3f17aae03a9fc6a4b662104200df0ff2897bda4de21131ce71c5bc9f",
        )
        old = load_rule_run_configuration(v5_path)
        new = load_rule_run_configuration(PRODUCTION_RULE_CONFIG_NAME)
        self.assertEqual(old.run_kwargs["fraction_direct_eae_threshold"], 0.0)
        self.assertEqual(new.run_kwargs["fraction_direct_eae_threshold"], 0.2)
        old_kwargs, new_kwargs = dict(old.run_kwargs), dict(new.run_kwargs)
        old_kwargs.pop("fraction_direct_eae_threshold")
        new_kwargs.pop("fraction_direct_eae_threshold")
        for key in list(old_kwargs):
            if key.startswith("cross_window_exception_"):
                old_kwargs.pop(key)
                new_kwargs.pop(key)
        self.assertEqual(old_kwargs.pop("interior_envelope_ext_df_gap_min"), 0.0)
        self.assertEqual(new_kwargs.pop("interior_envelope_ext_df_gap_min"), 0.001)
        self.assertTrue(old_kwargs.pop("interior_envelope_ext_df_gap_min_inclusive"))
        self.assertFalse(new_kwargs.pop("interior_envelope_ext_df_gap_min_inclusive"))
        for key in ("axis_energy_amplitude_min", "axis_energy_fraction_min"):
            self.assertIsNone(old_kwargs.pop(key))
            self.assertEqual(new_kwargs.pop(key), 0.5)
        self.assertIsNone(old_kwargs.pop("continuum_noise_top2_min"))
        self.assertEqual(new_kwargs.pop("continuum_noise_top2_min"), 0.01)
        self.assertEqual(old_kwargs, new_kwargs)
        self.assertEqual(
            classify_gap_region(0.1, 0.1, fraction_direct_eae_threshold=0), "mixed"
        )
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed-v5.json"
            document = json.loads(v5_path.read_text())
            document["deduplication"]["rel_freq_tol"] = 0.03
            changed.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "frozen configuration"):
                load_rule_run_configuration(changed)

    def test_invalid_thresholds_and_config_owned_cli_override_fail(self):
        for threshold in [-0.1, 0.41, np.nan, np.inf]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                classify_gap_region(0.1, 0.1, fraction_direct_eae_threshold=threshold)
        with tempfile.TemporaryDirectory() as tmp:
            document = json.loads(
                (REPO / "configs/rules/tae_rules_production_v6.yaml").read_text()
            )
            document["routing"].pop("fraction_direct_eae_threshold")
            path = Path(tmp) / "missing.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(
                ValueError, "missing fraction_direct_eae_threshold"
            ):
                load_rule_run_configuration(path)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                [
                    "--shot_dir",
                    "/tmp/shot",
                    "--out_dir",
                    "/tmp/out",
                    "--fraction_direct_eae_threshold",
                    "0",
                ]
            )
        args = parse_args(
            [
                "--shot_dir",
                "/tmp/shot",
                "--out_dir",
                "/tmp/out",
                "--method",
                "rf-cnn",
                "--rf_model",
                "rf",
                "--cnn_model",
                "cnn",
            ]
        )
        self.assertEqual(args.fraction_direct_eae_threshold, 0.2)

    def test_native_preprocessing_routes_small_gap_fraction_before_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "synthetic"
            n_dir = shot / "N1"
            n_dir.mkdir(parents=True)
            mode = np.ones((4, 201))
            np.concatenate(
                ([1.0], np.stack([mode, mode, mode]).ravel(), [201.0, 0.0, 1.0])
            ).tofile(n_dir / "egn01w.test")
            high = np.full(201, 0.9**2)
            high[-16:] = 2.0**2
            with (n_dir / "datcon1").open("w") as stream:
                stream.write("1 201\n")
                np.savetxt(stream, np.column_stack([np.full(201, 0.5**2), high]))
            old = preprocess_shot(shot, fraction_direct_eae_threshold=0)
            new = preprocess_shot(shot)
            self.assertEqual(old.rows[0]["gap_region"], "mixed")
            self.assertEqual(new.rows[0]["gap_region"], "eae_like")
            self.assertEqual(len(new.eae_rows), 1)
            self.assertEqual(len(new.rule_feature_data), 0)
            for field in ("signed_delta", "fraction_below_upper2", "input_fingerprint"):
                self.assertEqual(old.rows[0][field], new.rows[0][field])


if __name__ == "__main__":
    unittest.main()
