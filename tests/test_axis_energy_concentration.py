"""Joint amplitude/energy rejection, strict boundaries and frozen presets."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_rule_sorting import write_mode, write_datcon
from sort_shot_mixed import parse_args as parse_mixed_args, run_rules_method
from sort_shot_rules import parse_args, run_configured_shot
from tae_rule_config import load_rule_run_configuration
from tae_rule_engine import (
    AxisEnergyConcentrationConfig,
    BAD_AXIS_ENERGY_CONCENTRATION,
    BAD_AXIS_SPIKE,
    evaluate_mode,
    extract_axis_energy_concentration_features,
)


def long_axis_shoulder(nr=201):
    r = np.linspace(0, 1, nr)
    mode = np.zeros((4, nr))
    mode[1] = np.where(r <= 0.01, r / 0.01, np.exp(-(r - 0.01) / 0.075))
    return mode


class AxisEnergyTests(unittest.TestCase):
    def test_strict_amplitude_and_fraction_boundaries(self):
        for amplitude, expected in ((0.5, False), (np.nextafter(0.5, 1), True)):
            mode = np.zeros((2, 201))
            mode[0, :12] = -amplitude
            f = extract_axis_energy_concentration_features(mode)
            self.assertEqual(f["candidate_found"], expected)
            self.assertEqual(f["axis_signed_amplitude"], -amplitude)
            self.assertGreater(f["inner_energy_fraction"], 0.5)
        # Constant W has exactly half its energy inside r=0.5.
        mode = np.ones((2, 201))
        for radius, expected in ((0.5, False), (np.nextafter(0.5, 1), True)):
            f = extract_axis_energy_concentration_features(
                mode, config=AxisEnergyConcentrationConfig(energy_r_max=radius)
            )
            self.assertAlmostEqual(f["inner_energy_fraction"], radius, places=15)
            self.assertEqual(f["candidate_found"], expected)

    def test_radius_boundary_and_all_harmonic_energy(self):
        mode = np.zeros((2, 201))
        mode[0, 3] = -1
        f = extract_axis_energy_concentration_features(mode)
        self.assertTrue(f["candidate_found"])
        self.assertEqual(f["axis_peak_r"], 0.015)
        excluded = extract_axis_energy_concentration_features(
            mode,
            config=AxisEnergyConcentrationConfig(
                amplitude_r_max=np.nextafter(0.015, 0)
            ),
        )
        self.assertFalse(excluded["candidate_found"])
        self.assertEqual(excluded["axis_amplitude"], 0)
        # Another harmonic's broad body must count in the denominator.
        mode[1, 20:100] = 0.8
        extended = extract_axis_energy_concentration_features(mode)
        self.assertEqual(extended["axis_amplitude"], 1)
        self.assertLess(extended["inner_energy_fraction"], 0.1)
        self.assertFalse(extended["candidate_found"])

    def test_native_resolution_and_between_sample_energy_endpoint(self):
        for nr in (51, 101, 201, 401):
            with self.subTest(nr=nr):
                mode = np.ones((3, nr))
                f = extract_axis_energy_concentration_features(
                    mode, config=AxisEnergyConcentrationConfig(energy_r_max=0.753)
                )
                self.assertAlmostEqual(f["inner_energy_fraction"], 0.753, places=14)
                self.assertEqual(f["n_radial"], nr)
                self.assertTrue(f["candidate_found"])

    def test_disabled_gate_retains_measurements_and_zero_energy_is_undefined(self):
        mode = long_axis_shoulder()
        enabled = extract_axis_energy_concentration_features(mode)
        disabled = extract_axis_energy_concentration_features(
            mode,
            config=AxisEnergyConcentrationConfig(
                amplitude_min=None, energy_fraction_min=None
            ),
        )
        self.assertTrue(enabled["candidate_found"])
        self.assertFalse(disabled["candidate_found"])
        for key in (
            "axis_amplitude",
            "axis_peak_r",
            "inner_energy_fraction",
            "total_energy",
        ):
            self.assertEqual(enabled[key], disabled[key])
        zero = extract_axis_energy_concentration_features(np.zeros((2, 201)))
        self.assertIsNone(zero["inner_energy_fraction"])
        self.assertFalse(zero["candidate_found"])
        for kwargs in (
            dict(amplitude_r_max=0),
            dict(energy_r_max=float("nan")),
            dict(amplitude_min=-0.1),
            dict(energy_fraction_min=1.1),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AxisEnergyConcentrationConfig(**kwargs)
        with self.assertRaises(ValueError):
            extract_axis_energy_concentration_features(np.full((2, 201), np.nan))

    def test_existing_bad_reason_precedes_new_gate(self):
        mode = np.zeros((4, 201))
        mode[0, 1] = 1
        result = evaluate_mode(
            dict(
                path="shot/N1/egn01w.test",
                mode_key="shot/N1/egn01w.test",
                shot="shot",
                ntor=1,
                omega=1,
                gamma_d=0,
                gap_region="tae_like",
                input_fingerprint="a" * 64,
            ),
            mode=mode,
            low2=np.full(201, 0.25),
            high2=np.full(201, 2.25),
        )
        self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)
        self.assertTrue(
            result.features["boundary_features"]["axis_energy_concentration"][
                "candidate_found"
            ]
        )

    def test_current_canonical_rejects_long_shoulder_v8_preserves_survivor(self):
        for version in (5, 6, 7, 8):
            old = load_rule_run_configuration(f"tae_rules_production_v{version}")
            self.assertIsNone(old.run_kwargs["axis_energy_amplitude_min"])
            self.assertIsNone(old.run_kwargs["axis_energy_fraction_min"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "test_shot"
            write_mode(
                shot / "N1/egn01w.test",
                omega=1,
                ntor=1,
                nr=201,
                mode=long_axis_shoulder(),
            )
            write_datcon(shot / "N1/datcon1", nr=201, upper_frequency=1.5)
            old = run_configured_shot(
                shot, root / "old", rule_config="tae_rules_production_v8"
            )
            self.assertEqual(old.final_rows[0]["rule_decision"], "REVIEW")
            with contextlib.redirect_stdout(io.StringIO()):
                new = run_rules_method(
                    parse_mixed_args(
                        ["--shot_dir", str(shot), "--out_dir", str(root / "new")]
                    )
                )
            self.assertEqual(new.final_rows[0]["final_decision"], "BAD")
            self.assertEqual(
                new.final_rows[0]["rule_primary_reason"], BAD_AXIS_ENERGY_CONCENTRATION
            )
            f = json.loads(new.final_rows[0]["rule_features"])
            self.assertGreater(
                f["boundary_features"]["axis_artifact"]["axis_halfmax_width_grid"], 10
            )
            prior = json.loads(old.final_rows[0]["rule_features"])
            f["boundary_features"].pop("axis_energy_concentration")
            prior["boundary_features"].pop("axis_energy_concentration")
            f["numerical_structure_features"].pop("extended_continuum_noise")
            prior["numerical_structure_features"].pop("extended_continuum_noise")
            f.pop("severity_features")
            prior.pop("severity_features")
            self.assertEqual(f, prior)
            for s in (new.summary,):
                self.assertTrue(s["axis_energy_concentration_gate_enabled"])
                self.assertEqual(s["axis_energy_amplitude_r_max"], 0.015)
                self.assertEqual(s["axis_energy_amplitude_min"], 0.5)
                self.assertEqual(s["axis_energy_r_max"], 0.05)
                self.assertEqual(s["axis_energy_fraction_min"], 0.5)
            with (root / "new/shot_summary_by_n.csv").open() as f:
                per_n = next(csv.DictReader(f))
            for name in (
                "axis_energy_concentration_gate_enabled",
                "axis_energy_amplitude_r_max",
                "axis_energy_amplitude_min",
                "axis_energy_r_max",
                "axis_energy_fraction_min",
            ):
                self.assertEqual(per_n[name], str(new.summary[name]))

    def test_cli_defaults_and_named_config_override_protection(self):
        flags = ["--shot_dir", "shot", "--out_dir", "out"]
        args = parse_args(flags)
        self.assertEqual(args.axis_energy_amplitude_min, 0.5)
        self.assertFalse(args.disable_axis_energy_concentration)
        for flag in ("--disable_axis_energy_concentration", "--axis_energy_r_max"):
            extra = [flag] + (["0.1"] if flag == "--axis_energy_r_max" else [])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                parse_args(flags + ["--rule_config", "tae_rules_production_v9"] + extra)


if __name__ == "__main__":
    unittest.main()
