"""Analytic checks for the continuum-side roughness gate."""

import copy
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from continuum_noise import (
    ContinuumNoiseThresholds, assess_continuum_noise, measure_continuum_noise,
    extract_continuum_noise_features, BAD_EXTENDED_CONTINUUM_NOISE,
)


def outside(mode, **kwargs):
    nr = mode.shape[1]
    return measure_continuum_noise(mode, 2., np.ones(nr), np.full(nr, 2.), **kwargs)


class ContinuumNoiseTests(unittest.TestCase):
    def test_nyquist_normalization_and_smooth_linear_profile(self):
        a = np.tile((-1.) ** np.arange(201), (2, 1))
        f = outside(a)
        region = f["records"][0]
        self.assertAlmostEqual(region["hf_out_local_fraction"], 199 / 200)
        self.assertAlmostEqual(region["hf_out_radial_extent"], 199)
        self.assertAlmostEqual(region["hf_out_harmonic_extent"], 2)
        self.assertAlmostEqual(region["hf_out_top2_ratio"], 199 / 200)
        linear = outside(np.linspace(-1, 1, 201)[None, :])["records"][0]
        self.assertLess(linear["hf_out_local_fraction"], 1e-28)

    def test_isolated_spike_spreads_to_three_centers(self):
        a = np.zeros((1, 201))
        a[0, 100] = -1
        region = outside(a)["records"][0]
        self.assertAlmostEqual(region["hf_out_local_fraction"], 3 / 8)
        self.assertAlmostEqual(region["hf_out_radial_extent"], 2)
        self.assertEqual(region["hf_positive_radial_count"], 3)

    def test_full_energy_ranking_all_harmonic_numerator_and_scale_invariance(self):
        a = np.tile((-1.) ** np.arange(201), (4, 1))
        f = outside(a)
        self.assertEqual(f["top2_harmonic_indices"], [0, 1])
        self.assertGreater(f["records"][0]["hf_out_top2_ratio"], 1)
        self.assertEqual(f, outside(a * -17))
        self.assertEqual(f["records"], outside(np.vstack([a, np.zeros((5, 201))]))["records"])
        a = np.zeros((3, 201))
        a[0, 100] = 1
        a[1] = .2
        a[2] = .3
        self.assertEqual(outside(a)["top2_harmonic_indices"], [2, 1])

    def test_stencil_excludes_in_gap_neighbor_without_masking_signal(self):
        a = np.zeros((1, 201)); a[0, 20] = 1
        high = np.full(201, 5.); high[:20] = 2.
        f = measure_continuum_noise(a, 2., np.ones(201), high)
        region = f["records"][0]
        self.assertEqual(region["hf_out_energy"], 0)
        self.assertIsNone(region["hf_out_local_fraction"])
        self.assertGreater(region["hf_crossing_stencil_energy"], 0)
        # A constant original signal must stay flat despite an outside/inside edge.
        region = measure_continuum_noise(np.ones_like(a), 2., np.ones(201), high)["records"][0]
        self.assertEqual(region["hf_crossing_stencil_energy"], 0)

    def test_separate_regions_unknown_continuum_and_equality(self):
        a = np.ones((1, 201))
        low = np.ones(201); high = np.full(201, 5.)
        high[10:20] = 2.; high[30:40] = 2.; low[50:60] = 4.5
        high[14] = np.nan
        f = measure_continuum_noise(a, 2., low, high)
        self.assertEqual(len(f["records"]), 4)
        self.assertEqual(f["unknown_continuum_sample_count"], 1)
        self.assertEqual(f["records"][-1]["side"], "below_lower")
        self.assertEqual(sum(r["unknown_neighbor_stencil_count"] for r in f["records"]), 2)
        on_boundary = measure_continuum_noise(a, 2., low * 0 + 4., high * 0 + 4.)
        self.assertEqual(on_boundary["records"], [])

    def test_same_region_thresholds_and_inclusive_boundaries(self):
        f = outside(np.tile((-1.) ** np.arange(201), (2, 1)))
        r = f["records"][0]
        cuts = ContinuumNoiseThresholds(r["hf_out_top2_ratio"], r["hf_out_local_fraction"], r["hf_out_radial_length"])
        self.assertTrue(assess_continuum_noise(f, cuts)["candidate_found"])
        for name in ("top2_min", "local_min", "radial_length_min"):
            changed = replace(cuts, **{name: np.nextafter(getattr(cuts, name), np.inf)})
            self.assertFalse(assess_continuum_noise(f, changed)["candidate_found"])
        split = copy.deepcopy(f)
        split["records"] = [dict(r, hf_out_top2_ratio=.001), dict(r, region_id=1, hf_out_radial_length=.005)]
        self.assertFalse(assess_continuum_noise(split, ContinuumNoiseThresholds(.01, .1, .02))["candidate_found"])

    def test_disabled_zero_energy_and_other_resolutions(self):
        for nr in (51, 101, 201, 401):
            f = outside(np.tile((-1.) ** np.arange(nr), (2, 1)))
            self.assertFalse(assess_continuum_noise(f)["candidate_found"])
            self.assertTrue(assess_continuum_noise(f, ContinuumNoiseThresholds(.1, .1, .02))["candidate_found"])
            self.assertAlmostEqual(f["records"][0]["hf_out_radial_length"], (nr - 2) / (nr - 1))
        zero = outside(np.zeros((2, 201)))
        self.assertIsNone(zero["records"][0]["hf_out_radial_extent"])
        self.assertFalse(assess_continuum_noise(zero, ContinuumNoiseThresholds(0, 0, 1))["candidate_found"])

    def test_invalid_input_and_thresholds(self):
        for values in ((-.1, .1, .04), (.1, float('nan'), .04), (.1, .1, 0), (.1, .1, 1.1)):
            with self.assertRaises(ValueError):
                ContinuumNoiseThresholds(*values)
        with self.assertRaises(ValueError):
            outside(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            outside(np.full((2, 201), np.nan))
        with self.assertRaises(ValueError):
            measure_continuum_noise(np.ones((1, 201)), 0, np.ones(201), np.ones(201))

    def test_fixed_radial_extent_on_multiple_native_grids(self):
        for nr in (51, 101, 201, 401):
            r = np.linspace(0, 1, nr)
            a = np.zeros((4, nr))
            a[1] = np.exp(-((r - .65) / .08) ** 2)
            tail = (r >= .06) & (r <= .18)
            a[0, tail] = .12 * (-1.) ** np.arange(nr)[tail]
            f = extract_continuum_noise_features(a, 1., np.full(nr, .25),
                                                np.where(r < .3, .64, 2.25))
            self.assertTrue(f["candidate_found"], nr)
            self.assertAlmostEqual(f["witness"]["hf_out_radial_length"], .12, delta=.025)
            disabled = extract_continuum_noise_features(
                a, 1., np.full(nr, .25), np.where(r < .3, .64, 2.25),
                config=ContinuumNoiseThresholds(top2_min=None))
            self.assertEqual(f["records"], disabled["records"])
            self.assertFalse(disabled["candidate_found"])
        # Eight effective grid points mean different physical lengths.
        f = outside(np.ones((1, 201)))
        for nr, expected in ((101, True), (201, True), (401, False)):
            f["records"][0].update(hf_out_top2_ratio=.1, hf_out_local_fraction=.3,
                                    hf_out_radial_length=8 / (nr - 1))
            self.assertEqual(assess_continuum_noise(f, ContinuumNoiseThresholds())["candidate_found"], expected)

    def test_canonical_gate_legacy_presets_and_precedence(self):
        import contextlib
        import io
        from test_rule_sorting import write_mode, write_datcon
        from sort_shot_rules import run_configured_shot, run_shot, parse_args
        from sort_shot_mixed import parse_args as parse_mixed_args, run_rules_method
        from tae_rule_config import load_rule_run_configuration
        from tae_rule_engine import evaluate_mode, BAD_AXIS_SPIKE

        for version in range(5, 10):
            self.assertIsNone(load_rule_run_configuration(
                f"tae_rules_production_v{version}").run_kwargs["continuum_noise_top2_min"])
        flags = ["--shot_dir", "shot", "--out_dir", "out"]
        for flag, value in (("--disable_extended_continuum_noise", []),
                            ("--continuum_noise_top2_min", [".1"]),
                            ("--continuum_noise_local_min", [".3"]),
                            ("--continuum_noise_radial_length_min", [".1"])):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(flags + ["--rule_config", "tae_rules_production_v9", flag] + value)
                with self.assertRaises(SystemExit):
                    parse_mixed_args(flags + [flag] + value)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Native nr=101 proves that production also evaluates other grids.
            for nr in (101, 201):
                r = np.linspace(0, 1, nr)
                a = np.zeros((4, nr))
                a[1] = np.exp(-((r - .65) / .08) ** 2)
                tail = (r >= .06) & (r <= .18)
                a[0, tail] = .08 * (-1.) ** np.arange(nr)[tail]
                a[2, tail] = a[0, tail]
                shot = root / f"shot{nr}"
                write_mode(shot / "N1/egn01w.test", omega=1., ntor=1, nr=nr, mode=a)
                high = np.where((r > .04) & (r < .3), .8, 1.5)
                write_datcon(shot / "N1/datcon1", nr=nr, upper_frequency=high)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    old = run_configured_shot(shot, root / f"old{nr}", rule_config="tae_rules_production_v9",
                                              rule_survivor_policy="accept-as-good-v1")
                    new = run_rules_method(parse_mixed_args([
                        "--shot_dir", str(shot), "--out_dir", str(root / f"new{nr}")]))
                    off = run_shot(shot, root / f"off{nr}", continuum_noise_top2_min=None)
                self.assertEqual(old.final_rows[0]["final_decision"], "GOOD", old.final_rows[0]["rule_primary_reason"])
                self.assertEqual(new.final_rows[0]["rule_primary_reason"], BAD_EXTENDED_CONTINUUM_NOISE)
                self.assertEqual(off.final_rows[0]["final_decision"], "REVIEW")
                fields = ("extended_continuum_noise_gate_enabled", "continuum_noise_top2_min",
                          "continuum_noise_local_min", "continuum_noise_radial_length_min")
                with (root / f"new{nr}/shot_summary_by_n.csv").open() as stream:
                    per_n = next(csv.DictReader(stream))
                self.assertTrue(new.summary[fields[0]])
                for name in fields:
                    self.assertEqual(per_n[name], str(new.summary[name]))
                a[2, 1] = 1
                evidence = dict(path="shot/N1/egn01w.test", mode_key="shot/N1/egn01w.test",
                                shot="shot", ntor=1, omega=1., gamma_d=0,
                                gap_region="tae_like", input_fingerprint="a" * 64)
                result = evaluate_mode(evidence, mode=a, low2=np.full(nr, .25), high2=high**2)
                self.assertTrue(result.features["numerical_structure_features"]["extended_continuum_noise"]["candidate_found"])
                self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)

    def test_cli_measure_and_explicit_sweep_preserve_input(self):
        from test_rule_sorting import write_mode, write_datcon
        from tae_rule_io import input_fingerprint

        script = Path(__file__).resolve().parents[1] / "scripts/audit_continuum_noise.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "shot/N1/egn01w.test"
            r = np.linspace(0, 1, 201)
            a = np.zeros((4, 201)); a[1] = np.exp(-((r - .6) / .1) ** 2)
            a[0, 3:40] = .08 * (-1.) ** np.arange(37)
            write_mode(path, omega=1., ntor=1, nr=201, mode=a)
            datcon = path.parent / "datcon1"
            write_datcon(datcon, nr=201, upper_frequency=np.where(r < .25, .8, 1.5))
            fingerprint = input_fingerprint(path, datcon)
            manifest = root / "input.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["path", "validity", "final_decision", "input_fingerprint"])
                writer.writerow([str(path), "good", "GOOD", fingerprint])
            env = dict(os.environ); env.pop("PYTHONPATH", None)
            out = root / "out"
            subprocess.run([sys.executable, str(script), "measure", "--mode-list", str(manifest),
                            "--data-root", str(root), "--cohort", "synthetic", "--out-dir", str(out)],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["gate_enabled"])
            self.assertEqual(summary["counts"], {"MEASURED": 1})
            self.assertEqual(summary["baseline_fingerprints_verified"], 1)
            saved_measurements = (out / "measurements.jsonl").read_bytes()
            subprocess.run([sys.executable, str(script), "sweep", "--measurements", str(out / "measurements.jsonl"),
                            "--top2-min", ".00001", "--local-min", ".01", "--radial-length-min", ".01",
                            "--export-flags", "--out-dir", str(out / "sweep")],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            with (out / "sweep/threshold_sweep.csv").open() as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["good_survivors_flagged"], "1")
            self.assertEqual(input_fingerprint(path, datcon), fingerprint)
            # A reused baseline cannot silently describe altered raw inputs.
            with path.open("ab") as stream:
                stream.write(b"changed")
            failed = subprocess.run([sys.executable, str(script), "measure", "--mode-list", str(manifest),
                                     "--data-root", str(root), "--cohort", "synthetic", "--out-dir", str(out)],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("baseline fingerprint mismatch", failed.stderr)
            self.assertEqual((out / "measurements.jsonl").read_bytes(), saved_measurements)


if __name__ == "__main__":
    unittest.main()
