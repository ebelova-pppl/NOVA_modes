"""Scientific population checks and production wiring for distributed noise."""

import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distributed_harmonic_noise import (
    BAD_DISTRIBUTED_HARMONIC_NOISE as REASON,
    DistributedNoiseThresholds,
    extract_distributed_noise_features as extract,
)
from test_rule_severity import evaluate
from test_rule_sorting import write_mode, write_datcon
from tae_rule_config import load_rule_run_configuration
from sort_shot_rules import run_configured_shot, parse_args
from sort_shot_mixed import parse_args as parse_mixed_args, run_rules_method


def distributed_mode(nr=201):
    r = np.linspace(0, 1, nr)
    a = np.zeros((5, nr))
    a[0] = np.exp(-((r - .65) / .08) ** 2)
    tail = (r >= .06) & (r <= .16)
    a[1:, tail] = .08 * (-1.) ** np.arange(nr)[tail]
    return a


class DistributedNoiseTests(unittest.TestCase):
    def test_simultaneous_not_sequential_harmonics_and_no_gap_dependency(self):
        a = distributed_mode()
        f = extract(a)
        self.assertTrue(f["candidate_found"])
        self.assertTrue(all(n >= 4 for n in f["witness"]["qualifying_nhf"]))
        sequential = np.zeros_like(a)
        sequential[0] = a[0]
        for h in range(1, 5):
            start = 10 + 28 * (h - 1)
            sequential[h, start:start + 20] = .08 * (-1.) ** np.arange(20)
        self.assertFalse(extract(sequential)["candidate_found"])
        # The entire mode frequency is inside the gap in this engine fixture.
        evaluated = evaluate(a)
        self.assertEqual(evaluated.primary_reason, REASON)
        self.assertFalse(evaluated.features["numerical_structure_features"]
                         ["extended_continuum_noise"]["candidate_found"])

    def test_native_grids_and_unresolved_window(self):
        for nr in (101, 201, 401):
            f = extract(distributed_mode(nr))
            self.assertTrue(f["candidate_found"], nr)
            self.assertTrue(f["resolution_eligible"])
            self.assertLessEqual(f["actual_window_dr"], .05)
            self.assertGreater(f["witness"]["effective_length"], .03)
        f = extract(distributed_mode(21))
        self.assertFalse(f["resolution_eligible"])
        self.assertEqual(f["status"], "WINDOW_UNRESOLVED")
        self.assertFalse(f["candidate_found"])

    def test_no_pooling_short_strong_and_long_negligible_packets(self):
        a = np.zeros((5, 201))
        a[0] = np.exp(-((np.linspace(0, 1, 201) - .65) / .08) ** 2)
        a[1:, 20] = .2
        a[1:, 60:85] = .001 * (-1.) ** np.arange(25)
        self.assertFalse(extract(a)["candidate_found"])

    def test_full_window_raw_energy_and_top2_reference(self):
        a = np.tile((-1.) ** np.arange(201), (4, 1))
        f = extract(a)
        self.assertEqual(f["top2_harmonic_indices"], [0, 1])
        self.assertAlmostEqual(f["top2_energy_share"], .5)
        w = f["witness"]
        self.assertAlmostEqual(w["hf_top2_ratio"], 2 * w["hf_total_fraction"])
        # A smooth energetic harmonic counts in the window denominator even
        # though it supplies none of the qualifying high-pass energy.
        diluted = np.vstack([np.ones(201), a * .01])
        weak = extract(diluted, config=DistributedNoiseThresholds(top2_min=1e-6))
        self.assertFalse(weak["candidate_found"])
        self.assertLess(weak["witness"]["hf_window_fraction"], .001)
        self.assertEqual(f, extract(a * -17))
        self.assertEqual(extract(np.zeros_like(a))["status"], "ZERO_ENERGY")

    def test_strict_cut_equality_severity_and_disabled_diagnostics(self):
        a = np.tile((-1.) ** np.arange(201), (4, 1))
        measured = extract(a)
        exact = DistributedNoiseThresholds(top2_min=measured["witness"]["hf_top2_ratio"])
        self.assertFalse(extract(a, config=exact)["candidate_found"])
        self.assertTrue(extract(a, config=replace(exact, top2_min=np.nextafter(exact.top2_min, 0)))["candidate_found"])
        gate = evaluate(a, distributed_noise_config=exact).features["severity_features"]["gates"][REASON]
        self.assertEqual(gate["severity"], 1)
        self.assertFalse(gate["fired"])
        off = evaluate(a, distributed_noise_config=DistributedNoiseThresholds(top2_min=None))
        self.assertIsNone(off.features["severity_features"]["gates"][REASON]["severity"])
        self.assertEqual(off.features["numerical_structure_features"]["distributed_harmonic_noise"]["witness"]["hf_energy"], measured["witness"]["hf_energy"])

    def test_frozen_presets_cli_production_precedence_and_resolution_warning(self):
        for version in range(5, 12):
            config = load_rule_run_configuration(f"tae_rules_production_v{version}")
            self.assertIsNone(config.run_kwargs["distributed_noise_top2_min"])
        self.assertEqual(load_rule_run_configuration("tae_rules_production_v11").run_kwargs["duplicate_rank_method"], "rule_severity")
        flags = ["--shot_dir", "shot", "--out_dir", "out"]
        for flag in ("nhf_min", "top2_min", "local_min", "radial_length_min", "window_dr"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(flags + ["--rule_config", "tae_rules_production_v12", "--distributed_noise_" + flag, "0.1"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_mixed_args(flags + ["--disable_distributed_harmonic_noise"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for nr in (21, 201):
                shot = root / f"shot{nr}"
                write_mode(shot / "N2/egn02w.test", omega=1., ntor=2, nr=nr, mode=np.pad(distributed_mode(nr), ((0, 3), (0, 0))))
                write_datcon(shot / "N2/datcon2", nr=nr, upper_frequency=np.full(nr, 1.5))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    new = run_rules_method(parse_mixed_args(["--shot_dir", str(shot), "--out_dir", str(root / f"new{nr}")]))
                    old = run_configured_shot(shot, root / f"old{nr}", rule_config="tae_rules_production_v11", rule_survivor_policy="accept-as-good-v1")
                self.assertTrue(new.summary["distributed_harmonic_noise_gate_enabled"])
                self.assertFalse(old.summary["distributed_harmonic_noise_gate_enabled"])
                if nr == 201:
                    self.assertEqual(old.final_rows[0]["final_decision"], "GOOD")
                    self.assertEqual(new.final_rows[0]["rule_primary_reason"], REASON)
                    self.assertGreater(new.final_rows[0]["gate_severity_" + REASON], 1)
                else:
                    text = (root / f"new{nr}/resolution_warnings.txt").read_text()
                    self.assertIn("distributed_harmonic_noise", text)
                    self.assertIn("nr>=41", text)
                with (root / f"new{nr}/shot_summary_by_n.csv").open() as stream:
                    per_n = next(csv.DictReader(stream))
                self.assertEqual(per_n["distributed_noise_top2_min"], str(new.summary["distributed_noise_top2_min"]))
            a = distributed_mode(); a[0, 1] = 1
            both = evaluate(a)
            self.assertEqual(both.primary_reason, "BAD_AXIS_SPIKE")
            self.assertTrue(both.features["numerical_structure_features"]["distributed_harmonic_noise"]["candidate_found"])

    def test_invalid_settings(self):
        for kwargs in (dict(nhf_min=.5), dict(top2_min=0), dict(local_min=float('nan')),
                       dict(window_dr=.02), dict(radial_length_min=.05)):
            with self.assertRaises(ValueError):
                DistributedNoiseThresholds(**kwargs)


if __name__ == "__main__":
    unittest.main()
