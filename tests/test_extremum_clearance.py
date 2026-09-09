"""Strict extremum-clearance floor and frozen-preset compatibility."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_rule_sorting import narrow_total_energy_mode, write_mode, write_datcon
from tae_rule_engine import (
    evaluate_mode,
    ContinuumCrossingConfig,
    InteriorUnresolvedEnvelopeConfig,
    BAD_INTERIOR_UNRESOLVED_ENVELOPE,
)
from tae_rule_config import load_rule_run_configuration
from sort_shot_rules import run_configured_shot, parse_args


class ExtremumClearanceTests(unittest.TestCase):
    def test_strict_lower_cut_upper_bound_and_legacy_tangency(self):
        mode = narrow_total_energy_mode()
        row = dict(
            path="example/N1/egn01w.test",
            mode_key="example/N1/egn01w.test",
            shot="example",
            ntor=1,
            omega=1.0,
            gamma_d=0.0,
            gap_region="tae_like",
            input_fingerprint="a" * 64,
        )
        cases = [
            (0.0, False),
            (0.0009, False),
            (0.001, False),
            (np.nextafter(0.001, np.inf), True),
            (0.04, True),
            (0.041, False),
        ]
        for clearance, accepted in cases:
            with self.subTest(clearance=clearance), patch(
                "tae_rule_engine.continuum_extremum_features",
                return_value=(
                    {"ext_dr": 0.0, "ext_df_gap": clearance, "ext_energy_frac": 1.0},
                    True,
                ),
            ):
                result = evaluate_mode(
                    row,
                    mode=mode,
                    low2=np.full(65, 0.25),
                    high2=np.full(65, 2.25),
                    continuum_crossing_config=ContinuumCrossingConfig(
                        w_cross_threshold=None
                    ),
                )
                self.assertEqual(result.decision, "REVIEW" if accepted else "BAD")
                if not accepted:
                    self.assertEqual(
                        result.primary_reason, BAD_INTERIOR_UNRESOLVED_ENVELOPE
                    )
                features = result.features["resolution_features"][
                    "interior_unresolved_envelope"
                ]
                self.assertEqual(features["extremum_exception_applied"], accepted)
                self.assertFalse(features["ext_df_gap_min_inclusive"])
        with patch(
            "tae_rule_engine.continuum_extremum_features",
            return_value=(
                {"ext_dr": 0.0, "ext_df_gap": 0.0, "ext_energy_frac": 1.0},
                True,
            ),
        ):
            legacy = evaluate_mode(
                row,
                mode=mode,
                low2=np.full(65, 0.25),
                high2=np.full(65, 2.25),
                interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                    ext_df_gap_min=0.0, ext_df_gap_min_inclusive=True
                ),
            )
        self.assertEqual(legacy.decision, "REVIEW")

    def test_v7_v8_runs_preserve_width_and_only_tighten_clearance(self):
        for version in (5, 6, 7):
            old = load_rule_run_configuration(f"tae_rules_production_v{version}")
            self.assertEqual(old.run_kwargs["interior_envelope_ext_df_gap_min"], 0.0)
            self.assertTrue(
                old.run_kwargs["interior_envelope_ext_df_gap_min_inclusive"]
            )
        old = load_rule_run_configuration("tae_rules_production_v7")
        new = load_rule_run_configuration("tae_rules_production_v8")
        differences = {
            k for k in old.run_kwargs if old.run_kwargs[k] != new.run_kwargs[k]
        }
        self.assertEqual(
            differences,
            {
                "interior_envelope_ext_df_gap_min",
                "interior_envelope_ext_df_gap_min_inclusive",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "test_shot"
            r = np.linspace(0, 1, 201)
            mode = np.zeros((4, 201))
            mode[1] = np.exp(-(((r - 0.25) / 0.005) ** 2))
            write_mode(shot / "N1/egn01w.test", omega=1.0, ntor=1, nr=201, mode=mode)
            write_datcon(
                shot / "N1/datcon1",
                nr=201,
                upper_frequency=1.0005 + 5 * (r - 0.25) ** 2,
            )
            before = run_configured_shot(
                shot, root / "old", rule_config="tae_rules_production_v7"
            )
            after = run_configured_shot(
                shot, root / "new", rule_config="tae_rules_production_v8"
            )
            self.assertEqual(before.final_rows[0]["rule_decision"], "REVIEW")
            self.assertEqual(
                after.final_rows[0]["rule_primary_reason"],
                BAD_INTERIOR_UNRESOLVED_ENVELOPE,
            )
            self.assertEqual(after.summary["interior_envelope_ext_df_gap_min"], 0.001)
            self.assertFalse(
                after.summary["interior_envelope_ext_df_gap_min_inclusive"]
            )
            self.assertEqual(after.summary["interior_envelope_width_max_grid"], 2.0)
            self.assertTrue(
                before.summary["interior_envelope_ext_df_gap_min_inclusive"]
            )

    def test_cli_default_and_legacy_comparison_are_auditable(self):
        flags = ["--shot_dir", "shot", "--out_dir", "out"]
        args = parse_args(flags)
        self.assertEqual(args.interior_envelope_ext_df_gap_min, 0.001)
        self.assertFalse(args.interior_envelope_ext_df_gap_min_inclusive)
        args = parse_args(flags + ["--interior_envelope_ext_df_gap_min_inclusive"])
        self.assertTrue(args.interior_envelope_ext_df_gap_min_inclusive)
        with self.assertRaises(SystemExit):
            parse_args(
                flags
                + [
                    "--rule_config",
                    "tae_rules_production_v8",
                    "--interior_envelope_ext_df_gap_min_inclusive",
                ]
            )
        with self.assertRaises(ValueError):
            InteriorUnresolvedEnvelopeConfig(ext_df_gap_min_inclusive="false")


if __name__ == "__main__":
    unittest.main()
