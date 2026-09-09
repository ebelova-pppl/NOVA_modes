"""Scientific and workflow checks for the smooth crossing-window exception."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_rule_sorting import write_mode, write_datcon
from sort_shot_rules import parse_args, run_configured_shot
from tae_rule_config import load_rule_run_configuration
from tae_rule_engine import (
    BAD_AXIS_SPIKE,
    BAD_CONT_CROSS_WINDOW,
    NO_GOOD_TEMPLATE,
    ContinuumCrossingConfig,
    ContinuumCrossingWindowConfig,
    evaluate_mode,
    extract_continuum_crossing_tail_features,
    extract_crossing_window_exception_features,
)


def crossing(r=0.4):
    return dict(boundary="high", r_cross=r, W_peak=0.0, shear_weighted=0.0)


def smooth_node(nr=201):
    r = np.linspace(0, 1, nr)
    mode = (
        np.sin(40 * (nr - 1) / 200 * (r - 0.4)) * np.exp(-(((r - 0.55) / 0.25) ** 2))
    )[None, :]
    return mode / np.max(np.abs(mode))


def evaluate(mode, config=None):
    r = np.linspace(0, 1, mode.shape[1])
    row = dict(
        path="example/N1/egn01w.node",
        mode_key="example/N1/egn01w.node",
        shot="example",
        input_fingerprint="a" * 64,
        ntor=1,
        omega=1.0,
        gamma_d=0.0,
        gap_region="tae_like",
    )
    return evaluate_mode(
        row,
        mode=mode,
        low2=np.full_like(r, 0.1),
        high2=1 + r - 0.4,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
        continuum_crossing_window_config=config,
    )


class CrossingWindowExceptionTests(unittest.TestCase):
    def test_empty_individual_window_does_not_invalidate_another_crossing(self):
        # A zero-width calibration window has no sample at an off-grid crossing.
        result = extract_crossing_window_exception_features(
            np.ones((1, 201)),
            [crossing(0.401), crossing(0.55)],
            config=ContinuumCrossingWindowConfig(half_width_grid=0),
        )
        self.assertIsNone(result["records"][0]["window_A_max"])
        self.assertFalse(result["records"][0]["window_violation"])
        self.assertEqual(result["n_violating_crossings"], 1)
        self.assertFalse(result["all_violations_exempted"])

    def test_interpolate_signed_profiles_before_taking_magnitude(self):
        r = np.linspace(0, 1, 201)
        mode = np.clip(40 * (r - 0.4025), -1, 1)[None, :]
        result = extract_crossing_window_exception_features(mode, [crossing(0.4025)])
        record = result["records"][0]
        self.assertAlmostEqual(record["A_cross"], 0.0)
        self.assertAlmostEqual(np.interp(0.4025, r, np.abs(mode[0])), 0.1)
        self.assertGreater(record["window_A_max"], 0.25)
        self.assertTrue(result["all_violations_exempted"])

    def test_every_violating_crossing_must_qualify(self):
        mode = smooth_node()
        clean = extract_crossing_window_exception_features(mode, [crossing()])
        self.assertTrue(clean["all_violations_exempted"])
        multiple = extract_crossing_window_exception_features(
            mode, [crossing(), crossing(0.55)]
        )
        self.assertEqual(multiple["n_violating_crossings"], 2)
        self.assertEqual(multiple["n_exempted_crossings"], 1)
        self.assertEqual(multiple["n_unexcused_crossings"], 1)
        self.assertFalse(multiple["all_violations_exempted"])
        # A high-K, tiny-amplitude crossing that does not violate gate 4 is irrelevant.
        quiet = np.zeros_like(mode)
        quiet[0, :20] = 0.001 * (-1.0) ** np.arange(20)
        quiet = np.vstack([mode, quiet])
        result = extract_crossing_window_exception_features(
            quiet, [crossing(), crossing(0.05)]
        )
        self.assertTrue(result["all_violations_exempted"])
        self.assertGreater(result["records"][0]["K_cross"], 0.1)
        self.assertFalse(result["records"][0]["window_violation"])

    def test_low_point_amplitude_does_not_rescue_rough_crossing(self):
        mode = smooth_node()
        r = np.linspace(0, 1, mode.shape[1])
        rough = 0.15 * (-1.0) ** np.arange(len(r)) * np.exp(-(((r - 0.4) / 0.04) ** 2))
        result = extract_crossing_window_exception_features(
            np.vstack([mode, rough]), [crossing()]
        )
        record = result["records"][0]
        self.assertLess(record["A_cross"], 0.2)
        self.assertGreater(record["K_cross"], 0.1)
        self.assertFalse(result["all_violations_exempted"])

    def test_strict_cuts_disabled_evidence_and_shared_k(self):
        mode = smooth_node()
        records = [crossing(0.401)]
        baseline = extract_crossing_window_exception_features(mode, records)
        record = baseline["records"][0]
        self.assertTrue(baseline["all_violations_exempted"])
        tail = extract_continuum_crossing_tail_features(mode, records)
        self.assertEqual(record["K_cross"], tail["records"][0]["K_cross"])
        for field, measured in [
            ("exception_amplitude_max", record["A_cross"]),
            ("exception_k_max", record["K_cross"]),
        ]:
            exact = ContinuumCrossingWindowConfig(**{field: measured})
            above = replace(exact, **{field: np.nextafter(measured, np.inf)})
            self.assertFalse(
                extract_crossing_window_exception_features(mode, records, config=exact)[
                    "all_violations_exempted"
                ]
            )
            self.assertTrue(
                extract_crossing_window_exception_features(mode, records, config=above)[
                    "all_violations_exempted"
                ]
            )
        disabled = extract_crossing_window_exception_features(
            mode,
            records,
            config=ContinuumCrossingWindowConfig(exception_amplitude_max=None),
        )
        self.assertFalse(disabled["all_violations_exempted"])
        self.assertEqual(disabled["records"][0]["A_cross"], record["A_cross"])
        self.assertEqual(disabled["records"][0]["K_cross"], record["K_cross"])
        zero = extract_crossing_window_exception_features(np.zeros((2, 201)), records)
        self.assertIsNone(zero["records"][0]["K_cross"])
        self.assertFalse(zero["records"][0]["exception_conditions_pass"])

    def test_native_resolution_and_earlier_gates_remain_effective(self):
        mode = smooth_node()
        self.assertEqual(evaluate(mode).primary_reason, NO_GOOD_TEMPLATE)
        self.assertEqual(
            evaluate(
                mode, ContinuumCrossingWindowConfig(exception_amplitude_max=None)
            ).primary_reason,
            BAD_CONT_CROSS_WINDOW,
        )
        other = evaluate(smooth_node(401))
        self.assertEqual(other.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertFalse(
            other.features["crossing_features"]["continuum_crossing_window_exception"][
                "resolution_eligible"
            ]
        )
        mode[0, 0] = 0.8
        earlier = evaluate(mode)
        self.assertEqual(earlier.primary_reason, BAD_AXIS_SPIKE)
        self.assertTrue(
            earlier.features["crossing_features"][
                "continuum_crossing_window_exception"
            ]["all_violations_exempted"]
        )

    def test_configuration_cli_and_real_workflow(self):
        old = load_rule_run_configuration("tae_rules_production_v6")
        new = load_rule_run_configuration("tae_rules_production_v7")
        self.assertIsNone(old.run_kwargs["cross_window_exception_amplitude_max"])
        self.assertEqual(new.run_kwargs["cross_window_exception_amplitude_max"], 0.2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "node_shot"
            path = shot / "N1" / "egn01w.node"
            mode = smooth_node()
            r = np.linspace(0, 1, mode.shape[1])
            write_mode(
                path, mode=np.pad(mode, ((0, 3), (0, 0))), nr=len(r), omega=1.0, ntor=1
            )
            write_datcon(
                path.with_name("datcon1"),
                nr=len(r),
                lower_frequency=np.sqrt(0.1),
                upper_frequency=np.sqrt(1 + r - 0.4),
            )
            previous = run_configured_shot(
                shot, root / "v6", rule_config="tae_rules_production_v6"
            )
            current = run_configured_shot(
                shot, root / "v7", rule_config="tae_rules_production_v7"
            )
            self.assertEqual(
                previous.final_rows[0]["rule_primary_reason"], BAD_CONT_CROSS_WINDOW
            )
            self.assertEqual(
                current.final_rows[0]["rule_primary_reason"], NO_GOOD_TEMPLATE
            )
            self.assertEqual(
                current.summary["continuum_crossing_window_exception_amplitude_max"],
                0.2,
            )
            self.assertEqual(
                current.summary[
                    "n_continuum_crossing_window_exception_resolution_eligible"
                ],
                1,
            )
            flags = [
                "--shot_dir",
                str(shot),
                "--out_dir",
                str(root / "cli"),
                "--disable_cross_window_exception",
            ]
            self.assertTrue(parse_args(flags).disable_cross_window_exception)
            with self.assertRaises(SystemExit):
                parse_args(flags + ["--rule_config", "tae_rules_production_v7"])


if __name__ == "__main__":
    unittest.main()
