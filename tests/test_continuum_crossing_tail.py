"""Scientific invariants and workflow coverage for the crossing-tail gate."""

import contextlib
import csv
import io
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from test_rule_sorting import (
    AxisArtifactConfig,
    ContinuumCrossingConfig,
    ContinuumCrossingWindowConfig,
    EdgeArtifactConfig,
    GridScalePacketConfig,
    GridScaleSpikeConfig,
    InteriorHarmonicIncoherenceConfig,
    InteriorUnresolvedEnvelopeConfig,
    NearAxisGridOscillationConfig,
    evaluate_mode,
    parse_args,
    write_mode,
    write_datcon,
)
from sort_shot_rules import run_configured_shot, run_shot
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME
from tae_rule_engine import (
    BAD_CONT_CROSS_WINDOW,
    BAD_CONTINUUM_CROSSING_TAIL,
    NO_GOOD_TEMPLATE,
    ContinuumCrossingTailConfig,
    extract_continuum_crossing_tail_features,
)


def mode_with_tail(nr=201):
    r = np.linspace(0, 1, nr)
    body = np.exp(-(((r - 0.78) / 0.08) ** 2))
    tail = 0.4 * (-1.0) ** np.arange(nr) * np.exp(-(((r - 0.4) / 0.08) ** 2))
    weak_tail = 0.005 * (-1.0) ** np.arange(nr) * (r < 0.3)
    return np.vstack([body, 0.7 * body, tail, weak_tail])


def crossing(rc=0.4, boundary="high"):
    return {"r_cross": rc, "boundary": boundary, "W_peak": 0.0, "shear_weighted": 0.0}


class CrossingTailTests(unittest.TestCase):
    def test_constant_profiles_have_exact_integrals_and_zero_roughness(self):
        mode = np.tile([[1.0], [0.5], [0.25]], (1, 201))
        features = extract_continuum_crossing_tail_features(mode, [crossing(0.37)])
        record = features["records"][0]
        self.assertAlmostEqual(features["total_energy"], 1 + 0.25 + 0.0625)
        self.assertAlmostEqual(features["top2_energy"], 1.25)
        self.assertAlmostEqual(record["energy_fraction_inner"], 0.37)
        self.assertAlmostEqual(record["energy_fraction_outer"], 0.63)
        self.assertEqual(record["tail_side"], "outer")  # First W maximum is r=0.
        self.assertAlmostEqual(record["tail_over_top2"], 0.63 * 1.3125 / 1.25)
        self.assertEqual(record["K_cross"], 0.0)
        self.assertFalse(features["candidate_found"])

    def test_both_cuts_must_hold_at_the_same_crossing(self):
        mode = mode_with_tail()
        split = extract_continuum_crossing_tail_features(
            mode, [crossing(0.2), crossing(0.7)]
        )
        weak, smooth = split["records"]
        self.assertGreater(weak["K_cross"], 0.4)
        self.assertLess(weak["tail_over_top2"], 0.035)
        self.assertLess(smooth["K_cross"], 0.4)
        self.assertGreater(smooth["tail_over_top2"], 0.035)
        self.assertFalse(split["candidate_found"])
        together = extract_continuum_crossing_tail_features(
            mode, [crossing(0.2), crossing(0.4)]
        )
        self.assertTrue(together["candidate_found"])
        self.assertGreater(
            together["records"][0]["K_cross"], together["witness"]["K_cross"]
        )
        self.assertEqual(together["witness"]["r_cross"], 0.4)

    def test_strict_boundaries_and_disabled_evidence(self):
        mode = mode_with_tail()
        baseline = extract_continuum_crossing_tail_features(mode, [crossing()])
        witness = baseline["witness"]
        self.assertIsNotNone(witness)
        for field, value in [
            ("k_min", witness["K_cross"]),
            ("top2_ratio_min", witness["tail_over_top2"]),
        ]:
            exact = extract_continuum_crossing_tail_features(
                mode, [crossing()], config=ContinuumCrossingTailConfig(**{field: value})
            )
            below = extract_continuum_crossing_tail_features(
                mode,
                [crossing()],
                config=ContinuumCrossingTailConfig(
                    **{field: np.nextafter(value, -np.inf)}
                ),
            )
            self.assertFalse(exact["candidate_found"])
            self.assertTrue(below["candidate_found"])
        disabled = extract_continuum_crossing_tail_features(
            mode, [crossing()], config=ContinuumCrossingTailConfig(k_min=None)
        )
        self.assertIsNone(disabled["candidate_found"])
        self.assertEqual(disabled["records"][0]["K_cross"], witness["K_cross"])
        self.assertEqual(
            disabled["records"][0]["tail_over_top2"], witness["tail_over_top2"]
        )

    def test_scale_padding_and_resolution(self):
        mode = mode_with_tail()
        baseline = extract_continuum_crossing_tail_features(mode, [crossing()])
        for changed in (7 * mode, np.pad(mode, ((0, 10), (0, 0)))):
            result = extract_continuum_crossing_tail_features(changed, [crossing()])
            self.assertEqual(result["candidate_found"], baseline["candidate_found"])
            self.assertEqual(
                result["top2_harmonic_indices"], baseline["top2_harmonic_indices"]
            )
            for name in ("K_cross", "tail_fraction", "tail_over_top2"):
                self.assertAlmostEqual(
                    result["witness"][name], baseline["witness"][name]
                )
        other = extract_continuum_crossing_tail_features(
            mode_with_tail(401), [crossing()]
        )
        self.assertTrue(other["records"][0]["thresholds_pass"])
        self.assertFalse(other["resolution_eligible"])
        self.assertFalse(other["candidate_found"])

    def test_zero_energy_missing_crossings_and_boundary_stencils(self):
        for mode, records in [
            (np.zeros((2, 201)), [crossing()]),
            (mode_with_tail(), []),
        ]:
            result = extract_continuum_crossing_tail_features(mode, records)
            self.assertFalse(result["candidate_found"])
            self.assertIsNone(result["witness"])
        boundary = extract_continuum_crossing_tail_features(
            mode_with_tail(), [crossing(0), crossing(1)]
        )
        self.assertTrue(all(0 < row["n_centers"] <= 4 for row in boundary["records"]))
        short = extract_continuum_crossing_tail_features(np.ones((1, 2)), [crossing()])
        self.assertIsNone(short["records"][0]["K_cross"])
        self.assertFalse(short["resolution_eligible"])

    def test_engine_rejection_disable_and_earlier_reason_precedence(self):
        from continuum_noise import ContinuumNoiseThresholds
        mode = mode_with_tail()
        r = np.linspace(0, 1, mode.shape[1])
        row = {
            "path": "/data/shot/N1/egn01w.tail",
            "mode_key": "shot/N1/egn01w.tail",
            "shot": "shot",
            "input_fingerprint": "a" * 64,
            "ntor": 1,
            "omega": 1.0,
            "gamma_d": 0.01,
            "gap_region": "tae_like",
        }
        kwargs = dict(
            mode=mode,
            low2=np.full(len(r), 0.25),
            high2=(1 + 0.5 * (r - 0.4)) ** 2,
            axis_artifact_config=AxisArtifactConfig(axis_amplitude_min=None),
            grid_scale_spike_config=GridScaleSpikeConfig(amplitude_min=None),
            grid_scale_packet_config=GridScalePacketConfig(amplitude_min=None),
            near_axis_grid_oscillation_config=NearAxisGridOscillationConfig(
                amplitude_min=None
            ),
            continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                amplitude_min=None, w_min=None
            ),
            edge_artifact_config=EdgeArtifactConfig(edge_width_max_grid=None),
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None
            ),
            interior_harmonic_incoherence_config=InteriorHarmonicIncoherenceConfig(
                score_threshold=None
            ),
        )
        result = evaluate_mode(row, **kwargs)
        self.assertEqual(result.primary_reason, BAD_CONTINUUM_CROSSING_TAIL)
        disabled = evaluate_mode(
            row,
            **kwargs,
            continuum_crossing_tail_config=ContinuumCrossingTailConfig(k_min=None),
            continuum_noise_config=ContinuumNoiseThresholds(top2_min=None),
        )
        self.assertEqual(disabled.primary_reason, NO_GOOD_TEMPLATE)
        kwargs["continuum_crossing_window_config"] = ContinuumCrossingWindowConfig()
        earlier = evaluate_mode(row, **kwargs)
        self.assertEqual(earlier.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertTrue(
            earlier.features["crossing_features"]["continuum_crossing_tail"][
                "candidate_found"
            ]
        )

    def test_configuration_validation_and_cli_freeze(self):
        for kwargs in (
            {"k_min": float("nan")},
            {"top2_ratio_min": -1},
            {"half_width_grid": True},
            {"calibrated_n_radial": 2},
        ):
            with self.assertRaises(ValueError):
                ContinuumCrossingTailConfig(**kwargs)
        args = parse_args(
            [
                "--shot_dir",
                "shot",
                "--out_dir",
                "out",
                "--disable_continuum_crossing_tail",
            ]
        )
        self.assertTrue(args.disable_continuum_crossing_tail)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                [
                    "--shot_dir",
                    "shot",
                    "--out_dir",
                    "out",
                    "--rule_config",
                    PRODUCTION_RULE_CONFIG_NAME,
                    "--continuum_crossing_tail_k_min",
                    ".5",
                ]
            )

    def test_workflow_reports_ineligible_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            directory = shot / "N1"
            directory.mkdir(parents=True)
            write_mode(
                directory / "egn01w.tail",
                omega=1.0,
                ntor=1,
                nr=401,
                mode=mode_with_tail(401),
            )
            r = np.linspace(0, 1, 401)
            write_datcon(
                directory / "datcon1", nr=401, upper_frequency=1 + 0.5 * (r - 0.4)
            )
            result = run_configured_shot(shot, Path(temporary) / "out")
            self.assertEqual(
                result.summary["n_continuum_crossing_tail_resolution_ineligible"], 1
            )
            self.assertEqual(
                result.summary["n_continuum_crossing_tail_resolution_eligible"], 0
            )
            self.assertEqual(
                result.summary["continuum_crossing_tail_top2_ratio_min"], 0.035
            )
            with (Path(temporary) / "out/shot_summary_by_n.csv").open() as stream:
                by_n = list(csv.DictReader(stream))
            self.assertEqual(
                by_n[0]["n_continuum_crossing_tail_resolution_ineligible"], "1"
            )

    def test_production_cli_warns_for_unsupported_grid_and_clears_stale_report(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            output = Path(temporary) / "out"
            directory = shot / "N1"
            directory.mkdir(parents=True)
            write_datcon(directory / "datcon1", nr=401)
            write_mode(directory / "egn01w.supported", omega=1., ntor=1, nr=201)
            write_mode(directory / "egn01w.unsupported", omega=1., ntor=1, nr=401)
            command = [sys.executable, str(repo / "scripts/sort_shot_mixed.py"),
                       "--shot_dir", str(shot), "--out_dir", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("WARNING [shot]: 1 TAE-side mode(s)", completed.stderr)
            self.assertIn("interior_harmonic_incoherence: NOT APPLIED", completed.stderr)
            self.assertIn("continuum_crossing_tail: NOT APPLIED", completed.stderr)
            self.assertIn("requires nr=201; nr=401: 1", completed.stderr)
            self.assertIn("1 affected mode(s) are final GOOD", completed.stderr)
            self.assertEqual((output / "resolution_warnings.txt").read_text(), completed.stderr)
            with (output / "resolution_warnings.csv").open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 2)
            self.assertEqual({r["mode_key"] for r in records}, {"shot/N1/egn01w.unsupported"})
            self.assertEqual({r["final_decision"] for r in records}, {"GOOD"})
            # Reuse the output directory with only supported inputs: stale reports clear.
            supported = subprocess.run(command + ["--pattern", "egn01w.supported"],
                                       capture_output=True, text=True, check=False)
            self.assertEqual(supported.returncode, 0, supported.stderr)
            self.assertNotIn("WARNING", supported.stderr)
            with (output / "resolution_warnings.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_intentionally_disabled_gates_do_not_warn(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            output = Path(temporary) / "out"
            directory = shot / "N1"
            directory.mkdir(parents=True)
            write_datcon(directory / "datcon1", nr=401)
            write_mode(directory / "egn01w.mode", omega=1., ntor=1, nr=401)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                run_shot(shot, output, interior_harmonic_incoherence_score_threshold=None,
                         continuum_crossing_tail_k_min=None)
            self.assertEqual(stderr.getvalue(), "")
            with (output / "resolution_warnings.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])


if __name__ == "__main__":
    unittest.main()
