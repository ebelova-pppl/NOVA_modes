import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from label_modes_fast import adjudication_candidate_rows  # noqa: E402
from make_tae_like_list import (  # noqa: E402
    classify_gap_region,
    preprocess_shot,
)
from mode_features import (  # noqa: E402
    EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES,
    EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES,
    RF_FEATURE_NAMES,
    compute_features_for_mode,
)
from nova_mode_loader import load_mode_from_nova  # noqa: E402
from sort_shot_mixed import classify_gap_region as mixed_classify_gap_region  # noqa: E402
from sort_shot_rules import (  # noqa: E402
    OverrideAudit,
    apply_manual_overrides,
    build_summary,
    deduplicate_final_good,
    load_manual_overrides,
    run_shot,
)
from tae_rule_engine import (  # noqa: E402
    BAD_AXIS_SPIKE,
    BAD_CONT_CROSS,
    BAD_EDGE_SPIKE,
    BAD_GRID_SCALE_SPIKE,
    NO_GOOD_TEMPLATE,
    RULE_FEATURE_EXTRACTION_FAILED,
    RULE_FEATURE_GROUP_NAMES,
    RULE_FEATURE_METADATA_NAMES,
    RULE_FEATURE_NAMES,
    RULE_FEATURE_SCHEMA_VERSION,
    RULE_FEATURE_SOURCE_SCHEMA_VERSION,
    AxisArtifactConfig,
    ContinuumCrossingConfig,
    EdgeArtifactConfig,
    GridScaleSpikeConfig,
    evaluate_mode,
    extract_axis_artifact_features,
    extract_edge_artifact_features,
    extract_grid_scale_spike_features,
)
from tae_rule_io import (  # noqa: E402
    MANUAL_OVERRIDE_FIELDS,
    RULE_OUTPUT_FIELDS,
    input_fingerprint,
    portable_mode_key,
    read_dict_csv,
    sha256_file,
    stable_json,
    write_dict_csv,
)


REQUIRED_OUTPUTS = {
    "all_modes_rules.csv",
    "tae_like_all.csv",
    "eae_like.csv",
    "rejected_modes.csv",
    "rule_results.csv",
    "final_classifications.csv",
    "bad_tae_like.csv",
    "review_tae_like.csv",
    "good_tae_unchecked.csv",
    "good_tae_final.csv",
    "manual_overrides.csv",
    "shot_summary.csv",
    "shot_summary_wide.csv",
    "shot_summary_by_n.csv",
    "frequency_cluster_report.txt",
    "frequency_clusters.csv",
}


def flatten_grouped_rule_features(features: dict) -> dict:
    """Return only the 31 shared measurements from the grouped audit object."""
    flattened = dict(features["rf_standard_features"])
    flattened.update(features["crossing_features"])
    flattened.update(
        {
            name: features["extremum_features"][name]
            for name in EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES
        }
    )
    return flattened


def write_mode(
    path: Path,
    *,
    omega: float,
    ntor: int,
    nr: int = 10,
    mode: np.ndarray | None = None,
) -> np.ndarray:
    nhar = 4 * ntor
    if mode is None:
        r = np.linspace(0.0, 1.0, nr)
        envelope = np.exp(-((r - 0.4) / 0.18) ** 2)
        mode = np.stack([(index + 1) * envelope for index in range(nhar)])
        mode /= np.max(np.abs(mode))
    payload = np.zeros((3, nhar, nr), dtype=float)
    payload[0] = mode
    values = np.concatenate(
        (
            np.array([omega], dtype=float),
            payload.reshape(-1),
            np.array([nr, 0.01, ntor], dtype=float),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(path)
    return mode


def write_datcon(
    path: Path,
    *,
    nr: int = 10,
    lower_frequency: float | np.ndarray = 0.5,
    upper_frequency: float | np.ndarray = 1.5,
) -> None:
    lower = np.broadcast_to(np.asarray(lower_frequency, dtype=float), (nr,))
    upper = np.broadcast_to(np.asarray(upper_frequency, dtype=float), (nr,))
    lines = [f"1 {nr}"]
    lines.extend(f"{low**2:.17g} {high**2:.17g}" for low, high in zip(lower, upper))
    path.write_text("\n".join(lines) + "\n")


def make_tae_shot(root: Path, *, names: tuple[str, ...] = ("egn01w.one",)) -> Path:
    shot = root / "synthetic_shot"
    n_dir = shot / "N1"
    n_dir.mkdir(parents=True)
    write_datcon(n_dir / "datcon1")
    for index, name in enumerate(names):
        write_mode(n_dir / name, omega=1.0 + 0.005 * index, ntor=1)
    return shot


def override_row(result_row: dict[str, str], decision: str = "GOOD") -> dict[str, str]:
    return {
        "mode_key": result_row["mode_key"],
        "path": result_row["path"],
        "input_fingerprint": result_row["input_fingerprint"],
        "ntor": result_row["ntor"],
        "frequency": result_row["omega"],
        "original_rule_decision": result_row["rule_decision"],
        "manual_decision": decision,
        "manual_reason": "explicit synthetic adjudication",
        "reviewer": "unit-test",
        "adjudication_timestamp": "2026-08-15T00:00:00Z",
    }


class PreprocessingTests(unittest.TestCase):
    def test_missing_datcon_aborts_before_mode_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_missing_datcon"
            n_dir = shot / "N1"
            n_dir.mkdir(parents=True)
            (n_dir / "egn01w.broken").write_bytes(b"not a NOVA mode")
            with self.assertRaisesRegex(SystemExit, "required continuum file is missing"):
                preprocess_shot(shot)

    def test_malformed_mode_is_invalid_while_valid_mode_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = make_tae_shot(Path(temporary))
            (shot / "N1" / "egn01w.broken").write_bytes(b"broken")
            result = preprocess_shot(shot)

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(len(result.tae_rows), 1)
        self.assertEqual(len(result.invalid_rows), 1)
        self.assertEqual(
            result.invalid_rows[0]["preprocessing_primary_reason"], "MODE_LOAD_FAILED"
        )

    def test_tae_eae_mixed_routing_matches_canonical_function(self):
        cases = [(0.3, 0.6), (-0.8, 0.2), (0.2, 0.45)]
        for signed_delta, fraction in cases:
            expected = mixed_classify_gap_region(
                signed_delta,
                fraction,
                fraction_tae_threshold=0.5,
                fraction_eae_threshold=0.4,
                signed_delta_eae_threshold=-0.1,
            )
            actual = classify_gap_region(signed_delta, fraction)
            self.assertEqual(actual, expected)

    def test_input_fingerprint_changes_for_either_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = make_tae_shot(Path(temporary))
            mode = shot / "N1" / "egn01w.one"
            datcon = shot / "N1" / "datcon1"
            original = input_fingerprint(mode, datcon)
            mode.write_bytes(mode.read_bytes() + b"x")
            changed_mode = input_fingerprint(mode, datcon)
            mode.write_bytes(mode.read_bytes()[:-1])
            datcon.write_text(datcon.read_text() + "\n")
            changed_datcon = input_fingerprint(mode, datcon)

        self.assertNotEqual(original, changed_mode)
        self.assertNotEqual(original, changed_datcon)


class RuleAndOverrideTests(unittest.TestCase):
    def setUp(self):
        radial_grid = np.linspace(0.0, 1.0, 201)
        envelope = np.exp(-((radial_grid - 0.2) / 0.03) ** 2)
        self.mode = np.stack(
            [(index + 1) * envelope for index in range(4)]
        )
        self.mode /= np.max(np.abs(self.mode))
        lower = np.full_like(radial_grid, 0.5)
        upper = 0.9 + 4.0 * (radial_grid - 0.2) ** 2
        self.low2 = lower**2
        self.high2 = upper**2
        self.base = {
            "path": "/data/shot/N1/egn01w.one",
            "mode_key": "shot/N1/egn01w.one",
            "shot": "shot",
            "ntor": 1,
            "omega": 1.0,
            "gamma_d": 0.01,
            "input_fingerprint": "a" * 64,
            "gap_region": "tae_like",
            "processing_status": "READY_FOR_RULES",
        }

    def evaluate(
        self,
        row=None,
        axis_config=None,
        grid_config=None,
        crossing_config=None,
        edge_config=None,
    ):
        return evaluate_mode(
            self.base if row is None else row,
            mode=self.mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=axis_config,
            grid_scale_spike_config=grid_config,
            continuum_crossing_config=crossing_config,
            edge_artifact_config=edge_config,
        )

    def test_unmatched_partial_ruleset_returns_review_with_valid_json(self):
        result = self.evaluate()
        row = result.as_output_row(self.base)
        self.assertEqual(row["rule_decision"], "REVIEW")
        self.assertEqual(row["rule_primary_reason"], NO_GOOD_TEMPLATE)
        self.assertEqual(json.loads(row["rule_triggered_rules"]), [NO_GOOD_TEMPLATE])
        features = json.loads(row["rule_features"])
        self.assertEqual(
            features["feature_schema_version"], RULE_FEATURE_SCHEMA_VERSION
        )
        self.assertEqual(
            set(features) - set(RULE_FEATURE_METADATA_NAMES),
            set(RULE_FEATURE_GROUP_NAMES),
        )
        self.assertEqual(len(RULE_FEATURE_NAMES), 31)
        self.assertEqual(
            features["source_feature_schema_version"],
            RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            set(features["rf_standard_features"]), set(RF_FEATURE_NAMES)
        )
        self.assertEqual(
            set(features["crossing_features"]),
            set(EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES),
        )
        self.assertEqual(
            set(features["extremum_features"]),
            {"match_found", *EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES},
        )
        self.assertTrue(features["extremum_features"]["match_found"])
        self.assertEqual(features["resolution_features"], {})
        grid_features = features["numerical_structure_features"][
            "grid_scale_spike"
        ]
        self.assertFalse(grid_features["grid_scale_candidate_found"])
        self.assertEqual(grid_features["grid_scale_candidate_width_limit_grid"], 1.0)
        axis_features = features["boundary_features"]["axis_artifact"]
        self.assertEqual(axis_features["r_ax"], 0.03)
        self.assertEqual(axis_features["axis_peak_harmonic_index"], 3)
        self.assertFalse(axis_features["axis_peak_is_local_max"])
        self.assertGreater(axis_features["axis_halfmax_outer_edge_r"], 0.03)
        edge_features = features["boundary_features"]["edge_artifact"]
        self.assertEqual(edge_features["r_edge_min"], 0.97)
        self.assertFalse(edge_features["edge_energy_peak_in_window"])
        self.assertEqual(edge_features["edge_harmonic_peak_harmonic_index"], 3)
        self.assertNotIn("signed_delta", row["rule_features"])
        self.assertNotIn("fraction_below_upper2", row["rule_features"])
        self.assertEqual(stable_json({"missing": float("nan")}), '{"missing":null}')

    def test_named_rule_features_match_the_shared_rf31_vector(self):
        row = self.evaluate().as_output_row(self.base)
        features = json.loads(row["rule_features"])
        vector = compute_features_for_mode(
            self.mode,
            extra_info={
                "path": self.base["path"],
                "omega": self.base["omega"],
                "gamma_d": self.base["gamma_d"],
                "ntor": self.base["ntor"],
            },
            include_crossing_features=True,
            include_extremum_features=True,
            continuum_arrays=(self.low2, self.high2),
            strict_continuum=True,
        )
        flattened = flatten_grouped_rule_features(features)
        np.testing.assert_allclose(
            [flattened[name] for name in RULE_FEATURE_NAMES], vector
        )
        crossing_features = features["crossing_features"]
        crossing_records = features["crossing_records"]
        self.assertGreater(crossing_features["n_cross"], 0)
        self.assertEqual(len(crossing_records), int(crossing_features["n_cross"]))
        self.assertLess(features["extremum_features"]["ext_dr"], 0.01)
        self.assertTrue(
            all(
                set(record)
                == {"boundary", "r_cross", "W_peak", "shear_weighted"}
                for record in crossing_records
            )
        )
        order = [
            (
                0 if record["boundary"] == "low" else 1,
                record["r_cross"],
            )
            for record in crossing_records
        ]
        self.assertEqual(order, sorted(order))
        self.assertAlmostEqual(
            crossing_features["W_star_sum"],
            sum(record["W_peak"] for record in crossing_records),
        )
        self.assertAlmostEqual(
            crossing_features["W_star_high_shear_sum"],
            sum(record["shear_weighted"] for record in crossing_records),
        )
        strongest = max(
            crossing_records,
            key=lambda record: (record["W_peak"], record["r_cross"]),
        )
        self.assertAlmostEqual(
            crossing_features["r_star_max"], strongest["r_cross"]
        )

    def test_no_extremum_uses_explicit_status_and_null_measurements(self):
        flat_low2 = np.full(self.mode.shape[1], 0.5**2)
        flat_high2 = np.full(self.mode.shape[1], 1.5**2)
        result = evaluate_mode(
            self.base,
            mode=self.mode,
            low2=flat_low2,
            high2=flat_high2,
        )
        row = result.as_output_row(self.base)
        features = json.loads(row["rule_features"])
        self.assertEqual(row["rule_decision"], "REVIEW")
        self.assertFalse(features["extremum_features"]["match_found"])
        for name in ("ext_dr", "ext_df_gap", "ext_energy_frac"):
            self.assertIsNone(features["extremum_features"][name])
        self.assertEqual(features["crossing_records"], [])
        self.assertEqual(features["crossing_features"]["n_cross"], 0.0)
        self.assertIsNone(features["crossing_features"]["r_star_max"])
        self.assertIsNone(features["crossing_features"]["r_star_high_shear"])

    def test_missing_feature_arrays_return_invalid_with_null_features(self):
        result = evaluate_mode(self.base)
        row = result.as_output_row(self.base)
        features = json.loads(row["rule_features"])
        self.assertEqual(row["rule_decision"], "INVALID")
        self.assertEqual(row["rule_primary_reason"], RULE_FEATURE_EXTRACTION_FAILED)
        flattened = flatten_grouped_rule_features(features)
        self.assertTrue(all(flattened[name] is None for name in RULE_FEATURE_NAMES))
        self.assertIsNone(features["extremum_features"]["match_found"])
        self.assertEqual(features["crossing_records"], [])
        self.assertIsNone(
            features["boundary_features"]["axis_artifact"]["axis_peak"]
        )
        self.assertIsNone(
            features["numerical_structure_features"]["grid_scale_spike"][
                "grid_scale_candidate_found"
            ]
        )

    def test_narrow_axis_spike_fires_first_bad_gate(self):
        mode = np.zeros_like(self.mode)
        mode[2, 2] = 1.0
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=AxisArtifactConfig(
                axis_amplitude_min=0.8,
                axis_width_max_grid=2.0,
            ),
        )
        row = result.as_output_row(self.base)
        axis_features = json.loads(row["rule_features"])["boundary_features"][
            "axis_artifact"
        ]
        self.assertEqual(row["rule_decision"], "BAD")
        self.assertEqual(row["rule_primary_reason"], BAD_AXIS_SPIKE)
        self.assertEqual(json.loads(row["rule_triggered_rules"]), [BAD_AXIS_SPIKE])
        self.assertEqual(axis_features["axis_peak_harmonic_index"], 2)
        self.assertAlmostEqual(axis_features["axis_peak_r"], 0.01)
        self.assertTrue(axis_features["axis_peak_is_local_max"])
        self.assertAlmostEqual(axis_features["axis_halfmax_width_grid"], 1.0)
        self.assertFalse(axis_features["axis_component_touches_boundary"])

    def test_axis_window_includes_peak_exactly_at_r_ax(self):
        mode = np.zeros_like(self.mode)
        mode[2, 5] = 0.7
        mode[2, 6] = 1.0
        mode[2, 7] = 0.2
        features = extract_axis_artifact_features(mode)
        self.assertAlmostEqual(features["axis_peak_r"], 0.03)
        self.assertEqual(features["axis_peak"], 1.0)
        self.assertTrue(features["axis_peak_is_local_max"])
        self.assertLess(features["axis_halfmax_width_grid"], 10.0)

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
        )
        self.assertEqual(result.decision, "BAD")
        self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)

    def test_grid_scale_spike_uses_signed_lobe_width(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80] = 0.7
        mode[1, 81] = -0.7
        features = extract_grid_scale_spike_features(mode, width_max_grid=1.0)

        self.assertTrue(features["grid_scale_candidate_found"])
        self.assertEqual(features["grid_scale_peak_harmonic_index"], 1)
        self.assertAlmostEqual(features["grid_scale_peak_r"], 0.4)
        self.assertEqual(features["grid_scale_peak_sign"], 1)
        self.assertAlmostEqual(features["grid_scale_peak"], 0.7)
        self.assertAlmostEqual(features["grid_scale_halfmax_width_grid"], 0.75)

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
        )
        row = result.as_output_row(self.base)
        self.assertEqual(row["rule_decision"], "BAD")
        self.assertEqual(row["rule_primary_reason"], BAD_GRID_SCALE_SPIKE)
        self.assertEqual(
            json.loads(row["rule_triggered_rules"]),
            [BAD_GRID_SCALE_SPIKE],
        )

    def test_grid_scale_gate_requires_both_amplitude_and_width(self):
        low_amplitude = np.zeros_like(self.mode)
        low_amplitude[1, 80] = 0.29
        low_result = evaluate_mode(
            self.base,
            mode=low_amplitude,
            low2=self.low2,
            high2=self.high2,
        )
        self.assertEqual(low_result.decision, "REVIEW")

        resolved = np.zeros_like(self.mode)
        resolved[1, 78:83] = [0.2, 0.4, 0.6, 0.4, 0.2]
        resolved_result = evaluate_mode(
            self.base,
            mode=resolved,
            low2=self.low2,
            high2=self.high2,
        )
        self.assertEqual(resolved_result.decision, "REVIEW")
        features = extract_grid_scale_spike_features(
            resolved,
            width_max_grid=1.0,
        )
        self.assertFalse(features["grid_scale_candidate_found"])

    def test_null_grid_scale_amplitude_disables_gate_but_keeps_measurement(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80] = 0.7
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            grid_scale_spike_config=GridScaleSpikeConfig(
                amplitude_min=None,
                width_max_grid=1.0,
            ),
        )
        features = result.features["numerical_structure_features"][
            "grid_scale_spike"
        ]
        self.assertEqual(result.decision, "REVIEW")
        self.assertTrue(features["grid_scale_candidate_found"])
        self.assertAlmostEqual(features["grid_scale_peak"], 0.7)

    def test_grid_scale_gate_includes_one_sided_radial_endpoint(self):
        mode = np.zeros_like(self.mode)
        mode[2, -1] = -0.5
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
        )
        features = result.features["numerical_structure_features"][
            "grid_scale_spike"
        ]
        self.assertEqual(result.primary_reason, BAD_GRID_SCALE_SPIKE)
        self.assertAlmostEqual(features["grid_scale_peak_r"], 1.0)
        self.assertAlmostEqual(features["grid_scale_halfmax_width_grid"], 0.5)
        self.assertTrue(features["grid_scale_component_touches_boundary"])

    def test_axis_gate_precedes_grid_scale_gate(self):
        mode = np.zeros_like(self.mode)
        mode[1, 2] = 1.0
        mode[2, 80] = 0.8
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
        )
        self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)
        self.assertEqual(result.triggered_rules, (BAD_AXIS_SPIKE,))
        grid_features = result.features["numerical_structure_features"][
            "grid_scale_spike"
        ]
        self.assertTrue(grid_features["grid_scale_candidate_found"])

    def test_cont_cross_gate_requires_crossing_and_strictly_exceeds_threshold(self):
        mode = np.zeros_like(self.mode)
        crossing_index = 80
        mode[0, 100] = 1.0
        mode[0, crossing_index] = np.sqrt(0.06)
        upper = np.full(mode.shape[1], 1.1)
        upper[:crossing_index] = 0.9
        upper[crossing_index] = 1.0
        disabled_axis = AxisArtifactConfig(
            axis_amplitude_min=None,
            axis_width_max_grid=None,
        )
        disabled_grid = GridScaleSpikeConfig(
            amplitude_min=None,
            width_max_grid=None,
        )

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=0.05
            ),
        )
        self.assertEqual(result.decision, "BAD")
        self.assertEqual(result.primary_reason, BAD_CONT_CROSS)
        self.assertEqual(result.triggered_rules, (BAD_CONT_CROSS,))
        self.assertEqual(result.features["crossing_features"]["n_cross"], 1.0)
        self.assertGreater(
            result.features["rf_standard_features"]["W_star_max"],
            0.05,
        )

        disabled_result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=None
            ),
        )
        self.assertEqual(disabled_result.decision, "REVIEW")
        self.assertEqual(
            disabled_result.features["crossing_features"]["n_cross"],
            1.0,
        )

        measured_threshold = result.features["rf_standard_features"][
            "W_star_max"
        ]
        equal_result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=measured_threshold
            ),
        )
        self.assertEqual(equal_result.decision, "REVIEW")

        no_cross_result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=np.full(mode.shape[1], 1.5**2),
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=0.0
            ),
        )
        self.assertEqual(no_cross_result.decision, "REVIEW")
        self.assertEqual(
            no_cross_result.features["crossing_features"]["n_cross"],
            0.0,
        )

    def test_grid_scale_gate_precedes_cont_cross_gate(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80] = 0.7
        mode[1, 81] = -0.7
        upper = np.full(mode.shape[1], 1.1)
        upper[:80] = 0.9
        upper[80] = 1.0
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
        )
        self.assertGreater(
            result.features["rf_standard_features"]["W_star_max"],
            0.05,
        )
        self.assertEqual(result.primary_reason, BAD_GRID_SCALE_SPIKE)
        self.assertEqual(result.triggered_rules, (BAD_GRID_SCALE_SPIKE,))

    def test_edge_gate_uses_global_energy_envelope_and_full_grid_width(self):
        mode = np.zeros_like(self.mode)
        mode[1, 194:199] = [0.4, 0.8, 1.0, 0.8, 0.4]
        features = extract_edge_artifact_features(mode)

        self.assertTrue(features["edge_energy_peak_in_window"])
        self.assertAlmostEqual(features["edge_energy_peak_r"], 0.98)
        self.assertAlmostEqual(features["edge_energy_halfmax_width_grid"], 2.5833333333)
        self.assertAlmostEqual(features["edge_harmonic_peak_r"], 0.98)
        self.assertAlmostEqual(features["edge_harmonic_halfmax_width_grid"], 3.5)

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=np.full(mode.shape[1], 0.5**2),
            high2=np.full(mode.shape[1], 1.5**2),
        )
        self.assertEqual(result.decision, "BAD")
        self.assertEqual(result.primary_reason, BAD_EDGE_SPIKE)
        self.assertEqual(result.triggered_rules, (BAD_EDGE_SPIKE,))

    def test_edge_gate_is_inclusive_and_can_be_disabled(self):
        mode = np.zeros_like(self.mode)
        mode[1, 192:197] = [0.4, 0.8, 1.0, 0.8, 0.4]
        low2 = np.full(mode.shape[1], 0.5**2)
        high2 = np.full(mode.shape[1], 1.5**2)

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=low2,
            high2=high2,
        )
        self.assertEqual(result.primary_reason, BAD_EDGE_SPIKE)
        self.assertAlmostEqual(
            result.features["boundary_features"]["edge_artifact"][
                "edge_energy_peak_r"
            ],
            0.97,
        )

        disabled = evaluate_mode(
            self.base,
            mode=mode,
            low2=low2,
            high2=high2,
            edge_artifact_config=EdgeArtifactConfig(
                r_edge_min=0.97,
                edge_width_max_grid=None,
            ),
        )
        self.assertEqual(disabled.decision, "REVIEW")
        self.assertTrue(
            disabled.features["boundary_features"]["edge_artifact"][
                "edge_energy_peak_in_window"
            ]
        )

    def test_narrow_edge_harmonic_is_audit_only_when_total_peak_is_interior(self):
        radial_grid = np.linspace(0.0, 1.0, self.mode.shape[1])
        mode = np.zeros_like(self.mode)
        mode[0] = np.exp(-((radial_grid - 0.5) / 0.08) ** 2)
        mode[1, 194:199] = [0.2, 0.4, 0.5, 0.4, 0.2]
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=np.full(mode.shape[1], 0.5**2),
            high2=np.full(mode.shape[1], 1.5**2),
        )
        features = result.features["boundary_features"]["edge_artifact"]

        self.assertEqual(result.decision, "REVIEW")
        self.assertFalse(features["edge_energy_peak_in_window"])
        self.assertAlmostEqual(features["edge_energy_peak_r"], 0.5)
        self.assertAlmostEqual(features["edge_harmonic_peak"], 0.5)
        self.assertAlmostEqual(features["edge_harmonic_peak_r"], 0.98)

    def test_cont_cross_gate_precedes_edge_gate(self):
        mode = np.zeros_like(self.mode)
        mode[1, 194:199] = [0.4, 0.8, 1.0, 0.8, 0.4]
        upper = np.full(mode.shape[1], 1.1)
        upper[:196] = 0.9
        upper[196] = 1.0
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=np.full(mode.shape[1], 0.5**2),
            high2=upper**2,
        )

        self.assertGreater(
            result.features["rf_standard_features"]["W_star_max"],
            0.05,
        )
        self.assertTrue(
            result.features["boundary_features"]["edge_artifact"][
                "edge_energy_peak_in_window"
            ]
        )
        self.assertEqual(result.primary_reason, BAD_CONT_CROSS)
        self.assertEqual(result.triggered_rules, (BAD_CONT_CROSS,))

    def test_null_thresholds_disable_axis_gate_but_keep_measurements(self):
        mode = np.zeros_like(self.mode)
        mode[1, 0] = 1.0
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=AxisArtifactConfig(
                axis_amplitude_min=None,
                axis_width_max_grid=None,
            ),
            grid_scale_spike_config=GridScaleSpikeConfig(
                amplitude_min=None,
                width_max_grid=1.0,
            ),
        )
        row = result.as_output_row(self.base)
        axis_features = json.loads(row["rule_features"])["boundary_features"][
            "axis_artifact"
        ]
        self.assertEqual(row["rule_decision"], "REVIEW")
        self.assertEqual(row["rule_primary_reason"], NO_GOOD_TEMPLATE)
        self.assertTrue(axis_features["axis_peak_is_local_max"])
        self.assertTrue(axis_features["axis_component_touches_boundary"])

    def test_rising_axis_flank_is_not_a_local_max(self):
        mode = np.zeros_like(self.mode)
        mode[0, :17] = np.linspace(0.1, 0.9, 17)
        features = extract_axis_artifact_features(mode)
        self.assertAlmostEqual(features["axis_peak_r"], 0.03)
        self.assertFalse(features["axis_peak_is_local_max"])

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=AxisArtifactConfig(
                axis_amplitude_min=0.1,
                axis_width_max_grid=100.0,
            ),
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=None
            ),
        )
        self.assertEqual(result.decision, "REVIEW")

    def test_halfmax_component_uses_full_radial_grid(self):
        mode = np.zeros_like(self.mode)
        mode[0, :9] = [0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        features = extract_axis_artifact_features(mode)
        self.assertTrue(features["axis_peak_is_local_max"])
        self.assertTrue(features["axis_component_touches_boundary"])
        self.assertAlmostEqual(features["axis_halfmax_outer_edge_r"], 0.035)
        self.assertAlmostEqual(features["axis_halfmax_width_grid"], 7.0)
        self.assertGreater(features["axis_halfmax_outer_edge_r"], features["r_ax"])

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=AxisArtifactConfig(
                axis_amplitude_min=0.8,
                axis_width_max_grid=3.0,
            ),
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=None
            ),
        )
        self.assertEqual(result.decision, "REVIEW")

    def test_override_changes_final_but_preserves_rule_decision(self):
        row = self.evaluate().as_output_row(self.base)
        override = override_row({key: str(value) for key, value in row.items()})
        final, audit = apply_manual_overrides([row], [override])
        self.assertEqual(final[0]["rule_decision"], "REVIEW")
        self.assertEqual(final[0]["final_decision"], "GOOD")
        self.assertEqual(final[0]["decision_source"], "manual_override")
        self.assertEqual(audit.applied, 1)
        self.assertEqual(audit.decisions_changed, 1)

    def test_all_preliminary_classes_can_be_overridden(self):
        rows = []
        overrides = []
        for index, (rule_decision, manual_decision) in enumerate(
            (("GOOD", "BAD"), ("BAD", "REVIEW"), ("REVIEW", "GOOD")), start=1
        ):
            row = self.evaluate().as_output_row(self.base)
            row.update(
                {
                    "path": f"/data/shot/N1/egn01w.{index}",
                    "mode_key": f"shot/N1/egn01w.{index}",
                    "input_fingerprint": str(index) * 64,
                    "rule_decision": rule_decision,
                    "final_decision": rule_decision,
                }
            )
            rows.append(row)
            overrides.append(
                override_row({key: str(value) for key, value in row.items()}, manual_decision)
            )
        final, audit = apply_manual_overrides(rows, overrides)
        self.assertEqual([row["final_decision"] for row in final], ["BAD", "REVIEW", "GOOD"])
        self.assertEqual(audit.applied, 3)

    def test_stale_and_ambiguous_overrides_are_not_applied(self):
        row = self.evaluate().as_output_row(self.base)
        stale = override_row({key: str(value) for key, value in row.items()})
        stale["input_fingerprint"] = "b" * 64
        final, audit = apply_manual_overrides([row], [stale])
        self.assertEqual(final[0]["final_decision"], "REVIEW")
        self.assertEqual(final[0]["override_status"], "STALE_FINGERPRINT")
        self.assertEqual(audit.stale, 1)

        valid = override_row({key: str(value) for key, value in row.items()})
        final, audit = apply_manual_overrides([row], [valid, dict(valid)])
        self.assertEqual(final[0]["final_decision"], "REVIEW")
        self.assertEqual(final[0]["override_status"], "AMBIGUOUS_OVERRIDE")
        self.assertEqual(audit.ambiguous, 2)

    def test_summary_counts_only_primary_reason(self):
        row = self.evaluate().as_output_row(self.base)
        row["rule_triggered_rules"] = stable_json(
            [NO_GOOD_TEMPLATE, "SECONDARY_AUDIT_CODE"]
        )
        summary = build_summary(
            [row],
            shot="shot",
            selected_paths=frozenset(),
            duplicate_status="SKIPPED_NO_GOOD_MODES",
            override_audit=OverrideAudit(),
            override_sha256="",
            fraction_tae_threshold=0.5,
            fraction_eae_threshold=0.4,
            signed_delta_eae_threshold=-0.1,
            rel_freq_tol=0.02,
        )
        self.assertEqual(
            json.loads(summary["primary_reason_counts_json"]),
            {NO_GOOD_TEMPLATE: 1},
        )

    def test_empty_manual_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manual_overrides.csv"
            result_row = self.evaluate().as_output_row(self.base)
            row = override_row({key: str(value) for key, value in result_row.items()})
            row["manual_reason"] = ""
            write_dict_csv(path, MANUAL_OVERRIDE_FIELDS, [row])
            with self.assertRaisesRegex(ValueError, "manual_reason must be nonempty"):
                load_manual_overrides(path)


class DuplicateHandlingTests(unittest.TestCase):
    def _rows_and_scorer(self, root: Path):
        shot = make_tae_shot(root, names=("egn01w.low", "egn01w.high"))
        paths = [shot / "N1" / "egn01w.low", shot / "N1" / "egn01w.high"]
        rows = [
            {
                "path": str(path),
                "mode_key": portable_mode_key(path),
                "shot": shot.name,
                "ntor": 1,
                "omega": 1.0 + 0.005 * index,
                "final_decision": "GOOD",
                "rule_decision": "REVIEW",
            }
            for index, path in enumerate(paths)
        ]
        review_path = shot / "N1" / "egn01w.review"
        write_mode(review_path, omega=1.007, ntor=1)
        rows.append(
            {
                "path": str(review_path),
                "mode_key": portable_mode_key(review_path),
                "shot": shot.name,
                "ntor": 1,
                "omega": 1.007,
                "final_decision": "REVIEW",
                "rule_decision": "REVIEW",
            }
        )
        calls: list[str] = []

        def scorer(_classifier, path):
            calls.append(path)
            mode, omega, gamma_d, ntor = load_mode_from_nova(path)
            score = 0.9 if path.endswith("high") else 0.1
            return score, mode, omega, gamma_d, ntor

        return rows, paths, calls, scorer

    def test_no_rf_retains_every_good_cluster_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, _calls, _scorer = self._rows_and_scorer(Path(temporary))
            result = deduplicate_final_good(rows, rf_model_path=None, rel_freq_tol=0.02)
        self.assertEqual(result.selected_paths, frozenset(str(path) for path in paths))
        self.assertEqual(result.status, "SKIPPED_NO_RF_CHECKPOINT")
        self.assertEqual(result.cluster_records[0]["status"], "SKIPPED_NO_RF_CHECKPOINT")

    def test_unloadable_rf_retains_every_good_cluster_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, _calls, scorer = self._rows_and_scorer(Path(temporary))

            def failing_loader(_path):
                raise ValueError("synthetic unloadable checkpoint")

            result = deduplicate_final_good(
                rows,
                rf_model_path="bad.joblib",
                rel_freq_tol=0.02,
                rf_loader=failing_loader,
                rf_scorer=scorer,
            )
        self.assertEqual(result.selected_paths, frozenset(str(path) for path in paths))
        self.assertEqual(result.status, "SKIPPED_NO_RF_CHECKPOINT")
        self.assertIn(
            "could not be loaded", result.cluster_records[0]["diagnostic_message"]
        )

    def test_rf_scores_only_final_good_and_keeps_highest_equivalent_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, calls, scorer = self._rows_and_scorer(Path(temporary))
            decisions_before = [(row["rule_decision"], row["final_decision"]) for row in rows]
            result = deduplicate_final_good(
                rows,
                rf_model_path="fake.joblib",
                rel_freq_tol=0.02,
                rf_loader=lambda _path: object(),
                rf_scorer=scorer,
            )
        self.assertEqual(set(calls), {str(path) for path in paths})
        self.assertEqual(result.selected_paths, frozenset({str(paths[1])}))
        self.assertEqual(result.cluster_records[0]["status"], "PROCESSED_RF")
        self.assertEqual(
            [(row["rule_decision"], row["final_decision"]) for row in rows],
            decisions_before,
        )

    def test_one_rf_scoring_failure_retains_whole_cluster(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, _calls, scorer = self._rows_and_scorer(Path(temporary))

            def failing_scorer(classifier, path):
                if path.endswith("high"):
                    raise RuntimeError("synthetic score failure")
                return scorer(classifier, path)

            result = deduplicate_final_good(
                rows,
                rf_model_path="fake.joblib",
                rel_freq_tol=0.02,
                rf_loader=lambda _path: object(),
                rf_scorer=failing_scorer,
            )
        self.assertEqual(result.selected_paths, frozenset(str(path) for path in paths))
        self.assertEqual(
            result.cluster_records[0]["status"], "SKIPPED_RF_SCORING_FAILED"
        )


class WorkflowOutputTests(unittest.TestCase):
    def test_complete_output_contract_summary_counts_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            out_dir = root / "out"
            first = run_shot(shot, out_dir)
            self.assertEqual(set(path.name for path in out_dir.iterdir()), REQUIRED_OUTPUTS)
            for name in REQUIRED_OUTPUTS:
                if name.endswith(".csv"):
                    with (out_dir / name).open(newline="") as handle:
                        self.assertTrue(next(csv.reader(handle)), name)
            self.assertEqual(first.summary["n_preliminary_review"], 1)
            self.assertEqual(first.summary["n_final_review"], 1)
            self.assertEqual(first.summary["n_final_good"], 0)
            self.assertTrue(first.summary["axis_artifact_gate_enabled"])
            self.assertEqual(first.summary["axis_artifact_r_ax"], 0.03)
            self.assertEqual(first.summary["axis_artifact_amplitude_min"], 0.2)
            self.assertEqual(first.summary["axis_artifact_width_max_grid"], 10.0)
            self.assertTrue(first.summary["grid_scale_spike_gate_enabled"])
            self.assertEqual(first.summary["grid_scale_spike_amplitude_min"], 0.3)
            self.assertEqual(first.summary["grid_scale_spike_width_max_grid"], 1.0)
            self.assertTrue(first.summary["continuum_crossing_gate_enabled"])
            self.assertEqual(first.summary["continuum_crossing_w_threshold"], 0.05)
            self.assertTrue(first.summary["edge_artifact_gate_enabled"])
            self.assertEqual(first.summary["edge_artifact_r_min"], 0.97)
            self.assertEqual(first.summary["edge_artifact_width_max_grid"], 10.0)
            self.assertEqual(
                json.loads(first.summary["primary_reason_counts_json"]),
                {NO_GOOD_TEMPLATE: 1},
            )
            _fields, result_rows = read_dict_csv(out_dir / "rule_results.csv")
            self.assertEqual(
                json.loads(result_rows[0]["rule_triggered_rules"]),
                [NO_GOOD_TEMPLATE],
            )
            rule_features = json.loads(result_rows[0]["rule_features"])
            self.assertEqual(
                rule_features["feature_schema_version"],
                RULE_FEATURE_SCHEMA_VERSION,
            )
            self.assertEqual(
                set(rule_features) - set(RULE_FEATURE_METADATA_NAMES),
                set(RULE_FEATURE_GROUP_NAMES),
            )
            self.assertAlmostEqual(
                rule_features["rf_standard_features"]["rad_loc"],
                float(result_rows[0]["rad_loc"]),
            )
            self.assertAlmostEqual(
                rule_features["rf_standard_features"]["rad_width"],
                float(result_rows[0]["rad_width"]),
            )

            before = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
            run_shot(shot, out_dir)
            after = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
            self.assertEqual(before, after)

    def test_run_shot_applies_configured_axis_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = np.zeros((4, 101), dtype=float)
            mode[2, 1] = 1.0
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=101,
                mode=mode,
            )
            write_datcon(shot / "N1" / "datcon1", nr=101)
            result = run_shot(
                shot,
                root / "out",
                axis_r_ax=0.03,
                axis_amplitude_min=0.8,
                axis_width_max_grid=2.0,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertEqual(result.summary["n_final_bad"], 1)
        self.assertTrue(result.summary["axis_artifact_gate_enabled"])
        self.assertEqual(result.summary["axis_artifact_amplitude_min"], 0.8)
        self.assertEqual(result.summary["axis_artifact_width_max_grid"], 2.0)
        self.assertEqual(result.final_rows[0]["rule_primary_reason"], BAD_AXIS_SPIKE)

    def test_run_shot_applies_grid_scale_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = np.zeros((4, 101), dtype=float)
            mode[2, 50] = -0.7
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=101,
                mode=mode,
            )
            write_datcon(shot / "N1" / "datcon1", nr=101)
            result = run_shot(
                shot,
                root / "out",
                grid_scale_amplitude_min=0.6,
                grid_scale_width_max_grid=1.0,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(result.summary["grid_scale_spike_gate_enabled"])
        self.assertEqual(result.summary["grid_scale_spike_amplitude_min"], 0.6)
        self.assertEqual(result.summary["grid_scale_spike_width_max_grid"], 1.0)
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_GRID_SCALE_SPIKE,
        )

    def test_run_shot_applies_cont_cross_gate_and_reports_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            radial_grid = np.linspace(0.0, 1.0, 101)
            upper = 0.9 + 0.4 * radial_grid
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=101,
            )
            write_datcon(
                shot / "N1" / "datcon1",
                nr=101,
                upper_frequency=upper,
            )
            result = run_shot(
                shot,
                root / "out",
                w_cross_threshold=0.05,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(result.summary["continuum_crossing_gate_enabled"])
        self.assertEqual(result.summary["continuum_crossing_w_threshold"], 0.05)
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_CONT_CROSS,
        )

    def test_run_shot_applies_edge_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = np.zeros((4, 101), dtype=float)
            mode[2, 96:101] = [0.2, 0.6, 1.0, 0.6, 0.2]
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=101,
                mode=mode,
            )
            write_datcon(shot / "N1" / "datcon1", nr=101)
            result = run_shot(
                shot,
                root / "out",
                edge_r_min=0.97,
                edge_width_max_grid=4.0,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(result.summary["edge_artifact_gate_enabled"])
        self.assertEqual(result.summary["edge_artifact_r_min"], 0.97)
        self.assertEqual(result.summary["edge_artifact_width_max_grid"], 4.0)
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_EDGE_SPIKE,
        )

    def test_valid_override_hash_and_stale_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            initial_out = root / "initial"
            run_shot(shot, initial_out)
            _fields, rows = read_dict_csv(initial_out / "rule_results.csv")
            override_path = root / "overrides.csv"
            write_dict_csv(override_path, MANUAL_OVERRIDE_FIELDS, [override_row(rows[0])])

            final_out = root / "final"
            result = run_shot(
                shot, final_out, manual_overrides=override_path, rf_model=None
            )
            final_row = result.final_rows[0]
            self.assertEqual(final_row["rule_decision"], "REVIEW")
            self.assertEqual(final_row["final_decision"], "GOOD")
            self.assertEqual(result.summary["n_final_good"], 1)
            self.assertEqual(result.summary["manual_override_sha256"], sha256_file(override_path))
            self.assertEqual(
                json.loads(result.summary["transition_counts_json"]),
                {"REVIEW->GOOD": 1},
            )

            mode_path = shot / "N1" / "egn01w.one"
            original_mode = mode_path.read_bytes()
            values = np.fromfile(mode_path)
            values[1] *= 1.001
            values.tofile(mode_path)
            stale_mode_out = root / "stale_mode"
            stale_mode_result = run_shot(
                shot, stale_mode_out, manual_overrides=override_path
            )
            self.assertEqual(stale_mode_result.final_rows[0]["final_decision"], "REVIEW")
            self.assertEqual(stale_mode_result.summary["n_stale_overrides"], 1)

            mode_path.write_bytes(original_mode)
            datcon = shot / "N1" / "datcon1"
            datcon.write_text(datcon.read_text() + "\n")
            stale_datcon_out = root / "stale_datcon"
            stale_datcon_result = run_shot(
                shot, stale_datcon_out, manual_overrides=override_path
            )
            self.assertEqual(
                stale_datcon_result.final_rows[0]["final_decision"], "REVIEW"
            )
            self.assertEqual(stale_datcon_result.summary["n_stale_overrides"], 1)

    def test_adjudication_scope_defaults_to_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            rows = []
            for index, decision in enumerate(("GOOD", "BAD", "REVIEW", "INVALID")):
                mode = shot / "N1" / f"egn01w.scope{index}"
                row = {field: "" for field in RULE_OUTPUT_FIELDS}
                row.update(
                    {
                        "path": str(mode),
                        "mode_key": portable_mode_key(mode),
                        "input_fingerprint": "a" * 64,
                        "ntor": "1",
                        "omega": "1.0",
                        "rule_decision": decision,
                    }
                )
                rows.append(row)
            source = root / "classifications.csv"
            write_dict_csv(source, RULE_OUTPUT_FIELDS, rows)
            review = adjudication_candidate_rows(
                str(source), mode_dir=shot, data_dir=None, scope="review"
            )
            all_rows = adjudication_candidate_rows(
                str(source), mode_dir=shot, data_dir=None, scope="all"
            )
        self.assertEqual([row["rule_decision"] for row in review], ["REVIEW"])
        self.assertEqual(
            sorted(row["rule_decision"] for row in all_rows), ["BAD", "GOOD", "REVIEW"]
        )


class SkillStructureTests(unittest.TestCase):
    def test_renamed_and_new_skills_have_matching_metadata_and_paths(self):
        old = REPO_ROOT / ".agents/skills/label-tae-like-modes"
        visual = REPO_ROOT / ".agents/skills/visual-tae-rule-development"
        sorter = REPO_ROOT / ".agents/skills/sort-tae-like-modes"
        self.assertFalse(old.exists())
        for skill, name in (
            (visual, "visual-tae-rule-development"),
            (sorter, "sort-tae-like-modes"),
        ):
            skill_text = (skill / "SKILL.md").read_text()
            metadata = (skill / "agents/openai.yaml").read_text()
            self.assertIn(f"name: {name}", skill_text)
            self.assertIn(f"${name}", metadata)
        for script in (
            "prepare_blind_manifest.py",
            "render_blind_diagnostics.py",
            "seal_review.py",
            "compare_reviews.py",
        ):
            self.assertTrue((visual / "scripts" / script).is_file())
        for script in (
            "make_tae_like_list.py",
            "tae_rule_engine.py",
            "sort_shot_rules.py",
            "label_modes_fast.py",
        ):
            self.assertTrue((REPO_ROOT / "scripts" / script).is_file())

    def test_quick_validate_passes_for_both_skills(self):
        configured_root = os.environ.get("CODEX_HOME")
        skill_root = Path(configured_root) if configured_root else Path.home() / ".codex"
        validator = skill_root / "skills/.system/skill-creator/scripts/quick_validate.py"
        if not validator.is_file():
            self.skipTest(f"skill validator is unavailable: {validator}")
        for skill_name in ("visual-tae-rule-development", "sort-tae-like-modes"):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    str(REPO_ROOT / ".agents/skills" / skill_name),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
