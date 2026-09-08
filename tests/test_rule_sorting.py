import csv
import contextlib
import io
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
from sort_shot_mixed import (  # noqa: E402
    DEFAULT_METHOD,
    DEFAULT_RULE_CONFIG,
    RF_CNN_METHOD,
    RULES_METHOD,
    classify_gap_region as mixed_classify_gap_region,
    parse_args as parse_mixed_args,
    run_rules_method,
)
from sort_shot_rules import (  # noqa: E402
    OverrideAudit,
    RULE_SURVIVOR_POLICY_ACCEPT,
    RULE_SURVIVOR_POLICY_REVIEW,
    apply_manual_overrides,
    apply_rule_survivor_policy,
    build_summary,
    deduplicate_final_good,
    load_manual_overrides,
    parse_args,
    run_shot,
)
from tae_rule_config import (  # noqa: E402
    PRODUCTION_RULE_CONFIG_NAME,
    RULE_CONFIG_SCHEMA_VERSION,
    load_rule_run_configuration,
)
from tae_rule_engine import (  # noqa: E402
    BAD_AXIS_SPIKE,
    BAD_CONT_CROSS,
    BAD_CONT_CROSS_WINDOW,
    BAD_EDGE_SPIKE,
    BAD_GRID_SCALE_PACKET,
    BAD_GRID_SCALE_SPIKE,
    BAD_INTERIOR_HARMONIC_INCOHERENCE,
    BAD_INTERIOR_UNRESOLVED_ENVELOPE,
    BAD_NEAR_AXIS_GRID_OSCILLATION,
    NO_GOOD_TEMPLATE,
    RULE_FEATURE_EXTRACTION_FAILED,
    RULE_FEATURE_GROUP_NAMES,
    RULE_FEATURE_METADATA_NAMES,
    RULE_FEATURE_NAMES,
    RULE_FEATURE_SCHEMA_VERSION,
    RULE_FEATURE_SOURCE_SCHEMA_VERSION,
    RULESET_VERSION,
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
    extract_axis_artifact_features,
    extract_continuum_crossing_window_features,
    extract_edge_artifact_features,
    extract_grid_scale_packet_features,
    extract_grid_scale_spike_features,
    extract_interior_harmonic_incoherence_features,
    extract_near_axis_grid_oscillation_features,
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
    "resolution_warnings.csv",
    "resolution_warnings.txt",
}

CROSS_WINDOW_FEATURE_NAMES = {
    "cross_window_candidate_found",
    "cross_window_half_width_grid",
    "cross_window_half_width_r",
    "cross_window_A_max",
    "cross_window_A_harmonic_index",
    "cross_window_A_sample_r",
    "cross_window_A_crossing_boundary",
    "cross_window_A_crossing_r",
    "cross_window_A_distance_grid",
    "cross_window_A_neighbor_rms",
    "cross_window_A_neighbor_count",
    "cross_window_A_neighbor_stencil_complete",
    "cross_window_W_max",
    "cross_window_W_sample_r",
    "cross_window_W_crossing_boundary",
    "cross_window_W_crossing_r",
    "cross_window_W_distance_grid",
}

INTERIOR_HARMONIC_INCOHERENCE_FEATURE_NAMES = {
    "core_r_max",
    "active_core_energy_fraction_min",
    "max_lag_grid",
    "score_threshold",
    "calibrated_n_radial",
    "resolution_eligible",
    "candidate_found",
    "core_radial_sample_count",
    "core_positive_energy_sample_count",
    "core_adjacent_radial_pair_count",
    "active_core_harmonic_count",
    "active_adjacent_harmonic_pair_count",
    "core_energy_fraction",
    "core_js_divergence",
    "core_effective_harmonic_count_wmean",
    "effective_harmonic_count_threshold",
    "global_effective_harmonic_count_wmean",
    "global_energy_fraction_above_effective_harmonic_count_threshold",
    "core_energy_fraction_above_effective_harmonic_count_threshold",
    "total_energy_fraction_in_core_above_effective_harmonic_count_threshold",
    "core_adjacent_harmonic_coherence",
    "incoherence_score",
}

NEAR_AXIS_GRID_OSCILLATION_FEATURE_NAMES = {
    "candidate_found",
    "peak_r_max_exclusive",
    "amplitude_min",
    "min_consecutive_sign_flips",
    "step_l2_min",
    "qualifying_run_count",
    "step_l2_qualified_run_count",
    "near_axis_peak",
    "near_axis_peak_signed_amplitude",
    "near_axis_peak_harmonic_index",
    "near_axis_peak_r",
    "near_axis_peak_amplitude_qualified",
    "selected_run_harmonic_index",
    "selected_run_start_index",
    "selected_run_end_index",
    "selected_run_start_r",
    "selected_run_end_r",
    "selected_run_peak",
    "selected_run_peak_signed_amplitude",
    "selected_run_peak_r",
    "selected_run_consecutive_sign_flip_count",
    "selected_run_step_l2",
    "selected_run_total_variation",
    "selected_run_max_step",
    "selected_run_mean_abs_step",
    "near_axis_peak_and_run_same_harmonic",
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


def narrow_total_energy_mode(
    *, nr: int = 65, peak_index: int = 32
) -> np.ndarray:
    """Return a smooth harmonic whose total-W FWHM is exactly two intervals."""
    mode = np.zeros((4, nr), dtype=float)
    mode[1, peak_index - 1 : peak_index + 2] = [np.sqrt(0.5), 1.0, np.sqrt(0.5)]
    return mode


def broadband_incoherent_mode(*, nr: int = 201, n_harmonics: int = 8) -> np.ndarray:
    """Return deterministic simultaneous broadband activity inside r <= 0.5."""
    radial_grid = np.linspace(0.0, 1.0, nr)
    n_core = int(np.count_nonzero(radial_grid <= 0.5))
    generator = np.random.default_rng(1234)
    mode = np.zeros((n_harmonics, nr), dtype=float)
    mode[:, :n_core] = generator.normal(size=(n_harmonics, n_core))
    mode /= np.max(np.abs(mode))
    return mode


def evaluate_incoherence_only(
    row: dict,
    mode: np.ndarray,
    *,
    incoherence_config: InteriorHarmonicIncoherenceConfig | None = None,
    interior_config: InteriorUnresolvedEnvelopeConfig | None = None,
):
    """Evaluate a synthetic mode with every earlier rejection gate disabled."""
    n_radial = mode.shape[1]
    return evaluate_mode(
        row,
        mode=mode,
        low2=np.full(n_radial, 0.5**2),
        high2=np.full(n_radial, 1.5**2),
        axis_artifact_config=AxisArtifactConfig(
            axis_amplitude_min=None,
            axis_width_max_grid=None,
        ),
        grid_scale_spike_config=GridScaleSpikeConfig(
            amplitude_min=None,
            width_max_grid=None,
        ),
        grid_scale_packet_config=GridScalePacketConfig(amplitude_min=None),
        near_axis_grid_oscillation_config=NearAxisGridOscillationConfig(
            amplitude_min=None
        ),
        continuum_crossing_config=ContinuumCrossingConfig(
            w_cross_threshold=None
        ),
        continuum_crossing_window_config=ContinuumCrossingWindowConfig(
            amplitude_min=None,
            w_min=None,
        ),
        edge_artifact_config=EdgeArtifactConfig(edge_width_max_grid=None),
        interior_unresolved_envelope_config=(
            interior_config or InteriorUnresolvedEnvelopeConfig(width_max_grid=None)
        ),
        interior_harmonic_incoherence_config=incoherence_config,
    )


def evaluate_near_axis_oscillation_only(
    row: dict,
    mode: np.ndarray,
    *,
    oscillation_config: NearAxisGridOscillationConfig | None = None,
):
    """Evaluate one mode with only the near-axis oscillation gate enabled."""
    n_radial = mode.shape[1]
    return evaluate_mode(
        row,
        mode=mode,
        low2=np.full(n_radial, 0.5**2),
        high2=np.full(n_radial, 1.5**2),
        axis_artifact_config=AxisArtifactConfig(
            axis_amplitude_min=None,
            axis_width_max_grid=None,
        ),
        grid_scale_spike_config=GridScaleSpikeConfig(
            amplitude_min=None,
            width_max_grid=None,
        ),
        grid_scale_packet_config=GridScalePacketConfig(amplitude_min=None),
        near_axis_grid_oscillation_config=oscillation_config,
        continuum_crossing_config=ContinuumCrossingConfig(
            w_cross_threshold=None
        ),
        continuum_crossing_window_config=ContinuumCrossingWindowConfig(
            amplitude_min=None,
            w_min=None,
        ),
        edge_artifact_config=EdgeArtifactConfig(edge_width_max_grid=None),
        interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
            width_max_grid=None
        ),
        interior_harmonic_incoherence_config=InteriorHarmonicIncoherenceConfig(
            score_threshold=None
        ),
    )


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
        packet_config=None,
        near_axis_oscillation_config=None,
        crossing_config=None,
        cross_window_config=None,
        edge_config=None,
        interior_config=None,
    ):
        return evaluate_mode(
            self.base if row is None else row,
            mode=self.mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=axis_config,
            grid_scale_spike_config=grid_config,
            grid_scale_packet_config=packet_config,
            near_axis_grid_oscillation_config=near_axis_oscillation_config,
            continuum_crossing_config=crossing_config,
            continuum_crossing_window_config=cross_window_config,
            edge_artifact_config=edge_config,
            interior_unresolved_envelope_config=interior_config,
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
        self.assertEqual(RULE_FEATURE_SCHEMA_VERSION, "tae-rule-features-grouped-v18")
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
            set(EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES)
            | CROSS_WINDOW_FEATURE_NAMES
            | {"continuum_crossing_tail"},
        )
        self.assertEqual(
            set(features["extremum_features"]),
            {"match_found", *EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES},
        )
        self.assertTrue(features["extremum_features"]["match_found"])
        interior_features = features["resolution_features"][
            "interior_unresolved_envelope"
        ]
        self.assertEqual(interior_features["peak_r_max"], 0.5)
        self.assertEqual(interior_features["width_max_grid"], 2.0)
        self.assertEqual(interior_features["extremum_r_min"], 0.03)
        self.assertEqual(interior_features["extremum_r_max"], 0.5)
        self.assertEqual(interior_features["ext_dr_max"], 0.02)
        self.assertEqual(interior_features["ext_df_gap_min"], 0.0)
        self.assertEqual(interior_features["ext_df_gap_max"], 0.04)
        self.assertFalse(interior_features["candidate_found"])
        self.assertFalse(interior_features["extremum_exception_applied"])
        grid_features = features["numerical_structure_features"][
            "grid_scale_spike"
        ]
        self.assertFalse(grid_features["grid_scale_candidate_found"])
        self.assertIsNone(
            grid_features["grid_scale_candidate_width_limit_grid"]
        )
        self.assertEqual(grid_features["grid_scale_width_max_grid"], 1.0)
        self.assertEqual(grid_features["grid_scale_high_r_cutoff_r"], 0.7)
        self.assertEqual(
            grid_features["grid_scale_high_r_width_max_grid"], 0.75
        )
        packet_features = features["numerical_structure_features"][
            "grid_scale_packet"
        ]
        self.assertFalse(
            packet_features["grid_scale_packet_candidate_found"]
        )
        self.assertEqual(
            packet_features["grid_scale_packet_amplitude_min"], 0.3
        )
        self.assertEqual(packet_features["grid_scale_packet_step_min"], 0.2)
        self.assertEqual(
            packet_features["grid_scale_packet_min_large_turns"], 3
        )
        self.assertEqual(
            packet_features["grid_scale_packet_window_span_grid"], 4
        )
        self.assertEqual(
            packet_features["grid_scale_packet_peak_r_max"], 0.5
        )
        self.assertEqual(
            packet_features[
                "grid_scale_packet_radius_qualified_window_count"
            ],
            0,
        )
        oscillation_features = features["numerical_structure_features"][
            "near_axis_grid_oscillation"
        ]
        self.assertEqual(
            set(oscillation_features),
            NEAR_AXIS_GRID_OSCILLATION_FEATURE_NAMES,
        )
        self.assertFalse(oscillation_features["candidate_found"])
        self.assertEqual(oscillation_features["peak_r_max_exclusive"], 0.1)
        self.assertEqual(oscillation_features["amplitude_min"], 0.1)
        self.assertEqual(
            oscillation_features["min_consecutive_sign_flips"], 4
        )
        self.assertEqual(oscillation_features["step_l2_min"], 0.3)
        self.assertEqual(oscillation_features["qualifying_run_count"], 0)
        incoherence_features = features["numerical_structure_features"][
            "interior_harmonic_incoherence"
        ]
        self.assertEqual(
            set(incoherence_features),
            INTERIOR_HARMONIC_INCOHERENCE_FEATURE_NAMES,
        )
        self.assertEqual(incoherence_features["core_r_max"], 0.5)
        self.assertEqual(
            incoherence_features["active_core_energy_fraction_min"], 0.005
        )
        self.assertEqual(incoherence_features["max_lag_grid"], 5)
        self.assertEqual(incoherence_features["score_threshold"], 0.1)
        self.assertEqual(incoherence_features["calibrated_n_radial"], 201)
        self.assertTrue(incoherence_features["resolution_eligible"])
        axis_features = features["boundary_features"]["axis_artifact"]
        self.assertEqual(axis_features["r_ax"], 0.03)
        self.assertFalse(axis_features["axis_candidate_found"])
        self.assertEqual(axis_features["axis_candidate_amplitude_min"], 0.2)
        self.assertEqual(axis_features["axis_candidate_width_limit_grid"], 10.0)
        self.assertEqual(axis_features["axis_local_peak_count"], 0)
        self.assertEqual(axis_features["axis_amplitude_qualified_peak_count"], 0)
        self.assertEqual(axis_features["axis_width_qualified_peak_count"], 0)
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
        interior_features = features["resolution_features"][
            "interior_unresolved_envelope"
        ]
        self.assertIsNone(interior_features["candidate_found"])
        self.assertIsNone(interior_features["energy_peak_r"])
        self.assertIsNone(interior_features["extremum_match_found"])
        self.assertIsNone(interior_features["extremum_exception_applied"])
        incoherence_features = features["numerical_structure_features"][
            "interior_harmonic_incoherence"
        ]
        self.assertEqual(
            set(incoherence_features),
            INTERIOR_HARMONIC_INCOHERENCE_FEATURE_NAMES,
        )
        self.assertIsNone(incoherence_features["resolution_eligible"])
        self.assertIsNone(incoherence_features["candidate_found"])
        self.assertIsNone(incoherence_features["incoherence_score"])
        self.assertEqual(
            incoherence_features["effective_harmonic_count_threshold"], 3.0
        )
        self.assertIsNone(
            incoherence_features["global_effective_harmonic_count_wmean"]
        )
        self.assertIsNone(
            incoherence_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
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
        self.assertTrue(axis_features["axis_candidate_found"])
        self.assertEqual(axis_features["axis_local_peak_count"], 1)
        self.assertEqual(axis_features["axis_amplitude_qualified_peak_count"], 1)
        self.assertEqual(axis_features["axis_width_qualified_peak_count"], 1)
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

    def test_axis_gate_checks_local_peak_masked_by_larger_rising_flank(self):
        mode = np.zeros_like(self.mode)
        mode[1, :9] = [
            0.0,
            0.7867,
            0.1642,
            -0.4318,
            -0.6916,
            -0.8940,
            -0.9541,
            -0.9707,
            -0.9111,
        ]
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
        )
        axis_features = result.features["boundary_features"]["axis_artifact"]

        self.assertEqual(result.decision, "BAD")
        self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)
        self.assertTrue(axis_features["axis_candidate_found"])
        self.assertEqual(axis_features["axis_local_peak_count"], 1)
        self.assertEqual(axis_features["axis_amplitude_qualified_peak_count"], 1)
        self.assertAlmostEqual(axis_features["axis_peak_r"], 0.005)
        self.assertAlmostEqual(axis_features["axis_peak"], 0.7867)
        self.assertTrue(axis_features["axis_peak_is_local_max"])
        self.assertLess(axis_features["axis_halfmax_width_grid"], 2.0)

    def test_axis_gate_checks_narrower_peak_when_stronger_peak_is_too_broad(self):
        mode = np.zeros_like(self.mode)
        mode[1, 2] = 0.6
        mode[2, :9] = [0.0, 0.4, 0.7, 0.9, 1.0, 0.9, 0.7, 0.4, 0.0]
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            axis_artifact_config=AxisArtifactConfig(
                axis_amplitude_min=0.5,
                axis_width_max_grid=2.0,
            ),
        )
        axis_features = result.features["boundary_features"]["axis_artifact"]

        self.assertEqual(result.decision, "BAD")
        self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)
        self.assertTrue(axis_features["axis_candidate_found"])
        self.assertEqual(axis_features["axis_local_peak_count"], 2)
        self.assertEqual(axis_features["axis_amplitude_qualified_peak_count"], 2)
        self.assertEqual(axis_features["axis_width_qualified_peak_count"], 1)
        self.assertEqual(axis_features["axis_peak_harmonic_index"], 1)
        self.assertAlmostEqual(axis_features["axis_peak_r"], 0.01)
        self.assertAlmostEqual(axis_features["axis_peak"], 0.6)
        self.assertAlmostEqual(axis_features["axis_halfmax_width_grid"], 1.0)

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
        self.assertEqual(
            features["grid_scale_candidate_width_limit_grid"], 1.0
        )

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
        packet_features = result.features["numerical_structure_features"][
            "grid_scale_packet"
        ]
        self.assertFalse(packet_features["grid_scale_packet_candidate_found"])
        self.assertEqual(
            packet_features["grid_scale_packet_turn_qualified_window_count"],
            0,
        )

    def test_grid_scale_spike_uses_relaxed_width_above_high_r_cutoff(self):
        def mode_with_width_between_limits(peak_index):
            mode = np.zeros_like(self.mode)
            mode[1, peak_index - 1 : peak_index + 2] = [-0.175, 0.7, -0.175]
            return mode

        low_r_mode = mode_with_width_between_limits(120)
        low_r_features = extract_grid_scale_spike_features(low_r_mode)
        self.assertTrue(low_r_features["grid_scale_candidate_found"])
        self.assertAlmostEqual(low_r_features["grid_scale_peak_r"], 0.6)
        self.assertGreater(
            low_r_features["grid_scale_halfmax_width_grid"], 0.75
        )
        self.assertLessEqual(
            low_r_features["grid_scale_halfmax_width_grid"], 1.0
        )
        self.assertEqual(
            low_r_features["grid_scale_candidate_width_limit_grid"], 1.0
        )

        cutoff_mode = mode_with_width_between_limits(140)
        cutoff_features = extract_grid_scale_spike_features(cutoff_mode)
        self.assertTrue(cutoff_features["grid_scale_candidate_found"])
        self.assertAlmostEqual(cutoff_features["grid_scale_peak_r"], 0.7)
        self.assertEqual(
            cutoff_features["grid_scale_candidate_width_limit_grid"], 1.0
        )

        high_r_mode = mode_with_width_between_limits(160)
        high_r_features = extract_grid_scale_spike_features(high_r_mode)
        self.assertTrue(high_r_features["grid_scale_candidate_found"])
        self.assertLess(high_r_features["grid_scale_peak"], 0.3)
        self.assertEqual(
            high_r_features["grid_scale_candidate_width_limit_grid"], 0.75
        )
        high_r_result = evaluate_mode(
            self.base,
            mode=high_r_mode,
            low2=self.low2,
            high2=self.high2,
        )
        self.assertEqual(high_r_result.decision, "REVIEW")

        unresolved_high_r_mode = np.zeros_like(self.mode)
        unresolved_high_r_mode[1, 160] = 0.7
        unresolved_high_r_mode[1, 161] = -0.7
        unresolved_high_r_features = extract_grid_scale_spike_features(
            unresolved_high_r_mode
        )
        self.assertTrue(
            unresolved_high_r_features["grid_scale_candidate_found"]
        )
        self.assertAlmostEqual(
            unresolved_high_r_features["grid_scale_halfmax_width_grid"],
            0.75,
        )
        self.assertEqual(
            unresolved_high_r_features[
                "grid_scale_candidate_width_limit_grid"
            ],
            0.75,
        )

    def test_grid_scale_gate_requires_both_amplitude_and_width(self):
        low_amplitude = np.zeros_like(self.mode)
        low_amplitude[1, 80] = 0.29
        low_result = evaluate_mode(
            self.base,
            mode=low_amplitude,
            low2=self.low2,
            high2=self.high2,
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None,
            ),
        )
        self.assertEqual(low_result.decision, "REVIEW")

        resolved = np.zeros_like(self.mode)
        resolved[1, 78:83] = [0.2, 0.4, 0.6, 0.4, 0.2]
        resolved_result = evaluate_mode(
            self.base,
            mode=resolved,
            low2=self.low2,
            high2=self.high2,
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
            ),
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None,
            ),
        )
        self.assertEqual(resolved_result.decision, "REVIEW")
        features = extract_grid_scale_spike_features(
            resolved,
            width_max_grid=1.0,
        )
        self.assertFalse(features["grid_scale_candidate_found"])

    def test_grid_scale_packet_gate_catches_repeated_same_sign_excursions(self):
        mode = np.zeros_like(self.mode)
        mode[1, 93:98] = [0.571, 0.329, 1.0, 0.014, 0.329]
        features = extract_grid_scale_packet_features(mode)

        self.assertTrue(features["grid_scale_packet_candidate_found"])
        self.assertEqual(features["grid_scale_packet_peak_harmonic_index"], 1)
        self.assertAlmostEqual(features["grid_scale_packet_peak_r"], 0.475)
        self.assertEqual(features["grid_scale_packet_peak_r_max"], 0.5)
        self.assertGreaterEqual(
            features["grid_scale_packet_radius_qualified_window_count"], 1
        )
        self.assertEqual(features["grid_scale_packet_large_step_count"], 4)
        self.assertEqual(features["grid_scale_packet_large_turn_count"], 3)
        self.assertEqual(
            features["grid_scale_packet_direction_change_count"], 3
        )
        self.assertEqual(features["grid_scale_packet_sign_change_count"], 0)
        self.assertEqual(
            features["grid_scale_packet_window_values"],
            [0.0, 0.571, 0.329, 1.0, 0.014],
        )

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            grid_scale_spike_config=GridScaleSpikeConfig(
                amplitude_min=None,
            ),
        )
        self.assertEqual(result.primary_reason, BAD_GRID_SCALE_PACKET)
        self.assertEqual(result.triggered_rules, (BAD_GRID_SCALE_PACKET,))

    def test_grid_scale_packet_gate_does_not_reject_one_isolated_spike(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80] = 1.0
        features = extract_grid_scale_packet_features(mode)

        self.assertFalse(features["grid_scale_packet_candidate_found"])
        self.assertEqual(
            features["grid_scale_packet_turn_qualified_window_count"], 0
        )

    def test_grid_scale_packet_gate_does_not_reject_three_large_steps_without_three_turns(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80:85] = [0.0, 0.3, 0.0, 0.3, 0.29]
        features = extract_grid_scale_packet_features(mode)

        self.assertFalse(features["grid_scale_packet_candidate_found"])
        self.assertEqual(
            features["grid_scale_packet_turn_qualified_window_count"], 0
        )

    def test_grid_scale_packet_thresholds_are_inclusive(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80:85] = [0.0, 0.2, 0.0, 0.3, 0.0]
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            grid_scale_spike_config=GridScaleSpikeConfig(
                amplitude_min=None,
            ),
        )
        features = result.features["numerical_structure_features"][
            "grid_scale_packet"
        ]

        self.assertEqual(result.primary_reason, BAD_GRID_SCALE_PACKET)
        self.assertAlmostEqual(features["grid_scale_packet_peak"], 0.3)
        self.assertEqual(features["grid_scale_packet_large_step_count"], 4)
        self.assertEqual(features["grid_scale_packet_large_turn_count"], 3)

    def test_grid_scale_packet_rejects_impossible_turn_configuration(self):
        with self.assertRaisesRegex(ValueError, "min_large_turns"):
            GridScalePacketConfig(
                min_large_turns=4,
                window_span_grid=4,
            )

    def test_grid_scale_packet_rejects_invalid_peak_radius_cutoff(self):
        with self.assertRaisesRegex(ValueError, "peak_r_max"):
            GridScalePacketConfig(peak_r_max=1.01)

    def test_null_packet_amplitude_disables_gate_but_keeps_measurement(self):
        mode = np.zeros_like(self.mode)
        mode[1, 80:85] = [0.0, 0.4, 0.0, 0.8, 0.0]
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=self.high2,
            grid_scale_spike_config=GridScaleSpikeConfig(
                amplitude_min=None,
            ),
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
            ),
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None,
            ),
        )
        features = result.features["numerical_structure_features"][
            "grid_scale_packet"
        ]

        self.assertEqual(result.decision, "REVIEW")
        self.assertTrue(features["grid_scale_packet_candidate_found"])
        self.assertAlmostEqual(features["grid_scale_packet_peak"], 0.8)

    def test_grid_scale_packet_peak_radius_cutoff_is_inclusive(self):
        mode = np.zeros_like(self.mode)
        mode[2, 97:102] = [0.0, 0.4, 0.0, 0.8, 0.0]
        features = extract_grid_scale_packet_features(mode)

        self.assertTrue(features["grid_scale_packet_candidate_found"])
        self.assertAlmostEqual(features["grid_scale_packet_peak_r"], 0.5)

    def test_grid_scale_packet_gate_excludes_high_radius_windows_by_default(self):
        mode = np.zeros_like(self.mode)
        mode[2, 196:201] = [0.0, 0.4, 0.0, 0.8, 0.0]
        features = extract_grid_scale_packet_features(mode)

        self.assertFalse(features["grid_scale_packet_candidate_found"])
        self.assertGreater(
            features["grid_scale_packet_turn_qualified_window_count"], 0
        )
        self.assertEqual(
            features["grid_scale_packet_radius_qualified_window_count"], 0
        )

        unrestricted = extract_grid_scale_packet_features(
            mode,
            peak_r_max=None,
        )
        self.assertTrue(unrestricted["grid_scale_packet_candidate_found"])
        self.assertAlmostEqual(
            unrestricted["grid_scale_packet_window_start_r"], 0.98
        )
        self.assertAlmostEqual(
            unrestricted["grid_scale_packet_window_end_r"], 1.0
        )

    def test_near_axis_grid_oscillation_uses_mode_peak_and_strongest_single_run(self):
        mode = np.zeros_like(self.mode)
        mode[0, 19] = 0.2
        mode[2, 2:7] = [0.075, -0.075, 0.075, -0.075, 0.075]
        result = evaluate_near_axis_oscillation_only(self.base, mode)
        features = result.features["numerical_structure_features"][
            "near_axis_grid_oscillation"
        ]

        self.assertEqual(result.primary_reason, BAD_NEAR_AXIS_GRID_OSCILLATION)
        self.assertEqual(
            result.triggered_rules, (BAD_NEAR_AXIS_GRID_OSCILLATION,)
        )
        self.assertTrue(features["candidate_found"])
        self.assertAlmostEqual(features["near_axis_peak"], 0.2)
        self.assertEqual(features["near_axis_peak_harmonic_index"], 0)
        self.assertAlmostEqual(features["near_axis_peak_r"], 0.095)
        self.assertEqual(features["selected_run_harmonic_index"], 2)
        self.assertEqual(
            features["selected_run_consecutive_sign_flip_count"], 4
        )
        self.assertAlmostEqual(features["selected_run_step_l2"], 0.3)
        self.assertFalse(features["near_axis_peak_and_run_same_harmonic"])

    def test_near_axis_grid_oscillation_does_not_bridge_one_missing_flip(self):
        mode = np.zeros_like(self.mode)
        mode[0, 5] = 0.2
        mode[1, 2:8] = [0.11, -0.10, 0.09, 0.08, -0.08, 0.07]
        features = extract_near_axis_grid_oscillation_features(mode)

        self.assertFalse(features["candidate_found"])
        self.assertEqual(features["qualifying_run_count"], 0)
        self.assertIsNone(features["selected_run_step_l2"])

    def test_near_axis_grid_oscillation_does_not_sum_harmonic_runs(self):
        mode = np.zeros_like(self.mode)
        mode[0, 19] = 0.2
        values = [0.0625, -0.0625, 0.0625, -0.0625, 0.0625]
        mode[1, 2:7] = values
        mode[2, 8:13] = values
        features = extract_near_axis_grid_oscillation_features(mode)

        self.assertFalse(features["candidate_found"])
        self.assertEqual(features["qualifying_run_count"], 2)
        self.assertEqual(features["step_l2_qualified_run_count"], 0)
        self.assertAlmostEqual(features["selected_run_step_l2"], 0.25)

    def test_near_axis_grid_oscillation_radius_cutoff_is_strict(self):
        mode = np.zeros_like(self.mode)
        mode[0, 20] = 0.2
        mode[1, 2:7] = [0.075, -0.075, 0.075, -0.075, 0.075]
        features = extract_near_axis_grid_oscillation_features(mode)

        self.assertFalse(features["candidate_found"])
        self.assertAlmostEqual(features["near_axis_peak"], 0.075)
        self.assertAlmostEqual(features["near_axis_peak_r"], 0.01)
        self.assertFalse(features["near_axis_peak_amplitude_qualified"])

        run_boundary_mode = np.zeros_like(self.mode)
        run_boundary_mode[0, 10] = 0.2
        run_boundary_mode[1, 16:21] = [0.06, -0.07, 0.08, -0.09, 0.10]
        run_boundary_features = extract_near_axis_grid_oscillation_features(
            run_boundary_mode
        )
        self.assertFalse(run_boundary_features["candidate_found"])
        self.assertEqual(run_boundary_features["qualifying_run_count"], 0)

    def test_near_axis_grid_oscillation_disabled_keeps_evidence(self):
        mode = np.zeros_like(self.mode)
        mode[0, 4] = 0.2
        mode[1, 2:7] = [0.075, -0.075, 0.075, -0.075, 0.075]
        result = evaluate_near_axis_oscillation_only(
            self.base,
            mode,
            oscillation_config=NearAxisGridOscillationConfig(
                amplitude_min=None
            ),
        )
        features = result.features["numerical_structure_features"][
            "near_axis_grid_oscillation"
        ]

        self.assertEqual(result.decision, "REVIEW")
        self.assertFalse(features["candidate_found"])
        self.assertIsNone(features["near_axis_peak_amplitude_qualified"])
        self.assertAlmostEqual(features["selected_run_step_l2"], 0.3)

    def test_near_axis_grid_oscillation_validates_configuration(self):
        with self.assertRaisesRegex(ValueError, "peak_r_max"):
            NearAxisGridOscillationConfig(peak_r_max=0.0)
        with self.assertRaisesRegex(ValueError, "min_consecutive_sign_flips"):
            NearAxisGridOscillationConfig(min_consecutive_sign_flips=0)
        with self.assertRaisesRegex(ValueError, "step_l2_min"):
            NearAxisGridOscillationConfig(step_l2_min=-0.1)

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
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None,
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
        disabled_cross_window = ContinuumCrossingWindowConfig(
            half_width_grid=2,
            amplitude_min=None,
            w_min=None,
        )
        disabled_interior = InteriorUnresolvedEnvelopeConfig(
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
            continuum_crossing_window_config=disabled_cross_window,
            interior_unresolved_envelope_config=disabled_interior,
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
            continuum_crossing_window_config=disabled_cross_window,
            interior_unresolved_envelope_config=disabled_interior,
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
            continuum_crossing_window_config=disabled_cross_window,
            interior_unresolved_envelope_config=disabled_interior,
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
            continuum_crossing_window_config=disabled_cross_window,
            interior_unresolved_envelope_config=disabled_interior,
        )
        self.assertEqual(no_cross_result.decision, "REVIEW")
        self.assertEqual(
            no_cross_result.features["crossing_features"]["n_cross"],
            0.0,
        )

    def test_cross_window_extractor_uses_inclusive_radial_grid_distance(self):
        mode = np.zeros((2, 11), dtype=float)
        mode[0, 8] = 1.0
        mode[0, 4] = 0.1
        mode[1, 5] = -0.3
        crossing_records = [
            {
                "boundary": "high",
                "r_cross": 0.35,
                "W_peak": 0.0,
                "shear_weighted": 0.0,
            }
        ]

        one_grid = extract_continuum_crossing_window_features(
            mode,
            crossing_records,
            half_width_grid=1,
        )
        two_grid = extract_continuum_crossing_window_features(
            mode,
            crossing_records,
            half_width_grid=2,
        )

        self.assertTrue(two_grid["cross_window_candidate_found"])
        self.assertAlmostEqual(one_grid["cross_window_A_max"], 0.1)
        self.assertAlmostEqual(two_grid["cross_window_A_max"], 0.3)
        self.assertEqual(two_grid["cross_window_A_harmonic_index"], 1)
        self.assertAlmostEqual(two_grid["cross_window_A_sample_r"], 0.5)
        self.assertEqual(two_grid["cross_window_A_crossing_boundary"], "high")
        self.assertAlmostEqual(two_grid["cross_window_A_crossing_r"], 0.35)
        self.assertAlmostEqual(two_grid["cross_window_A_distance_grid"], 1.5)
        self.assertAlmostEqual(two_grid["cross_window_A_neighbor_rms"], 0.3)
        self.assertEqual(two_grid["cross_window_A_neighbor_count"], 4)
        self.assertTrue(
            two_grid["cross_window_A_neighbor_stencil_complete"]
        )
        self.assertAlmostEqual(two_grid["cross_window_W_max"], 0.09)
        self.assertAlmostEqual(two_grid["cross_window_W_sample_r"], 0.5)

        no_crossing = extract_continuum_crossing_window_features(
            mode,
            [],
            half_width_grid=2,
        )
        self.assertFalse(no_crossing["cross_window_candidate_found"])
        self.assertIsNone(no_crossing["cross_window_A_max"])
        self.assertIsNone(no_crossing["cross_window_A_neighbor_rms"])
        self.assertIsNone(no_crossing["cross_window_A_neighbor_count"])
        self.assertIsNone(
            no_crossing["cross_window_A_neighbor_stencil_complete"]
        )
        self.assertIsNone(no_crossing["cross_window_W_max"])
        self.assertAlmostEqual(no_crossing["cross_window_half_width_r"], 0.2)

        boundary_mode = np.zeros((1, 11), dtype=float)
        boundary_mode[0, 0] = 0.5
        boundary = extract_continuum_crossing_window_features(
            boundary_mode,
            [
                {
                    "boundary": "low",
                    "r_cross": 0.0,
                    "W_peak": 0.0,
                    "shear_weighted": 0.0,
                }
            ],
            half_width_grid=0,
        )
        self.assertEqual(boundary["cross_window_A_neighbor_count"], 2)
        self.assertFalse(
            boundary["cross_window_A_neighbor_stencil_complete"]
        )
        self.assertIsNone(boundary["cross_window_A_neighbor_rms"])

    def test_cross_window_gate_catches_signal_two_grid_steps_from_crossing(self):
        mode = np.zeros_like(self.mode)
        mode[0, 120] = 1.0
        mode[1, 80] = 0.01
        mode[1, 82] = 0.3
        upper = np.full(mode.shape[1], 1.1)
        upper[:80] = 0.9
        upper[80] = 1.0
        disabled_axis = AxisArtifactConfig(
            axis_amplitude_min=None,
            axis_width_max_grid=None,
        )
        disabled_grid = GridScaleSpikeConfig(
            amplitude_min=None,
            width_max_grid=None,
        )

        one_grid = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=0.03
            ),
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                half_width_grid=1,
                amplitude_min=0.25,
                w_min=0.05,
            ),
        )
        two_grid = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=0.03
            ),
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                half_width_grid=2,
                amplitude_min=0.25,
                w_min=0.05,
            ),
        )

        self.assertEqual(one_grid.decision, "REVIEW")
        self.assertLess(
            two_grid.features["rf_standard_features"]["W_star_max"],
            0.03,
        )
        self.assertEqual(two_grid.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertEqual(
            two_grid.triggered_rules,
            (BAD_CONT_CROSS_WINDOW,),
        )
        cross_features = two_grid.features["crossing_features"]
        self.assertAlmostEqual(cross_features["cross_window_A_max"], 0.3)
        self.assertAlmostEqual(cross_features["cross_window_W_max"], 0.09)
        self.assertAlmostEqual(
            cross_features["cross_window_A_neighbor_rms"],
            np.sqrt((0.29**2 + 3 * 0.3**2) / 4.0),
        )

        disabled = evaluate_mode(
            self.base,
            mode=mode,
            low2=self.low2,
            high2=upper**2,
            axis_artifact_config=disabled_axis,
            grid_scale_spike_config=disabled_grid,
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=0.03
            ),
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                half_width_grid=2,
                amplitude_min=None,
                w_min=None,
            ),
        )
        self.assertEqual(disabled.decision, "REVIEW")
        self.assertAlmostEqual(
            disabled.features["crossing_features"]["cross_window_A_max"],
            0.3,
        )

    def test_cross_window_gate_uses_amplitude_or_energy_and_inclusive_thresholds(self):
        mode = np.zeros_like(self.mode)
        mode[0, 120] = 1.0
        mode[0, 80] = 0.01
        mode[1, 82] = 0.2
        mode[2, 82] = 0.2
        upper = np.full(mode.shape[1], 1.1)
        upper[:80] = 0.9
        upper[80] = 1.0
        common = {
            "mode": mode,
            "low2": self.low2,
            "high2": upper**2,
            "axis_artifact_config": AxisArtifactConfig(
                axis_amplitude_min=None,
                axis_width_max_grid=None,
            ),
            "grid_scale_spike_config": GridScaleSpikeConfig(
                amplitude_min=None,
                width_max_grid=None,
            ),
            "continuum_crossing_config": ContinuumCrossingConfig(
                w_cross_threshold=None
            ),
        }

        energy_only = evaluate_mode(
            self.base,
            **common,
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                half_width_grid=2,
                amplitude_min=0.25,
                w_min=0.08,
            ),
        )
        self.assertEqual(energy_only.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertLess(
            energy_only.features["crossing_features"]["cross_window_A_max"],
            0.25,
        )
        self.assertAlmostEqual(
            energy_only.features["crossing_features"]["cross_window_W_max"],
            0.08,
        )
        self.assertAlmostEqual(
            energy_only.features["crossing_features"][
                "cross_window_A_neighbor_rms"
            ],
            0.2,
        )

        amplitude_mode = mode.copy()
        amplitude_mode[1, 82] = 0.25
        amplitude_mode[2, 82] = 0.0
        amplitude_only = evaluate_mode(
            self.base,
            **{**common, "mode": amplitude_mode},
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                half_width_grid=2,
                amplitude_min=0.25,
                w_min=None,
            ),
        )
        self.assertEqual(amplitude_only.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertAlmostEqual(
            amplitude_only.features["crossing_features"][
                "cross_window_A_neighbor_rms"
            ],
            0.25,
        )

        smooth_mode = np.zeros_like(self.mode)
        smooth_mode[0, 120] = 1.0
        smooth_mode[1, 76:85] = 0.3
        smooth = evaluate_mode(
            self.base,
            **{**common, "mode": smooth_mode},
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                half_width_grid=2,
                amplitude_min=0.25,
                w_min=0.05,
            ),
        )
        self.assertEqual(smooth.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertAlmostEqual(
            smooth.features["crossing_features"][
                "cross_window_A_neighbor_rms"
            ],
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
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
            ),
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
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
            ),
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
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
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
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
            ),
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

    def test_cross_window_gate_precedes_edge_gate(self):
        mode = np.zeros_like(self.mode)
        mode[1, 194:199] = [0.4, 0.8, 1.0, 0.8, 0.4]
        upper = np.full(mode.shape[1], 1.1)
        upper[:194] = 0.9
        upper[194] = 1.0
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=np.full(mode.shape[1], 0.5**2),
            high2=upper**2,
            axis_artifact_config=AxisArtifactConfig(
                axis_amplitude_min=None,
                axis_width_max_grid=None,
            ),
            grid_scale_spike_config=GridScaleSpikeConfig(
                amplitude_min=None,
                width_max_grid=None,
            ),
            grid_scale_packet_config=GridScalePacketConfig(
                amplitude_min=None,
            ),
            continuum_crossing_config=ContinuumCrossingConfig(
                w_cross_threshold=None
            ),
        )

        self.assertTrue(
            result.features["boundary_features"]["edge_artifact"][
                "edge_energy_peak_in_window"
            ]
        )
        self.assertEqual(result.primary_reason, BAD_CONT_CROSS_WINDOW)
        self.assertEqual(result.triggered_rules, (BAD_CONT_CROSS_WINDOW,))

    def test_interior_envelope_gate_includes_exact_width_and_radius_limits(self):
        mode = narrow_total_energy_mode()
        low2 = np.full(mode.shape[1], 0.5**2)
        high2 = np.full(mode.shape[1], 1.5**2)

        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=low2,
            high2=high2,
        )
        features = result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertEqual(result.decision, "BAD")
        self.assertEqual(result.primary_reason, BAD_INTERIOR_UNRESOLVED_ENVELOPE)
        self.assertEqual(
            result.triggered_rules,
            (BAD_INTERIOR_UNRESOLVED_ENVELOPE,),
        )
        self.assertTrue(features["candidate_found"])
        self.assertAlmostEqual(features["energy_peak_r"], 0.5)
        self.assertAlmostEqual(features["energy_halfmax_width_grid"], 2.0)
        self.assertFalse(features["extremum_match_found"])
        self.assertFalse(features["extremum_exception_applied"])

    def test_interior_envelope_inclusive_limits_tolerate_grid_roundoff(self):
        mode = narrow_total_energy_mode(nr=201, peak_index=100)
        radial_grid = np.linspace(0.0, 1.0, mode.shape[1])
        upper = 26.0 + 30.0 * (radial_grid - 0.48) ** 2
        result = evaluate_mode(
            dict(self.base, omega=25.0),
            mode=mode,
            low2=np.full(mode.shape[1], 10.0**2),
            high2=np.square(upper),
        )
        features = result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertGreater(features["energy_halfmax_width_grid"], 2.0)
        self.assertGreater(features["ext_dr"], 0.02)
        self.assertTrue(features["candidate_found"])
        self.assertTrue(features["extremum_exception_applied"])
        self.assertEqual(result.decision, "REVIEW")

    def test_interior_envelope_gate_excludes_high_r_or_resolved_envelopes(self):
        low2 = np.full(65, 0.5**2)
        high2 = np.full(65, 1.5**2)
        high_r = narrow_total_energy_mode(peak_index=33)
        high_r_result = evaluate_mode(
            self.base,
            mode=high_r,
            low2=low2,
            high2=high2,
        )
        high_r_features = high_r_result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertEqual(high_r_result.decision, "REVIEW")
        self.assertGreater(high_r_features["energy_peak_r"], 0.5)
        self.assertAlmostEqual(
            high_r_features["energy_halfmax_width_grid"], 2.0
        )
        self.assertFalse(high_r_features["candidate_found"])

        resolved = np.zeros((4, 65), dtype=float)
        resolved[1, 30:34] = [np.sqrt(0.5), 1.0, 1.0, np.sqrt(0.5)]
        resolved_result = evaluate_mode(
            self.base,
            mode=resolved,
            low2=low2,
            high2=high2,
        )
        resolved_features = resolved_result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertEqual(resolved_result.decision, "REVIEW")
        self.assertLessEqual(resolved_features["energy_peak_r"], 0.5)
        self.assertAlmostEqual(
            resolved_features["energy_halfmax_width_grid"], 3.0
        )
        self.assertFalse(resolved_features["candidate_found"])

    def test_interior_envelope_extremum_exception_requires_every_bound(self):
        mode = narrow_total_energy_mode()
        radial_grid = np.linspace(0.0, 1.0, mode.shape[1])
        row = dict(self.base, omega=25.0)
        low2 = np.full(mode.shape[1], 10.0**2)
        disabled_crossing = ContinuumCrossingConfig(w_cross_threshold=None)
        disabled_window = ContinuumCrossingWindowConfig(
            amplitude_min=None,
            w_min=None,
        )
        cases = (
            (
                "inclusive upper frequency bound",
                26.0 + 30.0 * (radial_grid - radial_grid[31]) ** 2,
                True,
                True,
            ),
            (
                "inclusive zero frequency bound",
                25.0 + 30.0 * (radial_grid - radial_grid[31]) ** 2,
                True,
                True,
            ),
            (
                "negative frequency clearance",
                24.75 + 30.0 * (radial_grid - radial_grid[31]) ** 2,
                True,
                False,
            ),
            (
                "frequency clearance above limit",
                26.25 + 30.0 * (radial_grid - radial_grid[31]) ** 2,
                True,
                False,
            ),
            (
                "radial mismatch above limit",
                26.0 + 30.0 * (radial_grid - radial_grid[30]) ** 2,
                True,
                False,
            ),
            (
                "no extremum match",
                np.full(mode.shape[1], 30.0),
                False,
                False,
            ),
        )

        for name, upper, match_found, exception_applied in cases:
            with self.subTest(name=name):
                result = evaluate_mode(
                    row,
                    mode=mode,
                    low2=low2,
                    high2=np.square(upper),
                    continuum_crossing_config=disabled_crossing,
                    continuum_crossing_window_config=disabled_window,
                )
                features = result.features["resolution_features"][
                    "interior_unresolved_envelope"
                ]
                self.assertEqual(features["extremum_match_found"], match_found)
                self.assertEqual(
                    features["extremum_exception_applied"], exception_applied
                )
                if name == "inclusive upper frequency bound":
                    self.assertAlmostEqual(features["ext_df_gap"], 0.04)
                elif name == "inclusive zero frequency bound":
                    self.assertAlmostEqual(features["ext_df_gap"], 0.0)
                expected_decision = "REVIEW" if exception_applied else "BAD"
                self.assertEqual(result.decision, expected_decision)
                if exception_applied:
                    self.assertEqual(result.primary_reason, NO_GOOD_TEMPLATE)
                    self.assertFalse(
                        result.features["extremum_features"]["match_found"]
                    )
                else:
                    self.assertEqual(
                        result.primary_reason,
                        BAD_INTERIOR_UNRESOLVED_ENVELOPE,
                    )

    def test_disabled_interior_envelope_gate_retains_evidence(self):
        mode = narrow_total_energy_mode()
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=np.full(mode.shape[1], 0.5**2),
            high2=np.full(mode.shape[1], 1.5**2),
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None,
            ),
        )
        features = result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertEqual(result.decision, "REVIEW")
        self.assertIsNone(features["candidate_found"])
        self.assertAlmostEqual(features["energy_peak_r"], 0.5)
        self.assertAlmostEqual(features["energy_halfmax_width_grid"], 2.0)
        self.assertFalse(features["extremum_match_found"])
        self.assertFalse(features["extremum_exception_applied"])

    def test_interior_envelope_exception_includes_radial_mismatch_limit(self):
        mode = narrow_total_energy_mode()
        radial_grid = np.linspace(0.0, 1.0, mode.shape[1])
        radial_mismatch = radial_grid[32] - radial_grid[31]
        upper = 26.0 + 30.0 * (radial_grid - radial_grid[31]) ** 2
        result = evaluate_mode(
            dict(self.base, omega=25.0),
            mode=mode,
            low2=np.full(mode.shape[1], 10.0**2),
            high2=np.square(upper),
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                ext_dr_max=radial_mismatch,
            ),
        )
        features = result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertAlmostEqual(features["ext_dr"], radial_mismatch)
        self.assertEqual(features["ext_dr"], features["ext_dr_max"])
        self.assertTrue(features["extremum_exception_applied"])
        self.assertEqual(result.decision, "REVIEW")

    def test_interior_envelope_gate_uses_total_energy_not_one_harmonic(self):
        mode = np.zeros((4, 65), dtype=float)
        mode[0, 24:41] = 1.0
        mode[1, 32] = 1.0
        result = evaluate_mode(
            self.base,
            mode=mode,
            low2=np.full(mode.shape[1], 0.5**2),
            high2=np.full(mode.shape[1], 1.5**2),
            grid_scale_spike_config=GridScaleSpikeConfig(amplitude_min=None),
        )
        grid_features = result.features["numerical_structure_features"][
            "grid_scale_spike"
        ]
        envelope_features = result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertEqual(result.decision, "REVIEW")
        self.assertTrue(grid_features["grid_scale_candidate_found"])
        self.assertAlmostEqual(grid_features["grid_scale_halfmax_width_grid"], 1.0)
        self.assertAlmostEqual(envelope_features["energy_peak_r"], 0.5)
        self.assertGreater(envelope_features["energy_halfmax_width_grid"], 2.0)
        self.assertFalse(envelope_features["candidate_found"])

    def test_edge_gate_precedes_interior_envelope_exception(self):
        mode = narrow_total_energy_mode()
        radial_grid = np.linspace(0.0, 1.0, mode.shape[1])
        row = dict(self.base, omega=25.0)
        upper = 26.0 + 30.0 * (radial_grid - radial_grid[31]) ** 2
        result = evaluate_mode(
            row,
            mode=mode,
            low2=np.full(mode.shape[1], 10.0**2),
            high2=np.square(upper),
            edge_artifact_config=EdgeArtifactConfig(
                r_edge_min=0.5,
                edge_width_max_grid=2.0,
            ),
        )
        features = result.features["resolution_features"][
            "interior_unresolved_envelope"
        ]

        self.assertTrue(features["candidate_found"])
        self.assertTrue(features["extremum_exception_applied"])
        self.assertEqual(result.primary_reason, BAD_EDGE_SPIKE)
        self.assertEqual(result.triggered_rules, (BAD_EDGE_SPIKE,))

    def test_interior_harmonic_effective_count_is_simultaneous(self):
        simultaneous = np.zeros((6, 201), dtype=float)
        simultaneous[:, :101] = np.array([1, -1, 1, -1, 1, -1])[:, None]
        simultaneous_features = extract_interior_harmonic_incoherence_features(
            simultaneous
        )

        self.assertAlmostEqual(
            simultaneous_features["core_effective_harmonic_count_wmean"],
            6.0,
        )
        self.assertEqual(
            simultaneous_features["effective_harmonic_count_threshold"], 3.0
        )
        self.assertAlmostEqual(
            simultaneous_features["global_effective_harmonic_count_wmean"],
            6.0,
        )
        self.assertAlmostEqual(
            simultaneous_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            simultaneous_features[
                "core_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            simultaneous_features[
                "total_energy_fraction_in_core_above_effective_harmonic_count_threshold"
            ],
            1.0,
        )
        self.assertEqual(simultaneous_features["active_core_harmonic_count"], 6)
        self.assertAlmostEqual(simultaneous_features["core_js_divergence"], 0.0)
        self.assertAlmostEqual(
            simultaneous_features["core_adjacent_harmonic_coherence"], 1.0
        )
        self.assertFalse(simultaneous_features["candidate_found"])

        sequential = np.zeros((6, 201), dtype=float)
        for radial_index in range(101):
            sequential[min(radial_index // 17, 5), radial_index] = 1.0
        sequential_features = extract_interior_harmonic_incoherence_features(
            sequential
        )

        self.assertEqual(sequential_features["active_core_harmonic_count"], 6)
        self.assertAlmostEqual(
            sequential_features["core_effective_harmonic_count_wmean"],
            1.0,
        )
        self.assertAlmostEqual(
            sequential_features["global_effective_harmonic_count_wmean"],
            1.0,
        )
        self.assertAlmostEqual(
            sequential_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            sequential_features[
                "core_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            sequential_features[
                "total_energy_fraction_in_core_above_effective_harmonic_count_threshold"
            ],
            0.0,
        )
        self.assertLess(sequential_features["incoherence_score"], 0.1)
        self.assertFalse(sequential_features["candidate_found"])

    def test_harmonic_participation_audit_weighting_and_null_semantics(self):
        split_energy = np.zeros((4, 201), dtype=float)
        split_energy[:, 0] = 1.0
        split_energy[0, -1] = 2.0
        split_features = extract_interior_harmonic_incoherence_features(
            split_energy
        )

        self.assertAlmostEqual(split_features["core_energy_fraction"], 0.5)
        self.assertAlmostEqual(
            split_features["global_effective_harmonic_count_wmean"], 2.5
        )
        self.assertAlmostEqual(
            split_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            split_features[
                "core_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            split_features[
                "total_energy_fraction_in_core_above_effective_harmonic_count_threshold"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            split_features[
                "total_energy_fraction_in_core_above_effective_harmonic_count_threshold"
            ],
            split_features["core_energy_fraction"]
            * split_features[
                "core_energy_fraction_above_effective_harmonic_count_threshold"
            ],
        )

        exactly_three = np.zeros((3, 201), dtype=float)
        exactly_three[:, 0] = 1.0
        equality_features = extract_interior_harmonic_incoherence_features(
            exactly_three
        )
        self.assertAlmostEqual(
            equality_features["global_effective_harmonic_count_wmean"], 3.0
        )
        self.assertAlmostEqual(
            equality_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            0.0,
        )

        low_energy_tail = np.zeros((6, 201), dtype=float)
        low_energy_tail[0, 0] = 1.0
        low_energy_tail[:, -1] = np.sqrt(0.01 / 6.0)
        tail_features = extract_interior_harmonic_incoherence_features(
            low_energy_tail
        )
        self.assertAlmostEqual(
            tail_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            0.01 / 1.01,
        )

        no_core = np.zeros((4, 201), dtype=float)
        no_core[:, -1] = 1.0
        no_core_features = extract_interior_harmonic_incoherence_features(no_core)
        self.assertAlmostEqual(no_core_features["core_energy_fraction"], 0.0)
        self.assertAlmostEqual(
            no_core_features["global_effective_harmonic_count_wmean"], 4.0
        )
        self.assertAlmostEqual(
            no_core_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ],
            1.0,
        )
        self.assertIsNone(
            no_core_features[
                "core_energy_fraction_above_effective_harmonic_count_threshold"
            ]
        )
        self.assertAlmostEqual(
            no_core_features[
                "total_energy_fraction_in_core_above_effective_harmonic_count_threshold"
            ],
            0.0,
        )

        zero_features = extract_interior_harmonic_incoherence_features(
            np.zeros((4, 201), dtype=float)
        )
        for name in (
            "global_effective_harmonic_count_wmean",
            "global_energy_fraction_above_effective_harmonic_count_threshold",
            "core_energy_fraction_above_effective_harmonic_count_threshold",
            "total_energy_fraction_in_core_above_effective_harmonic_count_threshold",
        ):
            self.assertIsNone(zero_features[name])

    def test_interior_harmonic_base2_js_and_undefined_pair_are_explicit(self):
        alternating = np.zeros((2, 201), dtype=float)
        alternating[0, :101:2] = 1.0
        alternating[1, 1:101:2] = 1.0
        alternating_features = extract_interior_harmonic_incoherence_features(
            alternating,
            config=InteriorHarmonicIncoherenceConfig(max_lag_grid=0),
        )

        self.assertAlmostEqual(alternating_features["core_js_divergence"], 1.0)
        self.assertAlmostEqual(
            alternating_features["core_effective_harmonic_count_wmean"], 1.0
        )
        self.assertAlmostEqual(
            alternating_features["core_adjacent_harmonic_coherence"], 0.0
        )
        self.assertAlmostEqual(alternating_features["incoherence_score"], 1.0)
        self.assertTrue(alternating_features["candidate_found"])

        one_harmonic = np.zeros((1, 201), dtype=float)
        one_harmonic[0, :101] = 1.0
        one_harmonic_features = extract_interior_harmonic_incoherence_features(
            one_harmonic
        )
        self.assertEqual(one_harmonic_features["active_core_harmonic_count"], 1)
        self.assertEqual(
            one_harmonic_features["active_adjacent_harmonic_pair_count"], 0
        )
        self.assertIsNone(
            one_harmonic_features["core_adjacent_harmonic_coherence"]
        )
        self.assertIsNone(one_harmonic_features["incoherence_score"])
        self.assertFalse(one_harmonic_features["candidate_found"])

    def test_interior_harmonic_core_and_active_boundaries_are_inclusive(self):
        boundary_mode = np.zeros((1, 201), dtype=float)
        boundary_mode[0, 100:102] = 1.0
        boundary_features = extract_interior_harmonic_incoherence_features(
            boundary_mode
        )
        self.assertEqual(boundary_features["core_radial_sample_count"], 101)
        self.assertAlmostEqual(boundary_features["core_energy_fraction"], 0.5)
        self.assertAlmostEqual(
            boundary_features["core_effective_harmonic_count_wmean"], 1.0
        )

        cutoff_mode = np.zeros((2, 201), dtype=float)
        cutoff_mode[0, :101] = 1.0
        cutoff_mode[1, :101] = np.sqrt(0.005 / 0.995)
        cutoff_features = extract_interior_harmonic_incoherence_features(
            cutoff_mode
        )
        self.assertEqual(cutoff_features["active_core_harmonic_count"], 2)
        self.assertEqual(cutoff_features["active_adjacent_harmonic_pair_count"], 1)

    def test_interior_harmonic_lag_limit_is_inclusive(self):
        mode = np.zeros((3, 201), dtype=float)
        mode[0, 30] = 1.0
        mode[1, 35] = 1.0
        mode[2, :101] = 0.001

        lag_four = extract_interior_harmonic_incoherence_features(
            mode,
            config=InteriorHarmonicIncoherenceConfig(max_lag_grid=4),
        )
        lag_five = extract_interior_harmonic_incoherence_features(
            mode,
            config=InteriorHarmonicIncoherenceConfig(max_lag_grid=5),
        )

        self.assertAlmostEqual(
            lag_four["core_adjacent_harmonic_coherence"], 0.0
        )
        self.assertGreater(lag_four["incoherence_score"], 0.1)
        self.assertAlmostEqual(
            lag_five["core_adjacent_harmonic_coherence"], 1.0
        )
        self.assertAlmostEqual(lag_five["incoherence_score"], 0.0)

    def test_interior_harmonic_score_is_scale_stable_and_strict(self):
        mode = broadband_incoherent_mode()
        baseline = extract_interior_harmonic_incoherence_features(mode)
        scaled = extract_interior_harmonic_incoherence_features(7.0 * mode)
        score = baseline["incoherence_score"]

        self.assertGreater(score, 0.1)
        for name in (
            "core_energy_fraction",
            "core_js_divergence",
            "core_effective_harmonic_count_wmean",
            "global_effective_harmonic_count_wmean",
            "global_energy_fraction_above_effective_harmonic_count_threshold",
            "core_energy_fraction_above_effective_harmonic_count_threshold",
            "total_energy_fraction_in_core_above_effective_harmonic_count_threshold",
            "core_adjacent_harmonic_coherence",
            "incoherence_score",
        ):
            self.assertAlmostEqual(baseline[name], scaled[name])

        exact = extract_interior_harmonic_incoherence_features(
            mode,
            config=InteriorHarmonicIncoherenceConfig(score_threshold=score),
        )
        just_below = extract_interior_harmonic_incoherence_features(
            mode,
            config=InteriorHarmonicIncoherenceConfig(
                score_threshold=np.nextafter(score, 0.0)
            ),
        )
        self.assertFalse(exact["candidate_found"])
        self.assertTrue(just_below["candidate_found"])

    def test_interior_harmonic_gate_and_resolution_guard(self):
        mode = broadband_incoherent_mode()
        result = evaluate_incoherence_only(self.base, mode)
        features = result.features["numerical_structure_features"][
            "interior_harmonic_incoherence"
        ]

        self.assertEqual(result.decision, "BAD")
        self.assertEqual(
            result.primary_reason, BAD_INTERIOR_HARMONIC_INCOHERENCE
        )
        self.assertEqual(
            result.triggered_rules,
            (BAD_INTERIOR_HARMONIC_INCOHERENCE,),
        )
        self.assertTrue(features["resolution_eligible"])
        self.assertTrue(features["candidate_found"])

        other_resolution = broadband_incoherent_mode(nr=200)
        guarded = evaluate_incoherence_only(self.base, other_resolution)
        guarded_features = guarded.features["numerical_structure_features"][
            "interior_harmonic_incoherence"
        ]
        self.assertEqual(guarded.decision, "REVIEW")
        self.assertFalse(guarded_features["resolution_eligible"])
        self.assertFalse(guarded_features["candidate_found"])
        self.assertGreater(guarded_features["incoherence_score"], 0.1)
        self.assertIsNotNone(
            guarded_features["global_effective_harmonic_count_wmean"]
        )
        self.assertIsNotNone(
            guarded_features[
                "global_energy_fraction_above_effective_harmonic_count_threshold"
            ]
        )

        disabled = evaluate_incoherence_only(
            self.base,
            mode,
            incoherence_config=InteriorHarmonicIncoherenceConfig(
                score_threshold=None
            ),
        )
        disabled_features = disabled.features["numerical_structure_features"][
            "interior_harmonic_incoherence"
        ]
        self.assertEqual(disabled.decision, "REVIEW")
        self.assertIsNone(disabled_features["candidate_found"])
        self.assertAlmostEqual(
            disabled_features["incoherence_score"],
            features["incoherence_score"],
        )
        for name in (
            "global_effective_harmonic_count_wmean",
            "global_energy_fraction_above_effective_harmonic_count_threshold",
            "core_energy_fraction_above_effective_harmonic_count_threshold",
            "total_energy_fraction_in_core_above_effective_harmonic_count_threshold",
        ):
            self.assertAlmostEqual(disabled_features[name], features[name])

    def test_interior_harmonic_configuration_rejects_invalid_values(self):
        invalid_cases = (
            {"core_r_max": 1.01},
            {"active_core_energy_fraction_min": 0.0},
            {"max_lag_grid": -1},
            {"score_threshold": float("nan")},
            {"calibrated_n_radial": 1},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InteriorHarmonicIncoherenceConfig(**kwargs)

    def test_interior_envelope_gate_precedes_harmonic_incoherence(self):
        mode = broadband_incoherent_mode()
        result = evaluate_incoherence_only(
            self.base,
            mode,
            interior_config=InteriorUnresolvedEnvelopeConfig(
                peak_r_max=1.0,
                width_max_grid=1000.0,
            ),
        )
        incoherence_features = result.features["numerical_structure_features"][
            "interior_harmonic_incoherence"
        ]

        self.assertTrue(incoherence_features["candidate_found"])
        self.assertEqual(result.primary_reason, BAD_INTERIOR_UNRESOLVED_ENVELOPE)
        self.assertEqual(
            result.triggered_rules,
            (BAD_INTERIOR_UNRESOLVED_ENVELOPE,),
        )

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
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(
                width_max_grid=None,
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
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                amplitude_min=None,
                w_min=None,
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
            continuum_crossing_window_config=ContinuumCrossingWindowConfig(
                amplitude_min=None,
                w_min=None,
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


class MixedSorterMethodIntegrationTests(unittest.TestCase):
    def _common_args(self) -> list[str]:
        return [
            "--shot_dir",
            "/tmp/synthetic_shot",
            "--out_dir",
            "/tmp/synthetic_out",
        ]

    def test_direct_mixed_sorter_bootstraps_src_without_pythonpath(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "sort_shot_mixed.py"),
                    "--help",
                ],
                cwd=temporary,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )
        self.assertIn("--method", completed.stdout)

    def test_rules_is_default_and_accepts_optional_rf_for_dedup_only(self):
        args = parse_mixed_args(self._common_args())
        self.assertEqual(DEFAULT_METHOD, RULES_METHOD)
        self.assertEqual(args.method, RULES_METHOD)
        self.assertEqual(args.rule_config, DEFAULT_RULE_CONFIG)
        self.assertEqual(args.rule_config, PRODUCTION_RULE_CONFIG_NAME)
        self.assertIsNone(args.rf_model)

        with_rf = parse_mixed_args(
            [*self._common_args(), "--rf_model", "/tmp/ranker.joblib"]
        )
        self.assertEqual(with_rf.method, RULES_METHOD)
        self.assertEqual(with_rf.rf_model, "/tmp/ranker.joblib")

    def test_method_specific_parser_validation_is_strict(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_mixed_args([*self._common_args(), "--cnn_model", "cnn.pt"])
            with self.assertRaises(SystemExit):
                parse_mixed_args([*self._common_args(), "--rel_freq_tol", "0.03"])
            with self.assertRaises(SystemExit):
                parse_mixed_args([*self._common_args(), "--method", RF_CNN_METHOD])
            with self.assertRaises(SystemExit):
                parse_mixed_args(
                    [
                        *self._common_args(),
                        "--method",
                        RF_CNN_METHOD,
                        "--rf_model",
                        "rf.joblib",
                    ]
                )

        legacy = parse_mixed_args(
            [
                *self._common_args(),
                "--method",
                RF_CNN_METHOD,
                "--rf_model",
                "rf.joblib",
                "--cnn_model",
                "cnn.pt",
            ]
        )
        self.assertEqual(legacy.method, RF_CNN_METHOD)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_mixed_args(
                    [
                        *self._common_args(),
                        "--method",
                        RF_CNN_METHOD,
                        "--rf_model",
                        "rf.joblib",
                        "--cnn_model",
                        "cnn.pt",
                        "--manual_overrides",
                        "manual.csv",
                    ]
                )

    def test_survivor_policy_preserves_rule_verdict_and_never_promotes_bad(self):
        review = {
            "path": "/data/shot/N1/review",
            "processing_status": "RULE_EVALUATED",
            "rule_decision": "REVIEW",
            "rule_primary_reason": NO_GOOD_TEMPLATE,
            "final_decision": "REVIEW",
            "decision_source": "rule_engine",
        }
        bad = {
            "path": "/data/shot/N1/bad",
            "processing_status": "RULE_EVALUATED",
            "rule_decision": "BAD",
            "rule_primary_reason": BAD_AXIS_SPIKE,
            "final_decision": "BAD",
            "decision_source": "rule_engine",
        }

        conservative = apply_rule_survivor_policy(
            [review, bad], RULE_SURVIVOR_POLICY_REVIEW
        )
        self.assertEqual(
            [row["final_decision"] for row in conservative], ["REVIEW", "BAD"]
        )
        self.assertFalse(conservative[0]["rule_survivor_accepted"])

        production = apply_rule_survivor_policy(
            [review, bad], RULE_SURVIVOR_POLICY_ACCEPT
        )
        self.assertEqual(production[0]["rule_decision"], "REVIEW")
        self.assertEqual(production[0]["rule_primary_reason"], NO_GOOD_TEMPLATE)
        self.assertEqual(production[0]["final_decision"], "GOOD")
        self.assertEqual(production[0]["decision_source"], "rule_survivor_policy")
        self.assertTrue(production[0]["rule_survivor_accepted"])
        self.assertEqual(production[1]["rule_decision"], "BAD")
        self.assertEqual(production[1]["final_decision"], "BAD")
        self.assertFalse(production[1]["rule_survivor_accepted"])

    def test_rules_method_promotes_survivors_writes_good_lists_and_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(
                root,
                names=("egn01w.low", "egn01w.high"),
            )
            out_dir = root / "out"
            args = parse_mixed_args(
                ["--shot_dir", str(shot), "--out_dir", str(out_dir)]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_rules_method(args)

            preliminary = [
                row for row in first.preliminary_rows if row.get("rule_version")
            ]
            self.assertEqual(len(preliminary), 2)
            self.assertTrue(
                all(row["rule_decision"] == "REVIEW" for row in preliminary)
            )
            self.assertTrue(
                all(row["rule_primary_reason"] == NO_GOOD_TEMPLATE for row in preliminary)
            )
            self.assertTrue(
                all(row["final_decision"] == "REVIEW" for row in preliminary)
            )

            final = [row for row in first.final_rows if row.get("rule_version")]
            self.assertEqual(len(final), 2)
            self.assertTrue(all(row["rule_decision"] == "REVIEW" for row in final))
            self.assertTrue(all(row["final_decision"] == "GOOD" for row in final))
            self.assertTrue(all(row["rule_survivor_accepted"] is True for row in final))
            self.assertTrue(all(row["selected_final"] is True for row in final))
            self.assertEqual(
                first.summary["rule_survivor_policy"],
                RULE_SURVIVOR_POLICY_ACCEPT,
            )
            self.assertEqual(first.summary["method"], RULES_METHOD)
            self.assertEqual(first.summary["n_rule_survivors_accepted"], 2)
            self.assertEqual(first.summary["n_preliminary_review"], 2)
            self.assertEqual(first.summary["n_final_review"], 0)
            self.assertEqual(first.summary["n_final_good_before_clustering"], 2)
            self.assertEqual(first.summary["n_final_good"], 2)
            self.assertEqual(
                first.summary["duplicate_processing_status"],
                "SKIPPED_NO_RF_CHECKPOINT",
            )
            self.assertFalse(first.summary["continuum_crossing_gate_enabled"])

            _fields, good_unchecked = read_dict_csv(
                out_dir / "good_tae_unchecked.csv"
            )
            _fields, good_final = read_dict_csv(out_dir / "good_tae_final.csv")
            _fields, review_rows = read_dict_csv(out_dir / "review_tae_like.csv")
            _fields, clusters = read_dict_csv(out_dir / "frequency_clusters.csv")
            self.assertEqual(len(good_unchecked), 2)
            self.assertEqual(len(good_final), 2)
            self.assertEqual(review_rows, [])
            self.assertEqual(len(clusters), 2)
            self.assertEqual(
                {row["cluster_status"] for row in clusters},
                {"SKIPPED_NO_RF_CHECKPOINT"},
            )
            self.assertEqual(
                {row["action"] for row in clusters},
                {"KEEP"},
            )

            before = {
                path.name: path.read_bytes() for path in sorted(out_dir.iterdir())
            }
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_rules_method(args)
            after = {
                path.name: path.read_bytes() for path in sorted(out_dir.iterdir())
            }
            self.assertEqual(first.summary, second.summary)
            self.assertEqual(before, after)

    def test_rules_method_does_not_promote_gate_rejection(self):
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
            args = parse_mixed_args(
                [
                    "--method",
                    RULES_METHOD,
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(root / "out"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_rules_method(args)

            row = next(row for row in result.final_rows if row.get("rule_version"))
            self.assertEqual(row["rule_decision"], "BAD")
            self.assertEqual(row["rule_primary_reason"], BAD_AXIS_SPIKE)
            self.assertEqual(row["final_decision"], "BAD")
            self.assertFalse(row["rule_survivor_accepted"])
            self.assertEqual(result.summary["n_rule_survivors_accepted"], 0)
            self.assertEqual(result.summary["n_final_good"], 0)
            _fields, bad_rows = read_dict_csv(root / "out" / "bad_tae_like.csv")
            _fields, good_rows = read_dict_csv(
                root / "out" / "good_tae_unchecked.csv"
            )
            self.assertEqual(len(bad_rows), 1)
            self.assertEqual(good_rows, [])

    def test_manual_override_can_demote_an_accepted_rule_survivor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            initial_args = parse_mixed_args(
                [
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(root / "initial"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                initial = run_rules_method(initial_args)
            accepted = next(
                row for row in initial.final_rows if row.get("rule_version")
            )
            self.assertEqual(accepted["rule_decision"], "REVIEW")
            self.assertEqual(accepted["final_decision"], "GOOD")

            override_path = root / "manual.csv"
            write_dict_csv(
                override_path,
                MANUAL_OVERRIDE_FIELDS,
                [override_row(accepted, decision="BAD")],
            )
            final_args = parse_mixed_args(
                [
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(root / "final"),
                    "--manual_overrides",
                    str(override_path),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_rules_method(final_args)

            row = next(row for row in result.final_rows if row.get("rule_version"))
            self.assertEqual(row["rule_decision"], "REVIEW")
            self.assertFalse(row["rule_survivor_accepted"])
            self.assertEqual(row["final_decision"], "BAD")
            self.assertEqual(row["decision_source"], "manual_override")
            self.assertEqual(row["override_status"], "APPLIED")
            self.assertEqual(result.summary["n_overrides_applied"], 1)
            self.assertEqual(
                json.loads(result.summary["transition_counts_json"]),
                {"REVIEW->BAD": 1},
            )
            self.assertEqual(result.summary["n_final_good"], 0)

    def test_stale_override_blocks_automatic_survivor_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            initial_args = parse_mixed_args(
                [
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(root / "initial"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                initial = run_rules_method(initial_args)
            accepted = next(
                row for row in initial.final_rows if row.get("rule_version")
            )

            stale = override_row(accepted, decision="BAD")
            stale["input_fingerprint"] = "0" * 64
            override_path = root / "stale.csv"
            write_dict_csv(override_path, MANUAL_OVERRIDE_FIELDS, [stale])
            final_args = parse_mixed_args(
                [
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(root / "final"),
                    "--manual_overrides",
                    str(override_path),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_rules_method(final_args)

            row = next(row for row in result.final_rows if row.get("rule_version"))
            self.assertEqual(row["rule_decision"], "REVIEW")
            self.assertEqual(row["final_decision"], "REVIEW")
            self.assertFalse(row["rule_survivor_accepted"])
            self.assertEqual(row["decision_source"], "override_review_required")
            self.assertEqual(row["override_status"], "STALE_FINGERPRINT")
            self.assertEqual(result.summary["n_stale_overrides"], 1)
            self.assertEqual(result.summary["n_final_good"], 0)


class WorkflowOutputTests(unittest.TestCase):
    def test_named_production_configuration_is_frozen_and_audited(self):
        configuration = load_rule_run_configuration(PRODUCTION_RULE_CONFIG_NAME)
        historical_v1 = (
            REPO_ROOT / "configs" / "rules" / "tae_rules_production_v1.yaml"
        )
        historical_v2 = (
            REPO_ROOT / "configs" / "rules" / "tae_rules_production_v2.yaml"
        )
        historical_v3 = (
            REPO_ROOT / "configs" / "rules" / "tae_rules_production_v3.yaml"
        )
        self.assertEqual(
            sha256_file(historical_v1),
            "a2c85d958eeebe4396a9ce0d2f52c3dbf157f1630d344279801c00bb826e6f39",
        )
        self.assertEqual(
            sha256_file(historical_v2),
            "7d31bd84466486f0c374372b489f04816c4f745503ef0328cb514c6ef3d7516f",
        )
        self.assertEqual(
            sha256_file(historical_v3),
            "fdaf5775a9908266ae0e539dcc5aa910ab8d16d4593cf4b2b87327a36c19ece3",
        )
        self.assertEqual(
            sha256_file(REPO_ROOT / "configs/rules/tae_rules_production_v4.yaml"),
            "ddefb105a8faac4d4050eda1636966d28dd6217c9af50305c7ae974c6666985b",
        )
        self.assertEqual(configuration.name, "tae_rules_production_v5")
        self.assertEqual(configuration.schema_version, RULE_CONFIG_SCHEMA_VERSION)
        self.assertEqual(configuration.rule_set_version, RULESET_VERSION)
        self.assertEqual(
            configuration.sha256,
            "982cc0ba3f17aae03a9fc6a4b662104200df0ff2897bda4de21131ce71c5bc9f",
        )
        self.assertEqual(
            dict(configuration.run_kwargs),
            {
                "fraction_tae_threshold": 0.5,
                "fraction_eae_threshold": 0.4,
                "signed_delta_eae_threshold": -0.1,
                "rel_freq_tol": 0.02,
                "axis_r_ax": 0.03,
                "axis_amplitude_min": 0.2,
                "axis_width_max_grid": 10.0,
                "grid_scale_amplitude_min": 0.3,
                "grid_scale_width_max_grid": 1.0,
                "grid_scale_high_r_cutoff_r": 0.7,
                "grid_scale_high_r_width_max_grid": 0.75,
                "grid_scale_packet_amplitude_min": 0.3,
                "grid_scale_packet_step_min": 0.2,
                "grid_scale_packet_min_large_turns": 3,
                "grid_scale_packet_window_span_grid": 4,
                "grid_scale_packet_peak_r_max": 0.5,
                "near_axis_grid_oscillation_peak_r_max": 0.1,
                "near_axis_grid_oscillation_amplitude_min": 0.1,
                "near_axis_grid_oscillation_min_consecutive_sign_flips": 4,
                "near_axis_grid_oscillation_step_l2_min": 0.3,
                "w_cross_threshold": None,
                "cross_window_half_width_grid": 2,
                "cross_window_amplitude_min": 0.25,
                "cross_window_w_min": 0.05,
                "edge_r_min": 0.97,
                "edge_width_max_grid": 10.0,
                "interior_envelope_peak_r_max": 0.5,
                "interior_envelope_width_max_grid": 2.0,
                "interior_envelope_extremum_r_min": 0.03,
                "interior_envelope_extremum_r_max": 0.5,
                "interior_envelope_ext_dr_max": 0.02,
                "interior_envelope_ext_df_gap_min": 0.0,
                "interior_envelope_ext_df_gap_max": 0.04,
                "interior_harmonic_core_r_max": 0.5,
                "interior_harmonic_active_core_energy_fraction_min": 0.005,
                "interior_harmonic_max_lag_grid": 5,
                "interior_harmonic_incoherence_score_threshold": 0.1,
                "interior_harmonic_calibrated_n_radial": 201,
                "continuum_crossing_tail_k_min": 0.4,
                "continuum_crossing_tail_top2_ratio_min": 0.035,
                "continuum_crossing_tail_half_width_grid": 4,
                "continuum_crossing_tail_calibrated_n_radial": 201,
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            result = run_shot(
                shot,
                root / "out",
                **configuration.run_kwargs,
                rule_configuration_name=configuration.name,
                rule_configuration_schema_version=configuration.schema_version,
                rule_configuration_sha256=configuration.sha256,
            )

        self.assertEqual(
            result.summary["rule_configuration_name"], configuration.name
        )
        self.assertEqual(
            result.summary["rule_configuration_schema_version"],
            configuration.schema_version,
        )
        self.assertEqual(
            result.summary["rule_configuration_sha256"], configuration.sha256
        )
        self.assertTrue(result.summary["axis_artifact_gate_enabled"])
        self.assertTrue(result.summary["grid_scale_spike_gate_enabled"])
        self.assertTrue(result.summary["grid_scale_packet_gate_enabled"])
        self.assertTrue(
            result.summary["near_axis_grid_oscillation_gate_enabled"]
        )
        self.assertEqual(
            result.summary["near_axis_grid_oscillation_peak_r_max"], 0.1
        )
        self.assertEqual(
            result.summary["near_axis_grid_oscillation_amplitude_min"], 0.1
        )
        self.assertEqual(
            result.summary[
                "near_axis_grid_oscillation_min_consecutive_sign_flips"
            ],
            4,
        )
        self.assertEqual(
            result.summary["near_axis_grid_oscillation_step_l2_min"], 0.3
        )
        self.assertFalse(result.summary["continuum_crossing_gate_enabled"])
        self.assertIsNone(result.summary["continuum_crossing_w_threshold"])
        self.assertTrue(
            result.summary["continuum_crossing_window_gate_enabled"]
        )
        self.assertTrue(result.summary["edge_artifact_gate_enabled"])
        self.assertTrue(result.summary["interior_envelope_gate_enabled"])
        self.assertEqual(result.summary["interior_envelope_peak_r_max"], 0.5)
        self.assertEqual(result.summary["interior_envelope_width_max_grid"], 2.0)
        self.assertEqual(result.summary["interior_envelope_extremum_r_min"], 0.03)
        self.assertEqual(result.summary["interior_envelope_extremum_r_max"], 0.5)
        self.assertEqual(result.summary["interior_envelope_ext_dr_max"], 0.02)
        self.assertEqual(result.summary["interior_envelope_ext_df_gap_min"], 0.0)
        self.assertEqual(result.summary["interior_envelope_ext_df_gap_max"], 0.04)
        self.assertTrue(
            result.summary["interior_harmonic_incoherence_gate_enabled"]
        )
        self.assertEqual(result.summary["interior_harmonic_core_r_max"], 0.5)
        self.assertEqual(
            result.summary[
                "interior_harmonic_active_core_energy_fraction_min"
            ],
            0.005,
        )
        self.assertEqual(result.summary["interior_harmonic_max_lag_grid"], 5)
        self.assertEqual(
            result.summary["interior_harmonic_incoherence_score_threshold"],
            0.1,
        )
        self.assertEqual(
            result.summary["interior_harmonic_calibrated_n_radial"], 201
        )
        self.assertEqual(
            result.summary["n_interior_harmonic_resolution_eligible"], 0
        )
        self.assertEqual(
            result.summary["n_interior_harmonic_resolution_ineligible"], 1
        )

        with tempfile.TemporaryDirectory() as temporary:
            modified = Path(temporary) / "modified.yaml"
            modified.write_text(configuration.source_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "frozen configuration"):
                load_rule_run_configuration(modified)

    def test_named_configuration_rejects_cli_threshold_overrides(self):
        common = [
            "--shot_dir",
            "/tmp/shot",
            "--out_dir",
            "/tmp/out",
            "--rule_config",
            PRODUCTION_RULE_CONFIG_NAME,
        ]
        args = parse_args(common)
        self.assertEqual(args.rule_configuration_name, PRODUCTION_RULE_CONFIG_NAME)
        self.assertIsNone(args.w_cross_threshold)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([*common, "--cross_window_w_min", "0.1"])
            with self.assertRaises(SystemExit):
                parse_args([*common, "--interior_envelope_width_max_grid", "3"])
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        *common,
                        "--near_axis_grid_oscillation_step_l2_min",
                        "0.4",
                    ]
                )
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        *common,
                        "--interior_harmonic_incoherence_score_threshold",
                        "0.2",
                    ]
                )
            with self.assertRaises(SystemExit):
                parse_args([*common, "--disable_interior_harmonic_incoherence"])

    def test_calibration_cli_accepts_interior_envelope_thresholds_and_disable(self):
        args = parse_args(
            [
                "--shot_dir",
                "/tmp/shot",
                "--out_dir",
                "/tmp/out",
                "--interior_envelope_peak_r_max",
                "0.45",
                "--interior_envelope_width_max_grid",
                "1.5",
                "--interior_envelope_extremum_r_min",
                "0.04",
                "--interior_envelope_extremum_r_max",
                "0.48",
                "--interior_envelope_ext_dr_max",
                "0.015",
                "--interior_envelope_ext_df_gap_min",
                "0.005",
                "--interior_envelope_ext_df_gap_max",
                "0.035",
                "--disable_interior_unresolved_envelope",
            ]
        )

        self.assertEqual(args.interior_envelope_peak_r_max, 0.45)
        self.assertEqual(args.interior_envelope_width_max_grid, 1.5)
        self.assertEqual(args.interior_envelope_extremum_r_min, 0.04)
        self.assertEqual(args.interior_envelope_extremum_r_max, 0.48)
        self.assertEqual(args.interior_envelope_ext_dr_max, 0.015)
        self.assertEqual(args.interior_envelope_ext_df_gap_min, 0.005)
        self.assertEqual(args.interior_envelope_ext_df_gap_max, 0.035)
        self.assertTrue(args.disable_interior_unresolved_envelope)

    def test_calibration_cli_accepts_harmonic_incoherence_settings(self):
        args = parse_args(
            [
                "--shot_dir",
                "/tmp/shot",
                "--out_dir",
                "/tmp/out",
                "--interior_harmonic_core_r_max",
                "0.45",
                "--interior_harmonic_active_core_energy_fraction_min",
                "0.01",
                "--interior_harmonic_max_lag_grid",
                "3",
                "--interior_harmonic_incoherence_score_threshold",
                "0.2",
                "--interior_harmonic_calibrated_n_radial",
                "301",
                "--disable_interior_harmonic_incoherence",
            ]
        )

        self.assertEqual(args.interior_harmonic_core_r_max, 0.45)
        self.assertEqual(
            args.interior_harmonic_active_core_energy_fraction_min, 0.01
        )
        self.assertEqual(args.interior_harmonic_max_lag_grid, 3)
        self.assertEqual(
            args.interior_harmonic_incoherence_score_threshold, 0.2
        )
        self.assertEqual(args.interior_harmonic_calibrated_n_radial, 301)
        self.assertTrue(args.disable_interior_harmonic_incoherence)

    def test_complete_output_contract_summary_counts_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            out_dir = root / "out"
            first = run_shot(
                shot,
                out_dir,
                grid_scale_packet_amplitude_min=None,
            )
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
            self.assertEqual(
                first.summary["grid_scale_spike_high_r_cutoff_r"], 0.7
            )
            self.assertEqual(
                first.summary["grid_scale_spike_high_r_width_max_grid"],
                0.75,
            )
            self.assertFalse(first.summary["grid_scale_packet_gate_enabled"])
            self.assertIsNone(
                first.summary["grid_scale_packet_amplitude_min"]
            )
            self.assertEqual(first.summary["grid_scale_packet_step_min"], 0.2)
            self.assertEqual(
                first.summary["grid_scale_packet_min_large_turns"], 3
            )
            self.assertEqual(
                first.summary["grid_scale_packet_window_span_grid"], 4
            )
            self.assertEqual(
                first.summary["grid_scale_packet_peak_r_max"], 0.5
            )
            self.assertTrue(first.summary["continuum_crossing_gate_enabled"])
            self.assertEqual(first.summary["continuum_crossing_w_threshold"], 0.03)
            self.assertTrue(
                first.summary["continuum_crossing_window_gate_enabled"]
            )
            self.assertEqual(
                first.summary["continuum_crossing_window_half_width_grid"],
                2,
            )
            self.assertEqual(
                first.summary["continuum_crossing_window_amplitude_min"],
                0.25,
            )
            self.assertEqual(
                first.summary["continuum_crossing_window_w_min"],
                0.05,
            )
            self.assertTrue(first.summary["edge_artifact_gate_enabled"])
            self.assertEqual(first.summary["edge_artifact_r_min"], 0.97)
            self.assertEqual(first.summary["edge_artifact_width_max_grid"], 10.0)
            self.assertTrue(first.summary["interior_envelope_gate_enabled"])
            self.assertEqual(first.summary["interior_envelope_peak_r_max"], 0.5)
            self.assertEqual(first.summary["interior_envelope_width_max_grid"], 2.0)
            self.assertEqual(
                first.summary["interior_envelope_extremum_r_min"], 0.03
            )
            self.assertEqual(
                first.summary["interior_envelope_extremum_r_max"], 0.5
            )
            self.assertEqual(first.summary["interior_envelope_ext_dr_max"], 0.02)
            self.assertEqual(
                first.summary["interior_envelope_ext_df_gap_min"], 0.0
            )
            self.assertEqual(
                first.summary["interior_envelope_ext_df_gap_max"], 0.04
            )
            self.assertTrue(
                first.summary["interior_harmonic_incoherence_gate_enabled"]
            )
            self.assertEqual(first.summary["interior_harmonic_core_r_max"], 0.5)
            self.assertEqual(
                first.summary[
                    "interior_harmonic_active_core_energy_fraction_min"
                ],
                0.005,
            )
            self.assertEqual(first.summary["interior_harmonic_max_lag_grid"], 5)
            self.assertEqual(
                first.summary[
                    "interior_harmonic_incoherence_score_threshold"
                ],
                0.1,
            )
            self.assertEqual(
                first.summary["interior_harmonic_calibrated_n_radial"], 201
            )
            self.assertEqual(
                first.summary["n_interior_harmonic_resolution_eligible"], 0
            )
            self.assertEqual(
                first.summary["n_interior_harmonic_resolution_ineligible"], 1
            )
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
            _fields, summary_by_n = read_dict_csv(out_dir / "shot_summary_by_n.csv")
            self.assertEqual(len(summary_by_n), 1)
            self.assertEqual(
                summary_by_n[0]["interior_envelope_gate_enabled"], "True"
            )
            self.assertEqual(
                float(summary_by_n[0]["interior_envelope_width_max_grid"]),
                2.0,
            )
            self.assertEqual(
                summary_by_n[0]["interior_harmonic_incoherence_gate_enabled"],
                "True",
            )
            self.assertEqual(
                int(summary_by_n[0]["interior_harmonic_calibrated_n_radial"]),
                201,
            )
            self.assertEqual(
                int(summary_by_n[0]["n_interior_harmonic_resolution_eligible"]),
                0,
            )
            self.assertEqual(
                int(summary_by_n[0]["n_interior_harmonic_resolution_ineligible"]),
                1,
            )

            before = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
            run_shot(
                shot,
                out_dir,
                grid_scale_packet_amplitude_min=None,
            )
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
        self.assertEqual(
            result.summary["n_interior_harmonic_resolution_eligible"], 0
        )
        self.assertEqual(
            result.summary["n_interior_harmonic_resolution_ineligible"], 1
        )
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
                grid_scale_high_r_cutoff_r=0.8,
                grid_scale_high_r_width_max_grid=0.7,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(result.summary["grid_scale_spike_gate_enabled"])
        self.assertEqual(result.summary["grid_scale_spike_amplitude_min"], 0.6)
        self.assertEqual(result.summary["grid_scale_spike_width_max_grid"], 1.0)
        self.assertEqual(
            result.summary["grid_scale_spike_high_r_cutoff_r"], 0.8
        )
        self.assertEqual(
            result.summary["grid_scale_spike_high_r_width_max_grid"], 0.7
        )
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_GRID_SCALE_SPIKE,
        )

    def test_run_shot_applies_packet_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = np.zeros((4, 101), dtype=float)
            mode[2, 48:53] = [0.0, 0.4, 0.0, 0.8, 0.0]
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
                grid_scale_amplitude_min=None,
                grid_scale_packet_amplitude_min=0.7,
                grid_scale_packet_step_min=0.3,
                grid_scale_packet_min_large_turns=3,
                grid_scale_packet_window_span_grid=4,
                grid_scale_packet_peak_r_max=0.6,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(result.summary["grid_scale_packet_gate_enabled"])
        self.assertEqual(
            result.summary["grid_scale_packet_amplitude_min"], 0.7
        )
        self.assertEqual(result.summary["grid_scale_packet_step_min"], 0.3)
        self.assertEqual(
            result.summary["grid_scale_packet_min_large_turns"], 3
        )
        self.assertEqual(
            result.summary["grid_scale_packet_window_span_grid"], 4
        )
        self.assertEqual(result.summary["grid_scale_packet_peak_r_max"], 0.6)
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_GRID_SCALE_PACKET,
        )

    def test_run_shot_applies_near_axis_oscillation_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = np.zeros((4, 201), dtype=float)
            mode[0, 19] = 0.2
            mode[2, 2:7] = [0.075, -0.075, 0.075, -0.075, 0.075]
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=201,
                mode=mode,
            )
            write_datcon(shot / "N1" / "datcon1", nr=201)
            result = run_shot(
                shot,
                root / "out",
                axis_amplitude_min=None,
                grid_scale_amplitude_min=None,
                grid_scale_packet_amplitude_min=None,
                near_axis_grid_oscillation_peak_r_max=0.1,
                near_axis_grid_oscillation_amplitude_min=0.15,
                near_axis_grid_oscillation_min_consecutive_sign_flips=4,
                near_axis_grid_oscillation_step_l2_min=0.3,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(
            result.summary["near_axis_grid_oscillation_gate_enabled"]
        )
        self.assertEqual(
            result.summary["near_axis_grid_oscillation_peak_r_max"], 0.1
        )
        self.assertEqual(
            result.summary["near_axis_grid_oscillation_amplitude_min"], 0.15
        )
        self.assertEqual(
            result.summary[
                "near_axis_grid_oscillation_min_consecutive_sign_flips"
            ],
            4,
        )
        self.assertEqual(
            result.summary["near_axis_grid_oscillation_step_l2_min"], 0.3
        )
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_NEAR_AXIS_GRID_OSCILLATION,
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

    def test_run_shot_applies_cross_window_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            radial_grid = np.linspace(0.0, 1.0, 101)
            mode = np.zeros((4, radial_grid.size), dtype=float)
            mode[0, 60] = 1.0
            mode[1, 40] = 0.01
            mode[1, 42] = 0.3
            upper = np.full(radial_grid.size, 1.1)
            upper[:40] = 0.9
            upper[40] = 1.0
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=radial_grid.size,
                mode=mode,
            )
            write_datcon(
                shot / "N1" / "datcon1",
                nr=radial_grid.size,
                upper_frequency=upper,
            )
            result = run_shot(
                shot,
                root / "out",
                grid_scale_amplitude_min=None,
                w_cross_threshold=0.03,
                cross_window_half_width_grid=2,
                cross_window_amplitude_min=0.25,
                cross_window_w_min=0.05,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(
            result.summary["continuum_crossing_window_gate_enabled"]
        )
        self.assertEqual(
            result.summary["continuum_crossing_window_half_width_grid"],
            2,
        )
        self.assertEqual(
            result.summary["continuum_crossing_window_amplitude_min"],
            0.25,
        )
        self.assertEqual(
            result.summary["continuum_crossing_window_w_min"],
            0.05,
        )
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_CONT_CROSS_WINDOW,
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
                grid_scale_packet_amplitude_min=None,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertTrue(result.summary["edge_artifact_gate_enabled"])
        self.assertEqual(result.summary["edge_artifact_r_min"], 0.97)
        self.assertEqual(result.summary["edge_artifact_width_max_grid"], 4.0)
        self.assertEqual(
            result.final_rows[0]["rule_primary_reason"],
            BAD_EDGE_SPIKE,
        )

    def test_run_shot_applies_interior_envelope_gate_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = narrow_total_energy_mode()
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=mode.shape[1],
                mode=mode,
            )
            write_datcon(
                shot / "N1" / "datcon1",
                nr=mode.shape[1],
            )
            result = run_shot(
                shot,
                root / "out",
                interior_envelope_peak_r_max=0.5,
                interior_envelope_width_max_grid=2.0,
                interior_envelope_extremum_r_min=0.03,
                interior_envelope_extremum_r_max=0.5,
                interior_envelope_ext_dr_max=0.02,
                interior_envelope_ext_df_gap_min=0.0,
                interior_envelope_ext_df_gap_max=0.04,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertEqual(result.summary["n_final_bad"], 1)
        self.assertTrue(result.summary["interior_envelope_gate_enabled"])
        self.assertEqual(result.summary["interior_envelope_peak_r_max"], 0.5)
        self.assertEqual(result.summary["interior_envelope_width_max_grid"], 2.0)
        self.assertEqual(result.summary["interior_envelope_extremum_r_min"], 0.03)
        self.assertEqual(result.summary["interior_envelope_extremum_r_max"], 0.5)
        self.assertEqual(result.summary["interior_envelope_ext_dr_max"], 0.02)
        self.assertEqual(result.summary["interior_envelope_ext_df_gap_min"], 0.0)
        self.assertEqual(result.summary["interior_envelope_ext_df_gap_max"], 0.04)
        self.assertEqual(
            json.loads(result.summary["primary_reason_counts_json"]),
            {BAD_INTERIOR_UNRESOLVED_ENVELOPE: 1},
        )
        row = result.final_rows[0]
        self.assertEqual(row["rule_primary_reason"], BAD_INTERIOR_UNRESOLVED_ENVELOPE)
        features = json.loads(row["rule_features"])["resolution_features"][
            "interior_unresolved_envelope"
        ]
        self.assertTrue(features["candidate_found"])
        self.assertAlmostEqual(features["energy_peak_r"], 0.5)
        self.assertAlmostEqual(features["energy_halfmax_width_grid"], 2.0)

    def test_run_shot_applies_interior_harmonic_incoherence_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            mode = broadband_incoherent_mode(n_harmonics=4)
            write_mode(
                shot / "N1" / "egn01w.one",
                omega=1.0,
                ntor=1,
                nr=mode.shape[1],
                mode=mode,
            )
            write_datcon(
                shot / "N1" / "datcon1",
                nr=mode.shape[1],
            )
            result = run_shot(
                shot,
                root / "out",
                axis_amplitude_min=None,
                axis_width_max_grid=None,
                grid_scale_amplitude_min=None,
                grid_scale_width_max_grid=None,
                grid_scale_packet_amplitude_min=None,
                near_axis_grid_oscillation_amplitude_min=None,
                w_cross_threshold=None,
                cross_window_amplitude_min=None,
                cross_window_w_min=None,
                edge_width_max_grid=None,
                interior_envelope_width_max_grid=None,
            )

        self.assertEqual(result.summary["n_preliminary_bad"], 1)
        self.assertEqual(result.summary["n_final_bad"], 1)
        self.assertEqual(
            result.summary["n_interior_harmonic_resolution_eligible"], 1
        )
        self.assertEqual(
            result.summary["n_interior_harmonic_resolution_ineligible"], 0
        )
        self.assertEqual(
            json.loads(result.summary["primary_reason_counts_json"]),
            {BAD_INTERIOR_HARMONIC_INCOHERENCE: 1},
        )
        row = result.final_rows[0]
        self.assertEqual(
            row["rule_primary_reason"], BAD_INTERIOR_HARMONIC_INCOHERENCE
        )
        features = json.loads(row["rule_features"])[
            "numerical_structure_features"
        ]["interior_harmonic_incoherence"]
        self.assertTrue(features["resolution_eligible"])
        self.assertTrue(features["candidate_found"])
        self.assertGreater(features["incoherence_score"], 0.1)

    def test_valid_override_hash_and_stale_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            initial_out = root / "initial"
            run_shot(
                shot,
                initial_out,
                grid_scale_packet_amplitude_min=None,
            )
            _fields, rows = read_dict_csv(initial_out / "rule_results.csv")
            override_path = root / "overrides.csv"
            write_dict_csv(override_path, MANUAL_OVERRIDE_FIELDS, [override_row(rows[0])])

            final_out = root / "final"
            result = run_shot(
                shot,
                final_out,
                manual_overrides=override_path,
                rf_model=None,
                grid_scale_packet_amplitude_min=None,
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
                shot,
                stale_mode_out,
                manual_overrides=override_path,
                grid_scale_packet_amplitude_min=None,
            )
            self.assertEqual(stale_mode_result.final_rows[0]["final_decision"], "REVIEW")
            self.assertEqual(stale_mode_result.summary["n_stale_overrides"], 1)

            mode_path.write_bytes(original_mode)
            datcon = shot / "N1" / "datcon1"
            datcon.write_text(datcon.read_text() + "\n")
            stale_datcon_out = root / "stale_datcon"
            stale_datcon_result = run_shot(
                shot,
                stale_datcon_out,
                manual_overrides=override_path,
                grid_scale_packet_amplitude_min=None,
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
