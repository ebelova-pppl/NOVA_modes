#!/usr/bin/env python3
"""Pure per-mode interface for deterministic NOVA TAE rule evaluation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cont_features import continuum_extremum_features
from mode_features import (
    EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES,
    EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES,
    RF_FEATURE_NAMES,
    compute_named_features_for_mode,
    get_feature_names,
    get_feature_schema_version,
)
from tae_rule_io import empty_rule_row, stable_json


DEFAULT_AXIS_R_AX = 0.03
DEFAULT_AXIS_AMPLITUDE_MIN = 0.2
DEFAULT_AXIS_WIDTH_MAX_GRID = 10.0
DEFAULT_GRID_SCALE_AMPLITUDE_MIN = 0.3
DEFAULT_GRID_SCALE_WIDTH_MAX_GRID = 1.0
DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R = 0.7
DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID = 0.75
DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN = 0.3
DEFAULT_GRID_SCALE_PACKET_STEP_MIN = 0.2
DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS = 3
DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID = 4
DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX = 0.5
DEFAULT_W_CROSS_THRESHOLD = 0.03
DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID = 2
DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN = 0.25
DEFAULT_CROSS_WINDOW_W_MIN = 0.05
DEFAULT_EDGE_R_MIN = 0.97
DEFAULT_EDGE_WIDTH_MAX_GRID = 10.0
DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX = 0.5
DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID = 2.0
DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN = 0.03
DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX = 0.50
DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX = 0.02
DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN = 0.0
DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX = 0.04

RULESET_VERSION = (
    "tae-rules-axis-all-peaks-grid-highr-packet-turns-rle05-cont-window-"
    "edge-interior-envelope-v15"
)
BAD_AXIS_SPIKE = "BAD_AXIS_SPIKE"
BAD_GRID_SCALE_SPIKE = "BAD_GRID_SCALE_SPIKE"
BAD_GRID_SCALE_PACKET = "BAD_GRID_SCALE_PACKET"
BAD_CONT_CROSS = "BAD_CONT_CROSS"
BAD_CONT_CROSS_WINDOW = "BAD_CONT_CROSS_WINDOW"
BAD_EDGE_SPIKE = "BAD_EDGE_SPIKE"
BAD_INTERIOR_UNRESOLVED_ENVELOPE = "BAD_INTERIOR_UNRESOLVED_ENVELOPE"
NO_GOOD_TEMPLATE = "NO_GOOD_TEMPLATE"
RULE_FEATURE_EXTRACTION_FAILED = "RULE_FEATURE_EXTRACTION_FAILED"
RULE_FEATURE_NAMES = tuple(
    get_feature_names(include_crossing_features=True, include_extremum_features=True)
)
RULE_FEATURE_SCHEMA_VERSION = "tae-rule-features-grouped-v14"
RULE_FEATURE_SOURCE_SCHEMA_VERSION = get_feature_schema_version(
    include_crossing_features=True,
    include_extremum_features=True,
)
RULE_FEATURE_METADATA_NAMES = (
    "feature_schema_version",
    "source_feature_schema_version",
)
RULE_FEATURE_GROUP_NAMES = (
    "rf_standard_features",
    "resolution_features",
    "numerical_structure_features",
    "crossing_features",
    "crossing_records",
    "extremum_features",
    "boundary_features",
)


@dataclass(frozen=True)
class AxisArtifactConfig:
    """Thresholds for the near-axis narrow-spike rejection gate."""

    r_ax: float = DEFAULT_AXIS_R_AX
    axis_amplitude_min: float | None = DEFAULT_AXIS_AMPLITUDE_MIN
    axis_width_max_grid: float | None = DEFAULT_AXIS_WIDTH_MAX_GRID

    def __post_init__(self) -> None:
        if not math.isfinite(self.r_ax) or not 0.0 < self.r_ax <= 1.0:
            raise ValueError("axis r_ax must be finite and in (0, 1]")
        if self.axis_amplitude_min is not None and (
            not math.isfinite(self.axis_amplitude_min)
            or not 0.0 <= self.axis_amplitude_min <= 1.0
        ):
            raise ValueError(
                "axis_amplitude_min must be null or finite and in [0, 1]"
            )
        if self.axis_width_max_grid is not None and (
            not math.isfinite(self.axis_width_max_grid)
            or self.axis_width_max_grid < 0.0
        ):
            raise ValueError(
                "axis_width_max_grid must be null or a finite nonnegative number"
            )

    @property
    def enabled(self) -> bool:
        """Return whether both thresholds required by the gate are configured."""
        return (
            self.axis_amplitude_min is not None
            and self.axis_width_max_grid is not None
        )


@dataclass(frozen=True)
class GridScaleSpikeConfig:
    """Thresholds for the radius-dependent unresolved signed-lobe gate."""

    amplitude_min: float | None = DEFAULT_GRID_SCALE_AMPLITUDE_MIN
    width_max_grid: float | None = DEFAULT_GRID_SCALE_WIDTH_MAX_GRID
    high_r_cutoff_r: float = DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R
    high_r_width_max_grid: float | None = (
        DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID
    )

    def __post_init__(self) -> None:
        if self.amplitude_min is not None and (
            not math.isfinite(self.amplitude_min)
            or not 0.0 <= self.amplitude_min <= 1.0
        ):
            raise ValueError(
                "grid_scale amplitude_min must be null or finite and in [0, 1]"
            )
        if self.width_max_grid is not None and (
            not math.isfinite(self.width_max_grid)
            or self.width_max_grid < 0.0
        ):
            raise ValueError(
                "grid_scale width_max_grid must be null or a finite "
                "nonnegative number"
            )
        if not math.isfinite(self.high_r_cutoff_r) or not (
            0.0 <= self.high_r_cutoff_r <= 1.0
        ):
            raise ValueError(
                "grid_scale high_r_cutoff_r must be finite and in [0, 1]"
            )
        if self.high_r_width_max_grid is not None and (
            not math.isfinite(self.high_r_width_max_grid)
            or self.high_r_width_max_grid < 0.0
        ):
            raise ValueError(
                "grid_scale high_r_width_max_grid must be null or a finite "
                "nonnegative number"
            )

    @property
    def enabled(self) -> bool:
        """Return whether amplitude and at least one radial width are configured."""
        return self.amplitude_min is not None and (
            self.width_max_grid is not None
            or self.high_r_width_max_grid is not None
        )


@dataclass(frozen=True)
class GridScalePacketConfig:
    """Thresholds for repeated large turning points in a harmonic window."""

    amplitude_min: float | None = DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN
    step_min: float = DEFAULT_GRID_SCALE_PACKET_STEP_MIN
    min_large_turns: int = DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS
    window_span_grid: int = DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID
    peak_r_max: float | None = DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX

    def __post_init__(self) -> None:
        if self.amplitude_min is not None and (
            not math.isfinite(self.amplitude_min)
            or not 0.0 <= self.amplitude_min <= 1.0
        ):
            raise ValueError(
                "grid_scale_packet amplitude_min must be null or finite "
                "and in [0, 1]"
            )
        if not math.isfinite(self.step_min) or not 0.0 <= self.step_min <= 2.0:
            raise ValueError(
                "grid_scale_packet step_min must be finite and in [0, 2]"
            )
        if self.peak_r_max is not None and (
            not math.isfinite(self.peak_r_max)
            or not 0.0 <= self.peak_r_max <= 1.0
        ):
            raise ValueError(
                "grid_scale_packet peak_r_max must be null or finite and "
                "in [0, 1]"
            )
        for name, value in (
            ("min_large_turns", self.min_large_turns),
            ("window_span_grid", self.window_span_grid),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"grid_scale_packet {name} must be an integer")
            if value < 1:
                raise ValueError(f"grid_scale_packet {name} must be positive")
        if self.window_span_grid < 2:
            raise ValueError(
                "grid_scale_packet window_span_grid must be at least 2"
            )
        if self.min_large_turns > self.window_span_grid - 1:
            raise ValueError(
                "grid_scale_packet min_large_turns cannot exceed the number "
                "of interior samples (window_span_grid - 1)"
            )

    @property
    def enabled(self) -> bool:
        """Return whether the packet amplitude threshold is configured."""
        return self.amplitude_min is not None


@dataclass(frozen=True)
class ContinuumCrossingConfig:
    """Threshold for the significant continuum-crossing rejection gate."""

    w_cross_threshold: float | None = DEFAULT_W_CROSS_THRESHOLD

    def __post_init__(self) -> None:
        if self.w_cross_threshold is not None and (
            not math.isfinite(self.w_cross_threshold)
            or not 0.0 <= self.w_cross_threshold <= 1.0
        ):
            raise ValueError(
                "w_cross_threshold must be null or finite and in [0, 1]"
            )

    @property
    def enabled(self) -> bool:
        """Return whether the crossing-energy threshold is configured."""
        return self.w_cross_threshold is not None


@dataclass(frozen=True)
class ContinuumCrossingWindowConfig:
    """Thresholds for crossing-neighborhood amplitude and energy."""

    half_width_grid: int = DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID
    amplitude_min: float | None = DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN
    w_min: float | None = DEFAULT_CROSS_WINDOW_W_MIN

    def __post_init__(self) -> None:
        if isinstance(self.half_width_grid, bool) or not isinstance(
            self.half_width_grid, int
        ):
            raise ValueError("cross_window half_width_grid must be an integer")
        if self.half_width_grid < 0:
            raise ValueError("cross_window half_width_grid must be nonnegative")
        for name, value in (
            ("amplitude_min", self.amplitude_min),
            ("w_min", self.w_min),
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"cross_window {name} must be null or finite and in [0, 1]"
                )
    @property
    def enabled(self) -> bool:
        """Return whether either crossing-neighborhood threshold is configured."""
        return self.amplitude_min is not None or self.w_min is not None


@dataclass(frozen=True)
class EdgeArtifactConfig:
    """Thresholds for the narrow global-energy edge-spike rejection gate."""

    r_edge_min: float = DEFAULT_EDGE_R_MIN
    edge_width_max_grid: float | None = DEFAULT_EDGE_WIDTH_MAX_GRID

    def __post_init__(self) -> None:
        if not math.isfinite(self.r_edge_min) or not 0.0 <= self.r_edge_min < 1.0:
            raise ValueError("edge r_edge_min must be finite and in [0, 1)")
        if self.edge_width_max_grid is not None and (
            not math.isfinite(self.edge_width_max_grid)
            or self.edge_width_max_grid < 0.0
        ):
            raise ValueError(
                "edge_width_max_grid must be null or a finite nonnegative number"
            )

    @property
    def enabled(self) -> bool:
        """Return whether the edge-energy width threshold is configured."""
        return self.edge_width_max_grid is not None


@dataclass(frozen=True)
class InteriorUnresolvedEnvelopeConfig:
    """Thresholds and extremum exception for an unresolved interior W envelope."""

    peak_r_max: float = DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX
    width_max_grid: float | None = DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID
    extremum_r_min: float = DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN
    extremum_r_max: float = DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX
    ext_dr_max: float = DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX
    ext_df_gap_min: float = DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN
    ext_df_gap_max: float = DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX

    def __post_init__(self) -> None:
        if not math.isfinite(self.peak_r_max) or not 0.0 <= self.peak_r_max <= 1.0:
            raise ValueError(
                "interior envelope peak_r_max must be finite and in [0, 1]"
            )
        if self.width_max_grid is not None and (
            not math.isfinite(self.width_max_grid) or self.width_max_grid < 0.0
        ):
            raise ValueError(
                "interior envelope width_max_grid must be null or a finite "
                "nonnegative number"
            )
        if not (
            math.isfinite(self.extremum_r_min)
            and math.isfinite(self.extremum_r_max)
            and 0.0 <= self.extremum_r_min < self.extremum_r_max <= 1.0
        ):
            raise ValueError(
                "interior envelope extremum interval must be finite and satisfy "
                "0 <= extremum_r_min < extremum_r_max <= 1"
            )
        if not math.isfinite(self.ext_dr_max) or not 0.0 <= self.ext_dr_max <= 1.0:
            raise ValueError(
                "interior envelope ext_dr_max must be finite and in [0, 1]"
            )
        if not (
            math.isfinite(self.ext_df_gap_min)
            and math.isfinite(self.ext_df_gap_max)
            and self.ext_df_gap_min <= self.ext_df_gap_max
        ):
            raise ValueError(
                "interior envelope frequency-gap limits must be finite and satisfy "
                "ext_df_gap_min <= ext_df_gap_max"
            )

    @property
    def enabled(self) -> bool:
        """Return whether the connected-energy width threshold is configured."""
        return self.width_max_grid is not None


def empty_axis_artifact_features(
    r_ax: float = DEFAULT_AXIS_R_AX,
    amplitude_min: float | None = DEFAULT_AXIS_AMPLITUDE_MIN,
    width_max_grid: float | None = DEFAULT_AXIS_WIDTH_MAX_GRID,
) -> dict[str, Any]:
    """Return the stable axis-artifact feature shape with null measurements."""
    return {
        "r_ax": r_ax,
        "axis_candidate_found": None,
        "axis_candidate_amplitude_min": amplitude_min,
        "axis_candidate_width_limit_grid": width_max_grid,
        "axis_local_peak_count": None,
        "axis_amplitude_qualified_peak_count": None,
        "axis_width_qualified_peak_count": None,
        "axis_peak": None,
        "axis_peak_harmonic_index": None,
        "axis_peak_r": None,
        "axis_peak_is_local_max": None,
        "axis_halfmax_width_r": None,
        "axis_halfmax_width_grid": None,
        "axis_halfmax_outer_edge_r": None,
        "axis_component_touches_boundary": None,
    }


def empty_grid_scale_spike_features(
    width_max_grid: float | None = DEFAULT_GRID_SCALE_WIDTH_MAX_GRID,
    *,
    high_r_cutoff_r: float = DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R,
    high_r_width_max_grid: float | None = (
        DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID
    ),
    candidate_found: bool | None = None,
) -> dict[str, Any]:
    """Return the stable grid-scale-spike shape with null measurements."""
    return {
        "grid_scale_candidate_found": candidate_found,
        "grid_scale_width_max_grid": width_max_grid,
        "grid_scale_high_r_cutoff_r": high_r_cutoff_r,
        "grid_scale_high_r_width_max_grid": high_r_width_max_grid,
        "grid_scale_candidate_width_limit_grid": None,
        "grid_scale_peak": None,
        "grid_scale_peak_signed_amplitude": None,
        "grid_scale_peak_sign": None,
        "grid_scale_peak_harmonic_index": None,
        "grid_scale_peak_r": None,
        "grid_scale_halfmax_width_r": None,
        "grid_scale_halfmax_width_grid": None,
        "grid_scale_halfmax_inner_edge_r": None,
        "grid_scale_halfmax_outer_edge_r": None,
        "grid_scale_component_touches_boundary": None,
    }


def empty_grid_scale_packet_features(
    amplitude_min: float | None = DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN,
    step_min: float = DEFAULT_GRID_SCALE_PACKET_STEP_MIN,
    min_large_turns: int = DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS,
    window_span_grid: int = DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID,
    peak_r_max: float | None = DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX,
    *,
    candidate_found: bool | None = None,
    turn_qualified_window_count: int | None = None,
    radius_qualified_window_count: int | None = None,
    amplitude_qualified_window_count: int | None = None,
) -> dict[str, Any]:
    """Return the stable grid-scale-packet shape with null measurements."""
    return {
        "grid_scale_packet_candidate_found": candidate_found,
        "grid_scale_packet_amplitude_min": amplitude_min,
        "grid_scale_packet_step_min": step_min,
        "grid_scale_packet_min_large_turns": min_large_turns,
        "grid_scale_packet_window_span_grid": window_span_grid,
        "grid_scale_packet_peak_r_max": peak_r_max,
        "grid_scale_packet_turn_qualified_window_count": (
            turn_qualified_window_count
        ),
        "grid_scale_packet_radius_qualified_window_count": (
            radius_qualified_window_count
        ),
        "grid_scale_packet_amplitude_qualified_window_count": (
            amplitude_qualified_window_count
        ),
        "grid_scale_packet_peak": None,
        "grid_scale_packet_peak_signed_amplitude": None,
        "grid_scale_packet_peak_harmonic_index": None,
        "grid_scale_packet_peak_r": None,
        "grid_scale_packet_window_start_index": None,
        "grid_scale_packet_window_end_index": None,
        "grid_scale_packet_window_start_r": None,
        "grid_scale_packet_window_end_r": None,
        "grid_scale_packet_large_step_count": None,
        "grid_scale_packet_large_turn_count": None,
        "grid_scale_packet_max_step": None,
        "grid_scale_packet_step_rms": None,
        "grid_scale_packet_total_variation": None,
        "grid_scale_packet_direction_change_count": None,
        "grid_scale_packet_sign_change_count": None,
        "grid_scale_packet_window_values": None,
    }


def empty_edge_artifact_features(
    r_edge_min: float = DEFAULT_EDGE_R_MIN,
) -> dict[str, Any]:
    """Return the stable edge-artifact feature shape with null measurements."""
    return {
        "r_edge_min": r_edge_min,
        "edge_energy_peak": None,
        "edge_energy_peak_r": None,
        "edge_energy_peak_in_window": None,
        "edge_energy_halfmax_width_r": None,
        "edge_energy_halfmax_width_grid": None,
        "edge_energy_halfmax_inner_edge_r": None,
        "edge_energy_halfmax_outer_edge_r": None,
        "edge_energy_component_touches_boundary": None,
        "edge_harmonic_peak": None,
        "edge_harmonic_peak_harmonic_index": None,
        "edge_harmonic_peak_r": None,
        "edge_harmonic_peak_is_local_max": None,
        "edge_harmonic_halfmax_width_r": None,
        "edge_harmonic_halfmax_width_grid": None,
        "edge_harmonic_halfmax_inner_edge_r": None,
        "edge_harmonic_halfmax_outer_edge_r": None,
        "edge_harmonic_component_touches_boundary": None,
    }


def empty_interior_unresolved_envelope_features(
    config: InteriorUnresolvedEnvelopeConfig | None = None,
) -> dict[str, Any]:
    """Return the stable unresolved-interior-envelope shape with null evidence."""
    resolved = config or InteriorUnresolvedEnvelopeConfig()
    return {
        "peak_r_max": resolved.peak_r_max,
        "width_max_grid": resolved.width_max_grid,
        "extremum_r_min": resolved.extremum_r_min,
        "extremum_r_max": resolved.extremum_r_max,
        "ext_dr_max": resolved.ext_dr_max,
        "ext_df_gap_min": resolved.ext_df_gap_min,
        "ext_df_gap_max": resolved.ext_df_gap_max,
        "candidate_found": None,
        "energy_peak": None,
        "energy_peak_r": None,
        "energy_halfmax_width_r": None,
        "energy_halfmax_width_grid": None,
        "energy_halfmax_inner_edge_r": None,
        "energy_halfmax_outer_edge_r": None,
        "energy_component_touches_boundary": None,
        "extremum_match_found": None,
        "ext_dr": None,
        "ext_df_gap": None,
        "ext_energy_frac": None,
        "extremum_exception_applied": None,
    }


def empty_continuum_crossing_window_features(
    half_width_grid: int = DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID,
    *,
    candidate_found: bool | None = None,
) -> dict[str, Any]:
    """Return the stable crossing-neighborhood shape with null measurements."""
    return {
        "cross_window_candidate_found": candidate_found,
        "cross_window_half_width_grid": half_width_grid,
        "cross_window_half_width_r": None,
        "cross_window_A_max": None,
        "cross_window_A_harmonic_index": None,
        "cross_window_A_sample_r": None,
        "cross_window_A_crossing_boundary": None,
        "cross_window_A_crossing_r": None,
        "cross_window_A_distance_grid": None,
        "cross_window_A_neighbor_rms": None,
        "cross_window_A_neighbor_count": None,
        "cross_window_A_neighbor_stencil_complete": None,
        "cross_window_W_max": None,
        "cross_window_W_sample_r": None,
        "cross_window_W_crossing_boundary": None,
        "cross_window_W_crossing_r": None,
        "cross_window_W_distance_grid": None,
    }


def empty_rule_features(
    axis_artifact_config: AxisArtifactConfig | None = None,
    grid_scale_spike_config: GridScaleSpikeConfig | None = None,
    edge_artifact_config: EdgeArtifactConfig | None = None,
    continuum_crossing_window_config: ContinuumCrossingWindowConfig | None = None,
    grid_scale_packet_config: GridScalePacketConfig | None = None,
    interior_unresolved_envelope_config: (
        InteriorUnresolvedEnvelopeConfig | None
    ) = None,
) -> dict[str, Any]:
    """Return the complete rule-feature schema with unavailable values as null."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    packet_config = grid_scale_packet_config or GridScalePacketConfig()
    edge_config = edge_artifact_config or EdgeArtifactConfig()
    interior_config = (
        interior_unresolved_envelope_config or InteriorUnresolvedEnvelopeConfig()
    )
    cross_window_config = (
        continuum_crossing_window_config or ContinuumCrossingWindowConfig()
    )
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "rf_standard_features": {name: None for name in RF_FEATURE_NAMES},
        "resolution_features": {
            "interior_unresolved_envelope": (
                empty_interior_unresolved_envelope_features(interior_config)
            ),
        },
        "numerical_structure_features": {
            "grid_scale_spike": empty_grid_scale_spike_features(
                grid_config.width_max_grid,
                high_r_cutoff_r=grid_config.high_r_cutoff_r,
                high_r_width_max_grid=grid_config.high_r_width_max_grid,
            ),
            "grid_scale_packet": empty_grid_scale_packet_features(
                packet_config.amplitude_min,
                packet_config.step_min,
                packet_config.min_large_turns,
                packet_config.window_span_grid,
                packet_config.peak_r_max,
            ),
        },
        "crossing_features": {
            **{name: None for name in EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES},
            **empty_continuum_crossing_window_features(
                cross_window_config.half_width_grid
            ),
        },
        "crossing_records": [],
        "extremum_features": {
            "match_found": None,
            **{name: None for name in EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES},
        },
        "boundary_features": {
            "axis_artifact": empty_axis_artifact_features(
                axis_config.r_ax,
                axis_config.axis_amplitude_min,
                axis_config.axis_width_max_grid,
            ),
            "edge_artifact": empty_edge_artifact_features(
                edge_config.r_edge_min
            ),
        },
    }


def grouped_rule_features(
    named_features: Mapping[str, Any],
    feature_status: Mapping[str, Any],
    axis_artifact_features: Mapping[str, Any],
    grid_scale_spike_features: Mapping[str, Any],
    grid_scale_packet_features: Mapping[str, Any],
    edge_artifact_features: Mapping[str, Any],
    continuum_crossing_window_features: Mapping[str, Any],
    interior_unresolved_envelope_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Organize shared RF31 measurements and deterministic rule evidence."""
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "rf_standard_features": {
            name: named_features[name] for name in RF_FEATURE_NAMES
        },
        "resolution_features": {
            "interior_unresolved_envelope": dict(
                interior_unresolved_envelope_features
            ),
        },
        "numerical_structure_features": {
            "grid_scale_spike": dict(grid_scale_spike_features),
            "grid_scale_packet": dict(grid_scale_packet_features),
        },
        "crossing_features": {
            **{
                name: named_features[name]
                for name in EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES
            },
            **dict(continuum_crossing_window_features),
        },
        "crossing_records": list(feature_status["crossing_records"]),
        "extremum_features": {
            "match_found": feature_status["extremum_match_found"],
            **{
                name: named_features[name]
                for name in EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES
            },
        },
        "boundary_features": {
            "axis_artifact": dict(axis_artifact_features),
            "edge_artifact": dict(edge_artifact_features),
        },
    }


def _interpolate_threshold_crossing(
    r_below: float,
    value_below: float,
    r_above: float,
    value_above: float,
    threshold: float,
) -> float:
    """Linearly locate a threshold between adjacent below/above samples."""
    delta = value_above - value_below
    if delta == 0.0:
        return r_above
    fraction = (threshold - value_below) / delta
    return r_below + fraction * (r_above - r_below)


def _signed_local_extrema(profile: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(index, plateau_left, plateau_right)`` for signed extrema.

    Positive maxima and negative minima are both returned. A boundary sample
    or plateau is accepted using its available one-sided neighbor; the
    leftmost plateau index is the deterministic representative.
    """
    n_radial = profile.size
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(profile)))
    )
    extrema: list[tuple[int, int, int]] = []
    plateau_left = 0
    while plateau_left < n_radial:
        value = float(profile[plateau_left])
        plateau_right = plateau_left
        while (
            plateau_right + 1 < n_radial
            and abs(float(profile[plateau_right + 1]) - value) <= tolerance
        ):
            plateau_right += 1

        if abs(value) > tolerance:
            sign = 1.0 if value > 0.0 else -1.0
            peak = sign * value
            left_is_lower = (
                plateau_left == 0
                or sign * float(profile[plateau_left - 1]) < peak - tolerance
            )
            right_is_lower = (
                plateau_right == n_radial - 1
                or sign * float(profile[plateau_right + 1]) < peak - tolerance
            )
            has_outer_neighbor = plateau_left > 0 or plateau_right < n_radial - 1
            if left_is_lower and right_is_lower and has_outer_neighbor:
                extrema.append((plateau_left, plateau_left, plateau_right))
        plateau_left = plateau_right + 1
    return extrema


def _signed_halfmax_component(
    profile: np.ndarray,
    *,
    peak_index: int,
    radial_grid: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """Measure one signed lobe without joining an adjacent opposite-sign lobe."""
    signed_amplitude = float(profile[peak_index])
    sign = 1.0 if signed_amplitude > 0.0 else -1.0
    signed_profile = sign * profile
    half_maximum = 0.5 * abs(signed_amplitude)

    component_left = peak_index
    while (
        component_left > 0
        and float(signed_profile[component_left - 1]) >= half_maximum
    ):
        component_left -= 1
    component_right = peak_index
    while (
        component_right < profile.size - 1
        and float(signed_profile[component_right + 1]) >= half_maximum
    ):
        component_right += 1

    if component_left == 0:
        inner_edge = 0.0
    else:
        inner_edge = _interpolate_threshold_crossing(
            float(radial_grid[component_left - 1]),
            float(signed_profile[component_left - 1]),
            float(radial_grid[component_left]),
            float(signed_profile[component_left]),
            half_maximum,
        )
    if component_right == profile.size - 1:
        outer_edge = 1.0
    else:
        outer_edge = _interpolate_threshold_crossing(
            float(radial_grid[component_right]),
            float(signed_profile[component_right]),
            float(radial_grid[component_right + 1]),
            float(signed_profile[component_right + 1]),
            half_maximum,
        )

    width_r = max(0.0, outer_edge - inner_edge)
    radial_interval = 1.0 / (profile.size - 1)
    return (
        float(inner_edge),
        float(outer_edge),
        float(width_r),
        float(width_r / radial_interval),
        bool(component_left == 0 or component_right == profile.size - 1),
    )


def extract_grid_scale_spike_features(
    mode: np.ndarray,
    *,
    width_max_grid: float | None = DEFAULT_GRID_SCALE_WIDTH_MAX_GRID,
    high_r_cutoff_r: float = DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R,
    high_r_width_max_grid: float | None = (
        DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID
    ),
) -> dict[str, Any]:
    """Find the strongest signed local lobe within its radial width limit.

    Each positive maximum or negative minimum is measured on its signed
    harmonic profile over the complete radial grid. This deliberately avoids
    ``abs(mode)``, which can join unresolved adjacent ``+A/-A`` lobes. Peaks
    strictly above ``high_r_cutoff_r`` use ``high_r_width_max_grid``; all
    others use ``width_max_grid``.
    """
    if width_max_grid is not None and (
        not math.isfinite(width_max_grid) or width_max_grid < 0.0
    ):
        raise ValueError(
            "grid-scale width_max_grid must be null or a finite nonnegative number"
        )
    if not math.isfinite(high_r_cutoff_r) or not 0.0 <= high_r_cutoff_r <= 1.0:
        raise ValueError(
            "grid-scale high_r_cutoff_r must be finite and in [0, 1]"
        )
    if high_r_width_max_grid is not None and (
        not math.isfinite(high_r_width_max_grid)
        or high_r_width_max_grid < 0.0
    ):
        raise ValueError(
            "grid-scale high_r_width_max_grid must be null or a finite "
            "nonnegative number"
        )
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")
    if width_max_grid is None and high_r_width_max_grid is None:
        return empty_grid_scale_spike_features(
            width_max_grid,
            high_r_cutoff_r=high_r_cutoff_r,
            high_r_width_max_grid=high_r_width_max_grid,
            candidate_found=False,
        )

    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    candidates: list[dict[str, Any]] = []
    cutoff_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(high_r_cutoff_r)
    )
    for harmonic_index, profile in enumerate(mode_array):
        for peak_index, _plateau_left, _plateau_right in _signed_local_extrema(
            profile
        ):
            inner_edge, outer_edge, width_r, width_grid, touches_boundary = (
                _signed_halfmax_component(
                    profile,
                    peak_index=peak_index,
                    radial_grid=radial_grid,
                )
            )
            peak_r = float(radial_grid[peak_index])
            candidate_width_limit = (
                high_r_width_max_grid
                if peak_r > high_r_cutoff_r + cutoff_tolerance
                else width_max_grid
            )
            if candidate_width_limit is None:
                continue
            width_tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, candidate_width_limit
            )
            if width_grid > candidate_width_limit + width_tolerance:
                continue
            signed_amplitude = float(profile[peak_index])
            candidates.append(
                {
                    "grid_scale_candidate_found": True,
                    "grid_scale_width_max_grid": width_max_grid,
                    "grid_scale_high_r_cutoff_r": high_r_cutoff_r,
                    "grid_scale_high_r_width_max_grid": high_r_width_max_grid,
                    "grid_scale_candidate_width_limit_grid": float(
                        candidate_width_limit
                    ),
                    "grid_scale_peak": abs(signed_amplitude),
                    "grid_scale_peak_signed_amplitude": signed_amplitude,
                    "grid_scale_peak_sign": 1 if signed_amplitude > 0.0 else -1,
                    "grid_scale_peak_harmonic_index": int(harmonic_index),
                    "grid_scale_peak_r": peak_r,
                    "grid_scale_halfmax_width_r": width_r,
                    "grid_scale_halfmax_width_grid": width_grid,
                    "grid_scale_halfmax_inner_edge_r": inner_edge,
                    "grid_scale_halfmax_outer_edge_r": outer_edge,
                    "grid_scale_component_touches_boundary": touches_boundary,
                }
            )

    if not candidates:
        return empty_grid_scale_spike_features(
            width_max_grid,
            high_r_cutoff_r=high_r_cutoff_r,
            high_r_width_max_grid=high_r_width_max_grid,
            candidate_found=False,
        )
    return min(
        candidates,
        key=lambda candidate: (
            -candidate["grid_scale_peak"],
            candidate["grid_scale_halfmax_width_grid"],
            candidate["grid_scale_peak_harmonic_index"],
            candidate["grid_scale_peak_r"],
        ),
    )


def extract_grid_scale_packet_features(
    mode: np.ndarray,
    *,
    amplitude_min: float | None = DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN,
    step_min: float = DEFAULT_GRID_SCALE_PACKET_STEP_MIN,
    min_large_turns: int = DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS,
    window_span_grid: int = DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID,
    peak_r_max: float | None = DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX,
) -> dict[str, Any]:
    """Find repeated large turning points on one signed harmonic.

    Every complete ``window_span_grid + 1`` sample window is considered. A
    large turn is an interior sample whose two adjacent signed-value steps
    both meet ``step_min`` and have opposing signs. A packet candidate has at
    least ``min_large_turns`` such extrema and its largest absolute sample is
    centered at or below ``peak_r_max`` when that cutoff is configured. The
    selected candidate has the largest absolute amplitude, followed
    deterministically by turn count, step count, total variation, harmonic
    index, and window location.
    """
    config = GridScalePacketConfig(
        amplitude_min=amplitude_min,
        step_min=step_min,
        min_large_turns=min_large_turns,
        window_span_grid=window_span_grid,
        peak_r_max=peak_r_max,
    )
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")
    if mode_array.shape[1] <= config.window_span_grid:
        return empty_grid_scale_packet_features(
            config.amplitude_min,
            config.step_min,
            config.min_large_turns,
            config.window_span_grid,
            config.peak_r_max,
            candidate_found=False,
            turn_qualified_window_count=0,
            radius_qualified_window_count=0,
            amplitude_qualified_window_count=(
                None if config.amplitude_min is None else 0
            ),
        )

    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    step_tolerance = 64.0 * np.finfo(float).eps * max(1.0, config.step_min)
    radius_tolerance = 64.0 * np.finfo(float).eps
    amplitude_tolerance = (
        None
        if config.amplitude_min is None
        else 64.0
        * np.finfo(float).eps
        * max(1.0, config.amplitude_min)
    )
    candidates: list[dict[str, Any]] = []
    turn_qualified_count = 0
    radius_qualified_count = 0
    amplitude_qualified_count = 0
    n_samples = config.window_span_grid + 1
    for harmonic_index, profile in enumerate(mode_array):
        profile_steps = np.diff(profile)
        large_steps = (
            np.abs(profile_steps) >= config.step_min - step_tolerance
        )
        large_turns = (
            large_steps[:-1]
            & large_steps[1:]
            & (profile_steps[:-1] * profile_steps[1:] < 0.0)
        )
        window_turn_counts = np.convolve(
            large_turns.astype(int),
            np.ones(config.window_span_grid - 1, dtype=int),
            mode="valid",
        )
        qualified_starts = np.flatnonzero(
            window_turn_counts >= config.min_large_turns
        )
        for raw_start_index in qualified_starts:
            turn_qualified_count += 1
            start_index = int(raw_start_index)
            end_index = start_index + config.window_span_grid
            window = profile[start_index : start_index + n_samples]
            signed_steps = profile_steps[start_index:end_index]
            step_magnitudes = np.abs(signed_steps)
            large_step_count = int(
                np.count_nonzero(
                    step_magnitudes >= config.step_min - step_tolerance
                )
            )
            large_turn_count = int(window_turn_counts[start_index])

            peak_offset = int(np.argmax(np.abs(window)))
            peak_signed_amplitude = float(window[peak_offset])
            peak = abs(peak_signed_amplitude)
            peak_index = start_index + peak_offset
            peak_r = float(radial_grid[peak_index])
            if (
                config.peak_r_max is not None
                and peak_r > config.peak_r_max + radius_tolerance
            ):
                continue
            radius_qualified_count += 1
            if (
                config.amplitude_min is not None
                and amplitude_tolerance is not None
                and peak >= config.amplitude_min - amplitude_tolerance
            ):
                amplitude_qualified_count += 1
            direction_change_count = int(
                np.count_nonzero(signed_steps[:-1] * signed_steps[1:] < 0.0)
            )
            sign_change_count = int(
                np.count_nonzero(window[:-1] * window[1:] < 0.0)
            )
            candidates.append(
                {
                    "grid_scale_packet_candidate_found": True,
                    "grid_scale_packet_amplitude_min": config.amplitude_min,
                    "grid_scale_packet_step_min": config.step_min,
                    "grid_scale_packet_min_large_turns": (
                        config.min_large_turns
                    ),
                    "grid_scale_packet_window_span_grid": (
                        config.window_span_grid
                    ),
                    "grid_scale_packet_peak_r_max": config.peak_r_max,
                    "grid_scale_packet_turn_qualified_window_count": None,
                    "grid_scale_packet_radius_qualified_window_count": None,
                    "grid_scale_packet_amplitude_qualified_window_count": None,
                    "grid_scale_packet_peak": peak,
                    "grid_scale_packet_peak_signed_amplitude": (
                        peak_signed_amplitude
                    ),
                    "grid_scale_packet_peak_harmonic_index": int(
                        harmonic_index
                    ),
                    "grid_scale_packet_peak_r": peak_r,
                    "grid_scale_packet_window_start_index": int(start_index),
                    "grid_scale_packet_window_end_index": int(end_index),
                    "grid_scale_packet_window_start_r": float(
                        radial_grid[start_index]
                    ),
                    "grid_scale_packet_window_end_r": float(
                        radial_grid[end_index]
                    ),
                    "grid_scale_packet_large_step_count": large_step_count,
                    "grid_scale_packet_large_turn_count": large_turn_count,
                    "grid_scale_packet_max_step": float(
                        np.max(step_magnitudes)
                    ),
                    "grid_scale_packet_step_rms": float(
                        np.sqrt(np.mean(np.square(step_magnitudes)))
                    ),
                    "grid_scale_packet_total_variation": float(
                        np.sum(step_magnitudes)
                    ),
                    "grid_scale_packet_direction_change_count": (
                        direction_change_count
                    ),
                    "grid_scale_packet_sign_change_count": sign_change_count,
                    "grid_scale_packet_window_values": [
                        float(value) for value in window
                    ],
                }
            )

    if not candidates:
        return empty_grid_scale_packet_features(
            config.amplitude_min,
            config.step_min,
            config.min_large_turns,
            config.window_span_grid,
            config.peak_r_max,
            candidate_found=False,
            turn_qualified_window_count=turn_qualified_count,
            radius_qualified_window_count=radius_qualified_count,
            amplitude_qualified_window_count=(
                None if config.amplitude_min is None else 0
            ),
        )

    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate["grid_scale_packet_peak"],
            -candidate["grid_scale_packet_large_turn_count"],
            -candidate["grid_scale_packet_large_step_count"],
            -candidate["grid_scale_packet_total_variation"],
            candidate["grid_scale_packet_peak_harmonic_index"],
            candidate["grid_scale_packet_window_start_index"],
        ),
    )
    selected["grid_scale_packet_turn_qualified_window_count"] = (
        turn_qualified_count
    )
    selected["grid_scale_packet_radius_qualified_window_count"] = (
        radius_qualified_count
    )
    selected["grid_scale_packet_amplitude_qualified_window_count"] = (
        None if config.amplitude_min is None else amplitude_qualified_count
    )
    return selected


def extract_axis_artifact_features(
    mode: np.ndarray,
    *,
    r_ax: float = DEFAULT_AXIS_R_AX,
    amplitude_min: float | None = None,
    width_max_grid: float | None = None,
) -> dict[str, Any]:
    """Select the strongest width-qualified local peak in the axis window.

    Every local maximum of every absolute harmonic profile centered in the
    inclusive axis window is measured on the complete radial grid. When a
    amplitude and width limits are supplied, the strongest peak satisfying
    both is the decision candidate. If none qualifies, retain the strongest
    raw window amplitude as fallback audit information.
    """
    if not math.isfinite(r_ax) or not 0.0 < r_ax <= 1.0:
        raise ValueError("axis r_ax must be finite and in (0, 1]")
    if amplitude_min is not None and (
        not math.isfinite(amplitude_min) or not 0.0 <= amplitude_min <= 1.0
    ):
        raise ValueError(
            "axis amplitude_min must be null or finite and in [0, 1]"
        )
    if width_max_grid is not None and (
        not math.isfinite(width_max_grid) or width_max_grid < 0.0
    ):
        raise ValueError(
            "axis width_max_grid must be null or a finite nonnegative number"
        )
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")

    n_radial = mode_array.shape[1]
    radial_grid = np.linspace(0.0, 1.0, n_radial)
    radial_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(r_ax))
    axis_indices = np.flatnonzero(radial_grid <= r_ax + radial_tolerance)
    if axis_indices.size == 0:
        raise ValueError("axis window contains no radial grid samples")

    absolute_mode = np.abs(mode_array)

    def measure_peak(
        harmonic_index: int,
        peak_index: int,
        *,
        is_local_max: bool,
    ) -> dict[str, Any]:
        profile = absolute_mode[harmonic_index]
        inner_edge, outer_edge, width_r, width_grid, touches_boundary = (
            _signed_halfmax_component(
                profile,
                peak_index=peak_index,
                radial_grid=radial_grid,
            )
        )
        return {
            "axis_peak": float(profile[peak_index]),
            "axis_peak_harmonic_index": int(harmonic_index),
            "axis_peak_r": float(radial_grid[peak_index]),
            "axis_peak_is_local_max": is_local_max,
            "axis_halfmax_width_r": width_r,
            "axis_halfmax_width_grid": width_grid,
            "axis_halfmax_outer_edge_r": outer_edge,
            "axis_component_touches_boundary": touches_boundary,
        }

    local_candidates: list[dict[str, Any]] = []
    local_peak_keys: set[tuple[int, int]] = set()
    for harmonic_index, profile in enumerate(absolute_mode):
        for peak_index, _plateau_left, _plateau_right in _signed_local_extrema(
            profile
        ):
            if radial_grid[peak_index] > r_ax + radial_tolerance:
                continue
            local_peak_keys.add((harmonic_index, peak_index))
            local_candidates.append(
                measure_peak(
                    harmonic_index,
                    peak_index,
                    is_local_max=True,
                )
            )

    window = absolute_mode[:, axis_indices]
    fallback_harmonic, fallback_window_index = np.unravel_index(
        np.argmax(window), window.shape
    )
    fallback_peak_index = int(axis_indices[fallback_window_index])
    fallback = measure_peak(
        int(fallback_harmonic),
        fallback_peak_index,
        is_local_max=(
            (int(fallback_harmonic), fallback_peak_index) in local_peak_keys
        ),
    )

    amplitude_qualified: list[dict[str, Any]] = []
    if amplitude_min is not None:
        amplitude_qualified = [
            candidate
            for candidate in local_candidates
            if candidate["axis_peak"] >= amplitude_min
        ]

    width_qualified: list[dict[str, Any]] = []
    if width_max_grid is not None:
        width_tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, width_max_grid
        )
        width_qualified = [
            candidate
            for candidate in local_candidates
            if candidate["axis_halfmax_width_grid"]
            <= width_max_grid + width_tolerance
        ]

    decision_candidates: list[dict[str, Any]] = []
    if amplitude_min is not None and width_max_grid is not None:
        decision_candidates = [
            candidate
            for candidate in width_qualified
            if candidate["axis_peak"] >= amplitude_min
        ]

    if decision_candidates:
        selected = min(
            decision_candidates,
            key=lambda candidate: (
                -candidate["axis_peak"],
                candidate["axis_halfmax_width_grid"],
                candidate["axis_peak_harmonic_index"],
                candidate["axis_peak_r"],
            ),
        )
    else:
        selected = fallback

    return {
        "r_ax": float(r_ax),
        "axis_candidate_found": (
            None
            if amplitude_min is None or width_max_grid is None
            else bool(decision_candidates)
        ),
        "axis_candidate_amplitude_min": amplitude_min,
        "axis_candidate_width_limit_grid": width_max_grid,
        "axis_local_peak_count": len(local_candidates),
        "axis_amplitude_qualified_peak_count": (
            None if amplitude_min is None else len(amplitude_qualified)
        ),
        "axis_width_qualified_peak_count": (
            None if width_max_grid is None else len(width_qualified)
        ),
        **selected,
    }


def extract_continuum_crossing_window_features(
    mode: np.ndarray,
    crossing_records: list[Mapping[str, Any]],
    *,
    half_width_grid: int = DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID,
) -> dict[str, Any]:
    """Measure harmonic amplitude and normalized energy near true crossings.

    The inclusive radial window around each interpolated crossing is
    ``abs(r_i - r_cross) <= half_width_grid * delta_r``. Winners are selected
    independently for individual-harmonic absolute amplitude and total radial
    energy so both kinds of evidence remain auditable.
    """
    if isinstance(half_width_grid, bool) or not isinstance(half_width_grid, int):
        raise ValueError("cross_window half_width_grid must be an integer")
    if half_width_grid < 0:
        raise ValueError("cross_window half_width_grid must be nonnegative")
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")

    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    delta_r = float(radial_grid[1] - radial_grid[0])
    half_width_r = float(half_width_grid * delta_r)
    result = empty_continuum_crossing_window_features(
        half_width_grid,
        candidate_found=False,
    )
    result["cross_window_half_width_r"] = half_width_r
    if not crossing_records:
        return result

    ordered_records: list[tuple[str, float]] = []
    for record in crossing_records:
        if set(record) != {"boundary", "r_cross", "W_peak", "shear_weighted"}:
            raise ValueError("crossing record does not match the audit schema")
        boundary = str(record["boundary"])
        r_cross = float(record["r_cross"])
        if boundary not in {"low", "high"} or not math.isfinite(r_cross):
            raise ValueError("crossing record contains invalid window coordinates")
        ordered_records.append((boundary, r_cross))
    ordered_records.sort(key=lambda item: (0 if item[0] == "low" else 1, item[1]))

    radial_energy = np.sum(mode_array**2, axis=0)
    energy_max = float(np.max(radial_energy))
    if energy_max > 0.0:
        normalized_energy = radial_energy / energy_max
    else:
        normalized_energy = np.zeros_like(radial_energy)
    absolute_mode = np.abs(mode_array)
    radial_tolerance = 64.0 * np.finfo(float).eps * max(1.0, half_width_r)

    amplitude_winner: tuple[float, int, int, str, float] | None = None
    energy_winner: tuple[float, int, str, float] | None = None
    for boundary, r_cross in ordered_records:
        sample_indices = np.flatnonzero(
            np.abs(radial_grid - r_cross) <= half_width_r + radial_tolerance
        )
        for sample_index_raw in sample_indices:
            sample_index = int(sample_index_raw)
            harmonic_index = int(np.argmax(absolute_mode[:, sample_index]))
            amplitude = float(absolute_mode[harmonic_index, sample_index])
            if amplitude_winner is None or amplitude > amplitude_winner[0]:
                amplitude_winner = (
                    amplitude,
                    harmonic_index,
                    sample_index,
                    boundary,
                    r_cross,
                )
            energy = float(normalized_energy[sample_index])
            if energy_winner is None or energy > energy_winner[0]:
                energy_winner = (energy, sample_index, boundary, r_cross)

    if amplitude_winner is None or energy_winner is None:
        raise ValueError("crossing window contains no radial grid samples")

    amplitude, harmonic_index, sample_index, boundary, r_cross = amplitude_winner
    sample_r = float(radial_grid[sample_index])
    neighbor_indices = sample_index + np.array([-2, -1, 1, 2], dtype=int)
    valid_neighbor_indices = neighbor_indices[
        (neighbor_indices >= 0) & (neighbor_indices < mode_array.shape[1])
    ]
    neighbor_count = int(valid_neighbor_indices.size)
    neighbor_stencil_complete = neighbor_count == 4
    neighbor_rms: float | None = None
    if neighbor_stencil_complete:
        signed_center = float(mode_array[harmonic_index, sample_index])
        signed_neighbors = mode_array[harmonic_index, neighbor_indices]
        signed_differences = signed_center - signed_neighbors
        neighbor_rms = float(np.sqrt(np.mean(signed_differences**2)))
    result.update(
        {
            "cross_window_candidate_found": True,
            "cross_window_A_max": amplitude,
            "cross_window_A_harmonic_index": harmonic_index,
            "cross_window_A_sample_r": sample_r,
            "cross_window_A_crossing_boundary": boundary,
            "cross_window_A_crossing_r": r_cross,
            "cross_window_A_distance_grid": abs(sample_r - r_cross) / delta_r,
            "cross_window_A_neighbor_rms": neighbor_rms,
            "cross_window_A_neighbor_count": neighbor_count,
            "cross_window_A_neighbor_stencil_complete": (
                neighbor_stencil_complete
            ),
        }
    )
    energy, sample_index, boundary, r_cross = energy_winner
    sample_r = float(radial_grid[sample_index])
    result.update(
        {
            "cross_window_W_max": energy,
            "cross_window_W_sample_r": sample_r,
            "cross_window_W_crossing_boundary": boundary,
            "cross_window_W_crossing_r": r_cross,
            "cross_window_W_distance_grid": abs(sample_r - r_cross) / delta_r,
        }
    )
    return result


def extract_edge_artifact_features(
    mode: np.ndarray,
    *,
    r_edge_min: float = DEFAULT_EDGE_R_MIN,
) -> dict[str, Any]:
    """Measure a narrow edge-localized total-energy envelope and its harmonic.

    The decision evidence uses the global peak of normalized radial energy
    ``sum_m |mode_m(r)|^2``. The strongest individual harmonic in the inclusive
    edge window is retained separately for audit because physical edge modes
    can contain narrow harmonics while their total envelope remains resolved.
    All half-maximum edges are searched over the complete radial grid.
    """
    if not math.isfinite(r_edge_min) or not 0.0 <= r_edge_min < 1.0:
        raise ValueError("edge r_edge_min must be finite and in [0, 1)")
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")

    n_radial = mode_array.shape[1]
    radial_grid = np.linspace(0.0, 1.0, n_radial)
    radial_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(r_edge_min)
    )
    radial_energy = np.sum(np.square(np.abs(mode_array)), axis=0)
    energy_peak_index = int(np.argmax(radial_energy))
    energy_peak_raw = float(radial_energy[energy_peak_index])
    energy_peak_r = float(radial_grid[energy_peak_index])

    result = empty_edge_artifact_features(float(r_edge_min))
    result.update(
        {
            "edge_energy_peak": 1.0 if energy_peak_raw > 0.0 else 0.0,
            "edge_energy_peak_r": energy_peak_r,
            "edge_energy_peak_in_window": bool(
                energy_peak_raw > 0.0
                and energy_peak_r >= r_edge_min - radial_tolerance
            ),
        }
    )
    if energy_peak_raw > 0.0:
        normalized_energy = radial_energy / energy_peak_raw
        inner_edge, outer_edge, width_r, width_grid, _touches_boundary = (
            _signed_halfmax_component(
                normalized_energy,
                peak_index=energy_peak_index,
                radial_grid=radial_grid,
            )
        )
        result.update(
            {
                "edge_energy_halfmax_width_r": width_r,
                "edge_energy_halfmax_width_grid": width_grid,
                "edge_energy_halfmax_inner_edge_r": inner_edge,
                "edge_energy_halfmax_outer_edge_r": outer_edge,
                "edge_energy_component_touches_boundary": bool(
                    outer_edge >= 1.0 - radial_tolerance
                ),
            }
        )

    # Reversing the radial axis lets the established inclusive axis-window
    # measurement audit the strongest individual harmonic near r=1.
    mirrored = extract_axis_artifact_features(
        mode_array[:, ::-1],
        r_ax=1.0 - r_edge_min,
    )
    harmonic_index = int(mirrored["axis_peak_harmonic_index"])
    harmonic_peak_r = 1.0 - float(mirrored["axis_peak_r"])
    harmonic_peak_index = int(round(harmonic_peak_r * (n_radial - 1)))
    harmonic_peak = float(mirrored["axis_peak"])
    result.update(
        {
            "edge_harmonic_peak": harmonic_peak,
            "edge_harmonic_peak_harmonic_index": harmonic_index,
            "edge_harmonic_peak_r": harmonic_peak_r,
            "edge_harmonic_peak_is_local_max": bool(
                mirrored["axis_peak_is_local_max"]
            ),
        }
    )
    if harmonic_peak > 0.0:
        harmonic_profile = np.abs(mode_array[harmonic_index])
        inner_edge, outer_edge, width_r, width_grid, _touches_boundary = (
            _signed_halfmax_component(
                harmonic_profile,
                peak_index=harmonic_peak_index,
                radial_grid=radial_grid,
            )
        )
        result.update(
            {
                "edge_harmonic_halfmax_width_r": width_r,
                "edge_harmonic_halfmax_width_grid": width_grid,
                "edge_harmonic_halfmax_inner_edge_r": inner_edge,
                "edge_harmonic_halfmax_outer_edge_r": outer_edge,
                "edge_harmonic_component_touches_boundary": bool(
                    outer_edge >= 1.0 - radial_tolerance
                ),
            }
        )
    return result


def extract_interior_unresolved_envelope_features(
    mode: np.ndarray,
    omega: float,
    low2: np.ndarray,
    high2: np.ndarray,
    *,
    total_energy_features: Mapping[str, Any],
    config: InteriorUnresolvedEnvelopeConfig | None = None,
) -> dict[str, Any]:
    """Combine shared total-W evidence with a gate-specific extremum match.

    The total-energy peak and connected FWHM are copied from the edge extractor,
    which already measures the global ``sum_h |mode_h(r)|^2`` envelope.  Only
    the continuum-extremum search is recomputed here: its wider rule-specific
    radial interval must not alter the established RF feature calculations.
    """
    resolved = config or InteriorUnresolvedEnvelopeConfig()
    result = empty_interior_unresolved_envelope_features(resolved)
    inner_edge = total_energy_features["edge_energy_halfmax_inner_edge_r"]
    outer_edge = total_energy_features["edge_energy_halfmax_outer_edge_r"]
    result.update(
        {
            "energy_peak": total_energy_features["edge_energy_peak"],
            "energy_peak_r": total_energy_features["edge_energy_peak_r"],
            "energy_halfmax_width_r": total_energy_features[
                "edge_energy_halfmax_width_r"
            ],
            "energy_halfmax_width_grid": total_energy_features[
                "edge_energy_halfmax_width_grid"
            ],
            "energy_halfmax_inner_edge_r": inner_edge,
            "energy_halfmax_outer_edge_r": outer_edge,
            "energy_component_touches_boundary": (
                None
                if inner_edge is None or outer_edge is None
                else bool(inner_edge <= 0.0 or outer_edge >= 1.0)
            ),
        }
    )

    extremum, match_found = continuum_extremum_features(
        mode,
        omega,
        low2,
        high2,
        r_min=resolved.extremum_r_min,
        r_max=resolved.extremum_r_max,
        return_match_status=True,
        filter_candidates_after_detection=True,
    )
    result["extremum_match_found"] = bool(match_found)
    if match_found:
        result.update(
            {
                "ext_dr": float(extremum["ext_dr"]),
                "ext_df_gap": float(extremum["ext_df_gap"]),
                "ext_energy_frac": float(extremum["ext_energy_frac"]),
            }
        )

    width_grid = result["energy_halfmax_width_grid"]
    peak_r = result["energy_peak_r"]
    energy_peak = result["energy_peak"]
    if resolved.width_max_grid is None:
        candidate_found: bool | None = None
    else:
        radial_tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(resolved.peak_r_max)
        )
        width_tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(resolved.width_max_grid)
        )
        candidate_found = bool(
            energy_peak is not None
            and energy_peak > 0.0
            and peak_r is not None
            and peak_r <= resolved.peak_r_max + radial_tolerance
            and width_grid is not None
            and width_grid <= resolved.width_max_grid + width_tolerance
        )
    result["candidate_found"] = candidate_found

    ext_dr = result["ext_dr"]
    ext_df_gap = result["ext_df_gap"]
    ext_dr_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(resolved.ext_dr_max)
    )
    ext_df_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(resolved.ext_df_gap_min),
        abs(resolved.ext_df_gap_max),
    )
    exception_qualified = bool(
        match_found
        and ext_dr is not None
        and ext_dr <= resolved.ext_dr_max + ext_dr_tolerance
        and ext_df_gap is not None
        and resolved.ext_df_gap_min - ext_df_tolerance
        <= ext_df_gap
        <= resolved.ext_df_gap_max + ext_df_tolerance
    )
    result["extremum_exception_applied"] = bool(
        candidate_found is True and exception_qualified
    )
    return result


@dataclass(frozen=True)
class RuleResult:
    """Stable, auditable result returned for one preprocessed TAE-side mode."""

    path: str
    mode_key: str
    shot: str
    ntor: int | None
    frequency: float | None
    input_fingerprint: str
    gap_region: str
    decision: str
    primary_reason: str
    triggered_rules: tuple[str, ...]
    rule_version: str = RULESET_VERSION
    features: Mapping[str, Any] = field(default_factory=empty_rule_features)
    processing_status: str = "RULE_EVALUATED"
    diagnostic_message: str = ""

    def as_output_row(self, base_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Merge the result into the shared CSV schema."""
        row = empty_rule_row()
        if base_row is not None:
            row.update(base_row)
        row.update(
            {
                "path": self.path,
                "mode_key": self.mode_key,
                "shot": self.shot,
                "ntor": "" if self.ntor is None else self.ntor,
                "omega": "" if self.frequency is None else self.frequency,
                "input_fingerprint": self.input_fingerprint,
                "gap_region": self.gap_region,
                "processing_status": self.processing_status,
                "rule_decision": self.decision,
                "rule_primary_reason": self.primary_reason,
                "rule_triggered_rules": stable_json(self.triggered_rules),
                "rule_version": self.rule_version,
                "rule_features": stable_json(self.features),
                "final_decision": self.decision,
                "decision_source": "rule_engine",
                "diagnostic_message": self.diagnostic_message,
            }
        )
        return row


def evaluate_mode(
    preprocessed_row: Mapping[str, Any],
    *,
    mode: np.ndarray | None = None,
    low2: np.ndarray | None = None,
    high2: np.ndarray | None = None,
    axis_artifact_config: AxisArtifactConfig | None = None,
    grid_scale_spike_config: GridScaleSpikeConfig | None = None,
    grid_scale_packet_config: GridScalePacketConfig | None = None,
    continuum_crossing_config: ContinuumCrossingConfig | None = None,
    continuum_crossing_window_config: ContinuumCrossingWindowConfig | None = None,
    edge_artifact_config: EdgeArtifactConfig | None = None,
    interior_unresolved_envelope_config: (
        InteriorUnresolvedEnvelopeConfig | None
    ) = None,
) -> RuleResult:
    """Extract named features and evaluate one valid, preprocessed TAE mode."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    packet_config = grid_scale_packet_config or GridScalePacketConfig()
    crossing_config = continuum_crossing_config or ContinuumCrossingConfig()
    cross_window_config = (
        continuum_crossing_window_config or ContinuumCrossingWindowConfig()
    )
    edge_config = edge_artifact_config or EdgeArtifactConfig()
    interior_config = (
        interior_unresolved_envelope_config or InteriorUnresolvedEnvelopeConfig()
    )
    path = str(preprocessed_row.get("path", ""))
    mode_key = str(preprocessed_row.get("mode_key", ""))
    shot = str(preprocessed_row.get("shot", ""))
    fingerprint = str(preprocessed_row.get("input_fingerprint", ""))
    gap_region = str(preprocessed_row.get("gap_region", ""))
    try:
        ntor = int(preprocessed_row.get("ntor"))
        frequency = float(preprocessed_row.get("omega"))
        gamma_d = float(preprocessed_row.get("gamma_d"))
        if not (math.isfinite(frequency) and math.isfinite(gamma_d)):
            raise ValueError("frequency or gamma_d is non-finite")
        if not path or not mode_key or not shot or len(fingerprint) != 64:
            raise ValueError("missing path, mode key, shot, or input fingerprint")
        if gap_region not in {"tae_like", "mixed"}:
            raise ValueError(f"unsupported gap region {gap_region!r}")
    except (TypeError, ValueError) as exc:
        reason = "RULE_INPUT_INVALID"
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=None,
            frequency=None,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="INVALID",
            primary_reason=reason,
            triggered_rules=(reason,),
            features=empty_rule_features(
                axis_config,
                grid_config,
                edge_config,
                cross_window_config,
                packet_config,
                interior_config,
            ),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    try:
        if mode is None or low2 is None or high2 is None:
            raise ValueError("mode, low2, and high2 arrays are required")
        axis_features = extract_axis_artifact_features(
            mode,
            r_ax=axis_config.r_ax,
            amplitude_min=axis_config.axis_amplitude_min,
            width_max_grid=axis_config.axis_width_max_grid,
        )
        grid_scale_features = extract_grid_scale_spike_features(
            mode,
            width_max_grid=grid_config.width_max_grid,
            high_r_cutoff_r=grid_config.high_r_cutoff_r,
            high_r_width_max_grid=grid_config.high_r_width_max_grid,
        )
        grid_scale_packet_features = extract_grid_scale_packet_features(
            mode,
            amplitude_min=packet_config.amplitude_min,
            step_min=packet_config.step_min,
            min_large_turns=packet_config.min_large_turns,
            window_span_grid=packet_config.window_span_grid,
            peak_r_max=packet_config.peak_r_max,
        )
        edge_features = extract_edge_artifact_features(
            mode,
            r_edge_min=edge_config.r_edge_min,
        )
        named_features, feature_status = compute_named_features_for_mode(
            mode,
            extra_info={
                "path": path,
                "omega": frequency,
                "gamma_d": gamma_d,
                "ntor": ntor,
            },
            include_crossing_features=True,
            include_extremum_features=True,
            continuum_arrays=(low2, high2),
            strict_continuum=True,
            null_missing_crossings=True,
            null_missing_extremum=True,
            return_feature_status=True,
            return_crossing_records=True,
        )
        if tuple(named_features) != RULE_FEATURE_NAMES:
            raise ValueError("named feature order does not match the rule schema")
        if not all(
            value is None or math.isfinite(value)
            for value in named_features.values()
        ):
            raise ValueError("one or more rule features are non-finite")
        for record in feature_status["crossing_records"]:
            if set(record) != {"boundary", "r_cross", "W_peak", "shear_weighted"}:
                raise ValueError("crossing record does not match the audit schema")
            if record["boundary"] not in {"low", "high"} or not all(
                math.isfinite(record[name])
                for name in ("r_cross", "W_peak", "shear_weighted")
            ):
                raise ValueError("crossing record contains invalid values")
        cross_window_features = extract_continuum_crossing_window_features(
            mode,
            feature_status["crossing_records"],
            half_width_grid=cross_window_config.half_width_grid,
        )
        interior_envelope_features = (
            extract_interior_unresolved_envelope_features(
                mode,
                frequency,
                low2,
                high2,
                total_energy_features=edge_features,
                config=interior_config,
            )
        )
        features = grouped_rule_features(
            named_features,
            feature_status,
            axis_features,
            grid_scale_features,
            grid_scale_packet_features,
            edge_features,
            cross_window_features,
            interior_envelope_features,
        )
    except Exception as exc:
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="INVALID",
            primary_reason=RULE_FEATURE_EXTRACTION_FAILED,
            triggered_rules=(RULE_FEATURE_EXTRACTION_FAILED,),
            features=empty_rule_features(
                axis_config,
                grid_config,
                edge_config,
                cross_window_config,
                packet_config,
                interior_config,
            ),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    amplitude_min = axis_config.axis_amplitude_min
    width_max_grid = axis_config.axis_width_max_grid
    if (
        amplitude_min is not None
        and width_max_grid is not None
        and axis_features["axis_candidate_found"]
        and axis_features["axis_peak_is_local_max"]
        and axis_features["axis_peak"] >= amplitude_min
        and axis_features["axis_halfmax_width_grid"] <= width_max_grid
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_AXIS_SPIKE,
            triggered_rules=(BAD_AXIS_SPIKE,),
            features=features,
        )

    if (
        grid_config.enabled
        and grid_scale_features["grid_scale_candidate_found"]
        and grid_scale_features["grid_scale_peak"] >= grid_config.amplitude_min
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_GRID_SCALE_SPIKE,
            triggered_rules=(BAD_GRID_SCALE_SPIKE,),
            features=features,
        )

    if (
        packet_config.enabled
        and grid_scale_packet_features["grid_scale_packet_candidate_found"]
        and grid_scale_packet_features["grid_scale_packet_peak"]
        >= packet_config.amplitude_min
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_GRID_SCALE_PACKET,
            triggered_rules=(BAD_GRID_SCALE_PACKET,),
            features=features,
        )

    n_cross = named_features["n_cross"]
    w_star_max = named_features["W_star_max"]
    w_cross_threshold = crossing_config.w_cross_threshold
    if (
        w_cross_threshold is not None
        and n_cross is not None
        and n_cross > 0.0
        and w_star_max is not None
        and w_star_max > w_cross_threshold
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_CONT_CROSS,
            triggered_rules=(BAD_CONT_CROSS,),
            features=features,
        )

    cross_window_amplitude = cross_window_features["cross_window_A_max"]
    cross_window_w = cross_window_features["cross_window_W_max"]
    amplitude_hit = (
        cross_window_config.amplitude_min is not None
        and cross_window_amplitude is not None
        and cross_window_amplitude >= cross_window_config.amplitude_min
    )
    energy_hit = (
        cross_window_config.w_min is not None
        and cross_window_w is not None
        and cross_window_w >= cross_window_config.w_min
    )
    if (
        cross_window_config.enabled
        and n_cross is not None
        and n_cross > 0.0
        and cross_window_features["cross_window_candidate_found"]
        and (amplitude_hit or energy_hit)
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_CONT_CROSS_WINDOW,
            triggered_rules=(BAD_CONT_CROSS_WINDOW,),
            features=features,
        )

    edge_width_max_grid = edge_config.edge_width_max_grid
    if (
        edge_width_max_grid is not None
        and edge_features["edge_energy_peak_in_window"]
        and edge_features["edge_energy_halfmax_width_grid"] is not None
        and edge_features["edge_energy_halfmax_width_grid"]
        <= edge_width_max_grid
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_EDGE_SPIKE,
            triggered_rules=(BAD_EDGE_SPIKE,),
            features=features,
        )

    if (
        interior_config.enabled
        and interior_envelope_features["candidate_found"]
        and not interior_envelope_features["extremum_exception_applied"]
    ):
        return RuleResult(
            path=path,
            mode_key=mode_key,
            shot=shot,
            ntor=ntor,
            frequency=frequency,
            input_fingerprint=fingerprint,
            gap_region=gap_region,
            decision="BAD",
            primary_reason=BAD_INTERIOR_UNRESOLVED_ENVELOPE,
            triggered_rules=(BAD_INTERIOR_UNRESOLVED_ENVELOPE,),
            features=features,
        )

    # Not rejected is not equivalent to GOOD. Positive templates and later
    # ordered gates remain to be implemented.
    return RuleResult(
        path=path,
        mode_key=mode_key,
        shot=shot,
        ntor=ntor,
        frequency=frequency,
        input_fingerprint=fingerprint,
        gap_region=gap_region,
        decision="REVIEW",
        primary_reason=NO_GOOD_TEMPLATE,
        triggered_rules=(NO_GOOD_TEMPLATE,),
        features=features,
    )
