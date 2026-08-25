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
DEFAULT_W_CROSS_THRESHOLD = 0.03
DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID = 2
DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN = 0.25
DEFAULT_CROSS_WINDOW_W_MIN = 0.05
DEFAULT_EDGE_R_MIN = 0.97
DEFAULT_EDGE_WIDTH_MAX_GRID = 10.0

RULESET_VERSION = "tae-rules-axis-grid-cont-window-edge-v9"
BAD_AXIS_SPIKE = "BAD_AXIS_SPIKE"
BAD_GRID_SCALE_SPIKE = "BAD_GRID_SCALE_SPIKE"
BAD_CONT_CROSS = "BAD_CONT_CROSS"
BAD_CONT_CROSS_WINDOW = "BAD_CONT_CROSS_WINDOW"
BAD_EDGE_SPIKE = "BAD_EDGE_SPIKE"
NO_GOOD_TEMPLATE = "NO_GOOD_TEMPLATE"
RULE_FEATURE_EXTRACTION_FAILED = "RULE_FEATURE_EXTRACTION_FAILED"
RULE_FEATURE_NAMES = tuple(
    get_feature_names(include_crossing_features=True, include_extremum_features=True)
)
RULE_FEATURE_SCHEMA_VERSION = "tae-rule-features-grouped-v8"
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
    """Thresholds for the unresolved signed-harmonic spike gate."""

    amplitude_min: float | None = DEFAULT_GRID_SCALE_AMPLITUDE_MIN
    width_max_grid: float | None = DEFAULT_GRID_SCALE_WIDTH_MAX_GRID

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

    @property
    def enabled(self) -> bool:
        """Return whether both thresholds required by the gate are configured."""
        return self.amplitude_min is not None and self.width_max_grid is not None


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


def empty_axis_artifact_features(
    r_ax: float = DEFAULT_AXIS_R_AX,
) -> dict[str, Any]:
    """Return the stable axis-artifact feature shape with null measurements."""
    return {
        "r_ax": r_ax,
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
    candidate_found: bool | None = None,
) -> dict[str, Any]:
    """Return the stable grid-scale-spike shape with null measurements."""
    return {
        "grid_scale_candidate_found": candidate_found,
        "grid_scale_candidate_width_limit_grid": width_max_grid,
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
) -> dict[str, Any]:
    """Return the complete rule-feature schema with unavailable values as null."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    edge_config = edge_artifact_config or EdgeArtifactConfig()
    cross_window_config = (
        continuum_crossing_window_config or ContinuumCrossingWindowConfig()
    )
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "rf_standard_features": {name: None for name in RF_FEATURE_NAMES},
        "resolution_features": {},
        "numerical_structure_features": {
            "grid_scale_spike": empty_grid_scale_spike_features(
                grid_config.width_max_grid
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
            "axis_artifact": empty_axis_artifact_features(axis_config.r_ax),
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
    edge_artifact_features: Mapping[str, Any],
    continuum_crossing_window_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Organize shared RF31 measurements and deterministic rule evidence."""
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "rf_standard_features": {
            name: named_features[name] for name in RF_FEATURE_NAMES
        },
        "resolution_features": {},
        "numerical_structure_features": {
            "grid_scale_spike": dict(grid_scale_spike_features),
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
) -> dict[str, Any]:
    """Find the strongest signed local lobe within a grid-width limit.

    Each positive maximum or negative minimum is measured on its signed
    harmonic profile over the complete radial grid. This deliberately avoids
    ``abs(mode)``, which can join unresolved adjacent ``+A/-A`` lobes.
    """
    if width_max_grid is not None and (
        not math.isfinite(width_max_grid) or width_max_grid < 0.0
    ):
        raise ValueError(
            "grid-scale width_max_grid must be null or a finite nonnegative number"
        )
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")
    if width_max_grid is None:
        return empty_grid_scale_spike_features(
            width_max_grid,
            candidate_found=False,
        )

    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    candidates: list[dict[str, Any]] = []
    width_tolerance = 64.0 * np.finfo(float).eps * max(1.0, width_max_grid)
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
            if width_grid > width_max_grid + width_tolerance:
                continue
            signed_amplitude = float(profile[peak_index])
            candidates.append(
                {
                    "grid_scale_candidate_found": True,
                    "grid_scale_candidate_width_limit_grid": float(width_max_grid),
                    "grid_scale_peak": abs(signed_amplitude),
                    "grid_scale_peak_signed_amplitude": signed_amplitude,
                    "grid_scale_peak_sign": 1 if signed_amplitude > 0.0 else -1,
                    "grid_scale_peak_harmonic_index": int(harmonic_index),
                    "grid_scale_peak_r": float(radial_grid[peak_index]),
                    "grid_scale_halfmax_width_r": width_r,
                    "grid_scale_halfmax_width_grid": width_grid,
                    "grid_scale_halfmax_inner_edge_r": inner_edge,
                    "grid_scale_halfmax_outer_edge_r": outer_edge,
                    "grid_scale_component_touches_boundary": touches_boundary,
                }
            )

    if not candidates:
        return empty_grid_scale_spike_features(
            float(width_max_grid),
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


def extract_axis_artifact_features(
    mode: np.ndarray,
    *,
    r_ax: float = DEFAULT_AXIS_R_AX,
) -> dict[str, Any]:
    """Measure the strongest harmonic amplitude in the inclusive axis window.

    The half-maximum component and local-maximum test use the complete radial
    profile of the selected stored harmonic, not only the axis search window.
    """
    if not math.isfinite(r_ax) or not 0.0 < r_ax <= 1.0:
        raise ValueError("axis r_ax must be finite and in (0, 1]")
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
    window = absolute_mode[:, axis_indices]
    harmonic_index, window_index = np.unravel_index(np.argmax(window), window.shape)
    peak_index = int(axis_indices[window_index])
    peak = float(absolute_mode[harmonic_index, peak_index])
    profile = absolute_mode[harmonic_index]

    # Treat a flat-topped peak as one plateau and compare its outer neighbors.
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, peak)
    plateau_left = peak_index
    while (
        plateau_left > 0
        and abs(float(profile[plateau_left - 1]) - peak) <= tolerance
    ):
        plateau_left -= 1
    plateau_right = peak_index
    while (
        plateau_right < n_radial - 1
        and abs(float(profile[plateau_right + 1]) - peak) <= tolerance
    ):
        plateau_right += 1
    left_is_lower = (
        plateau_left == 0
        or float(profile[plateau_left - 1]) < peak - tolerance
    )
    right_is_lower = (
        plateau_right == n_radial - 1
        or float(profile[plateau_right + 1]) < peak - tolerance
    )
    has_outer_neighbor = plateau_left > 0 or plateau_right < n_radial - 1
    is_local_max = bool(
        peak > 0.0 and has_outer_neighbor and left_is_lower and right_is_lower
    )

    half_maximum = 0.5 * peak
    component_left = peak_index
    while component_left > 0 and profile[component_left - 1] >= half_maximum:
        component_left -= 1
    component_right = peak_index
    while (
        component_right < n_radial - 1
        and profile[component_right + 1] >= half_maximum
    ):
        component_right += 1

    if component_left == 0:
        inner_edge = 0.0
    else:
        inner_edge = _interpolate_threshold_crossing(
            float(radial_grid[component_left - 1]),
            float(profile[component_left - 1]),
            float(radial_grid[component_left]),
            float(profile[component_left]),
            half_maximum,
        )
    if component_right == n_radial - 1:
        outer_edge = 1.0
    else:
        outer_edge = _interpolate_threshold_crossing(
            float(radial_grid[component_right + 1]),
            float(profile[component_right + 1]),
            float(radial_grid[component_right]),
            float(profile[component_right]),
            half_maximum,
        )
    width_r = max(0.0, outer_edge - inner_edge)
    radial_interval = 1.0 / (n_radial - 1)

    return {
        "r_ax": float(r_ax),
        "axis_peak": peak,
        "axis_peak_harmonic_index": int(harmonic_index),
        "axis_peak_r": float(radial_grid[peak_index]),
        "axis_peak_is_local_max": is_local_max,
        "axis_halfmax_width_r": float(width_r),
        "axis_halfmax_width_grid": float(width_r / radial_interval),
        "axis_halfmax_outer_edge_r": float(outer_edge),
        "axis_component_touches_boundary": bool(component_left == 0),
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
    continuum_crossing_config: ContinuumCrossingConfig | None = None,
    continuum_crossing_window_config: ContinuumCrossingWindowConfig | None = None,
    edge_artifact_config: EdgeArtifactConfig | None = None,
) -> RuleResult:
    """Extract named features and evaluate one valid, preprocessed TAE mode."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    crossing_config = continuum_crossing_config or ContinuumCrossingConfig()
    cross_window_config = (
        continuum_crossing_window_config or ContinuumCrossingWindowConfig()
    )
    edge_config = edge_artifact_config or EdgeArtifactConfig()
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
            ),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    try:
        if mode is None or low2 is None or high2 is None:
            raise ValueError("mode, low2, and high2 arrays are required")
        axis_features = extract_axis_artifact_features(mode, r_ax=axis_config.r_ax)
        grid_scale_features = extract_grid_scale_spike_features(
            mode,
            width_max_grid=grid_config.width_max_grid,
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
        features = grouped_rule_features(
            named_features,
            feature_status,
            axis_features,
            grid_scale_features,
            edge_features,
            cross_window_features,
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
            ),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    amplitude_min = axis_config.axis_amplitude_min
    width_max_grid = axis_config.axis_width_max_grid
    if (
        amplitude_min is not None
        and width_max_grid is not None
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
