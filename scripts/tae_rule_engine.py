#!/usr/bin/env python3
"""Pure per-mode interface for deterministic NOVA TAE rule evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from _repo_bootstrap import ensure_repo_src_on_path


ensure_repo_src_on_path()

from cont_features import continuum_extremum_features, _energy_fraction_in_window  # noqa: E402
from mode_features import (  # noqa: E402
    EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES,
    EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES,
    RF_FEATURE_NAMES,
    compute_named_features_for_mode,
    get_feature_names,
    get_feature_schema_version,
)
from tae_rule_io import empty_rule_row, stable_json  # noqa: E402
from continuum_noise import (  # noqa: E402
    BAD_EXTENDED_CONTINUUM_NOISE,
    ContinuumNoiseThresholds,
    empty_continuum_noise_features,
    extract_continuum_noise_features,
)


DEFAULT_AXIS_R_AX = 0.03
DEFAULT_AXIS_AMPLITUDE_MIN = 0.2
DEFAULT_AXIS_WIDTH_MAX_GRID = 10.0
DEFAULT_AXIS_ENERGY_AMPLITUDE_R_MAX = 0.015
DEFAULT_AXIS_ENERGY_AMPLITUDE_MIN = 0.5
DEFAULT_AXIS_ENERGY_R_MAX = 0.05
DEFAULT_AXIS_ENERGY_FRACTION_MIN = 0.5
DEFAULT_GRID_SCALE_AMPLITUDE_MIN = 0.3
DEFAULT_GRID_SCALE_WIDTH_MAX_GRID = 1.0
DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R = 0.7
DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID = 0.75
DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN = 0.3
DEFAULT_GRID_SCALE_PACKET_STEP_MIN = 0.2
DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS = 3
DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID = 4
DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX = 0.5
DEFAULT_NEAR_AXIS_GRID_OSCILLATION_PEAK_R_MAX = 0.1
DEFAULT_NEAR_AXIS_GRID_OSCILLATION_AMPLITUDE_MIN = 0.10
DEFAULT_NEAR_AXIS_GRID_OSCILLATION_MIN_CONSECUTIVE_SIGN_FLIPS = 4
DEFAULT_NEAR_AXIS_GRID_OSCILLATION_STEP_L2_MIN = 0.30
DEFAULT_W_CROSS_THRESHOLD = 0.03
DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID = 2
DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN = 0.25
DEFAULT_CROSS_WINDOW_W_MIN = 0.05
DEFAULT_CROSS_WINDOW_EXCEPTION_AMPLITUDE_MAX = 0.2
DEFAULT_CROSS_WINDOW_EXCEPTION_K_MAX = 0.1
DEFAULT_CROSS_WINDOW_EXCEPTION_HALF_WIDTH_GRID = 4
DEFAULT_CROSS_WINDOW_EXCEPTION_CALIBRATED_N_RADIAL = 201
DEFAULT_EDGE_R_MIN = 0.97
DEFAULT_EDGE_WIDTH_MAX_GRID = 10.0
DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX = 0.5
DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID = 2.0
DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN = 0.03
DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX = 0.50
DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX = 0.02
DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN = 0.001
DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX = 0.04
DEFAULT_INTERIOR_HARMONIC_CORE_R_MAX = 0.5
DEFAULT_INTERIOR_HARMONIC_ACTIVE_CORE_ENERGY_FRACTION_MIN = 0.005
DEFAULT_INTERIOR_HARMONIC_MAX_LAG_GRID = 5
DEFAULT_INTERIOR_HARMONIC_INCOHERENCE_SCORE_THRESHOLD = 0.10
DEFAULT_INTERIOR_HARMONIC_CALIBRATED_N_RADIAL = 201
DEFAULT_CONTINUUM_CROSSING_TAIL_K_MIN = 0.4
DEFAULT_CONTINUUM_CROSSING_TAIL_TOP2_RATIO_MIN = 0.035
DEFAULT_CONTINUUM_CROSSING_TAIL_HALF_WIDTH_GRID = 4
DEFAULT_CONTINUUM_CROSSING_TAIL_CALIBRATED_N_RADIAL = 201
HARMONIC_PARTICIPATION_EFFECTIVE_COUNT_THRESHOLD = 3.0

LEGACY_RULESET_VERSION = (
    "tae-rules-axis-all-peaks-grid-highr-packet-turns-rle05-near-axis-"
    "grid-oscillation-cont-window-edge-interior-envelope-harmonic-incoherence-"
    "continuum-crossing-tail-v18"
)
PREVIOUS_RULESET_VERSION = (
    LEGACY_RULESET_VERSION.removesuffix("-v18") + "-smooth-crossing-window-v19"
)
CLEARANCE_RULESET_VERSION = PREVIOUS_RULESET_VERSION.removesuffix("-v19") + "-extremum-clearance-v20"
AXIS_ENERGY_RULESET_VERSION = CLEARANCE_RULESET_VERSION.removesuffix("-v20") + "-axis-energy-concentration-v21"
RULESET_VERSION = AXIS_ENERGY_RULESET_VERSION.removesuffix("-v21") + "-extended-continuum-noise-v22"
BAD_AXIS_SPIKE = "BAD_AXIS_SPIKE"
BAD_AXIS_ENERGY_CONCENTRATION = "BAD_AXIS_ENERGY_CONCENTRATION"
BAD_GRID_SCALE_SPIKE = "BAD_GRID_SCALE_SPIKE"
BAD_GRID_SCALE_PACKET = "BAD_GRID_SCALE_PACKET"
BAD_NEAR_AXIS_GRID_OSCILLATION = "BAD_NEAR_AXIS_GRID_OSCILLATION"
BAD_CONT_CROSS = "BAD_CONT_CROSS"
BAD_CONT_CROSS_WINDOW = "BAD_CONT_CROSS_WINDOW"
BAD_EDGE_SPIKE = "BAD_EDGE_SPIKE"
BAD_INTERIOR_UNRESOLVED_ENVELOPE = "BAD_INTERIOR_UNRESOLVED_ENVELOPE"
BAD_INTERIOR_HARMONIC_INCOHERENCE = "BAD_INTERIOR_HARMONIC_INCOHERENCE"
BAD_CONTINUUM_CROSSING_TAIL = "BAD_CONTINUUM_CROSSING_TAIL"
NO_GOOD_TEMPLATE = "NO_GOOD_TEMPLATE"
RULE_FEATURE_EXTRACTION_FAILED = "RULE_FEATURE_EXTRACTION_FAILED"
RULE_FEATURE_NAMES = tuple(
    get_feature_names(include_crossing_features=True, include_extremum_features=True)
)
RULE_FEATURE_SCHEMA_VERSION = "tae-rule-features-grouped-v22"
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
class AxisEnergyConcentrationConfig:
    """Width-independent amplitude and integrated-energy cuts near the axis."""

    amplitude_r_max: float = DEFAULT_AXIS_ENERGY_AMPLITUDE_R_MAX
    amplitude_min: float | None = DEFAULT_AXIS_ENERGY_AMPLITUDE_MIN
    energy_r_max: float = DEFAULT_AXIS_ENERGY_R_MAX
    energy_fraction_min: float | None = DEFAULT_AXIS_ENERGY_FRACTION_MIN

    def __post_init__(self) -> None:
        for name in ("amplitude_r_max", "energy_r_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"axis energy {name} must be finite and in (0, 1]")
        for name in ("amplitude_min", "energy_fraction_min"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"axis energy {name} must be null or finite and in [0, 1]")

    @property
    def enabled(self) -> bool:
        return self.amplitude_min is not None and self.energy_fraction_min is not None


def empty_axis_energy_concentration_features(
    config: AxisEnergyConcentrationConfig | None = None,
) -> dict[str, Any]:
    resolved = config or AxisEnergyConcentrationConfig()
    return {
        "enabled": resolved.enabled,
        "amplitude_r_max": resolved.amplitude_r_max,
        "amplitude_min": resolved.amplitude_min,
        "energy_r_max": resolved.energy_r_max,
        "energy_fraction_min": resolved.energy_fraction_min,
        "n_radial": None,
        "axis_sample_count": None,
        "axis_amplitude": None,
        "axis_signed_amplitude": None,
        "axis_harmonic_index": None,
        "axis_peak_r": None,
        "total_energy": None,
        "inner_energy_fraction": None,
        "candidate_found": None,
    }


def extract_axis_energy_concentration_features(
    mode: np.ndarray, *, config: AxisEnergyConcentrationConfig | None = None,
) -> dict[str, Any]:
    """Measure native samples and piecewise-linear all-harmonic W energy."""
    resolved = config or AxisEnergyConcentrationConfig()
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError("mode must have shape (n_harmonics, n_radial), n_radial >= 2")
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")
    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    indices = np.flatnonzero(radial_grid <= resolved.amplitude_r_max)
    window = mode_array[:, indices]
    h, j = np.unravel_index(np.argmax(np.abs(window)), window.shape)
    amplitude = float(abs(window[h, j]))
    W = np.sum(mode_array**2, axis=0)
    total = float(np.trapezoid(W, radial_grid))
    if not math.isfinite(total):
        raise ValueError("axis energy integral is non-finite")
    # A zero-energy input has no defined energy fraction and cannot qualify.
    fraction = (
        _energy_fraction_in_window(
            W, radial_grid, resolved.energy_r_max / 2, resolved.energy_r_max / 2
        ) if total > 0.0 else None
    )
    result = empty_axis_energy_concentration_features(resolved)
    result.update(
        n_radial=mode_array.shape[1], axis_sample_count=int(indices.size),
        axis_amplitude=amplitude, axis_signed_amplitude=float(window[h, j]),
        axis_harmonic_index=int(h), axis_peak_r=float(radial_grid[indices[j]]),
        total_energy=total, inner_energy_fraction=fraction,
        candidate_found=bool(
            resolved.enabled and fraction is not None
            and amplitude > resolved.amplitude_min
            and fraction > resolved.energy_fraction_min
        ),
    )
    return result


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
class NearAxisGridOscillationConfig:
    """Thresholds for consecutive sign-flip oscillations near the axis."""

    peak_r_max: float = DEFAULT_NEAR_AXIS_GRID_OSCILLATION_PEAK_R_MAX
    amplitude_min: float | None = DEFAULT_NEAR_AXIS_GRID_OSCILLATION_AMPLITUDE_MIN
    min_consecutive_sign_flips: int = (
        DEFAULT_NEAR_AXIS_GRID_OSCILLATION_MIN_CONSECUTIVE_SIGN_FLIPS
    )
    step_l2_min: float | None = DEFAULT_NEAR_AXIS_GRID_OSCILLATION_STEP_L2_MIN

    def __post_init__(self) -> None:
        if not math.isfinite(self.peak_r_max) or not 0.0 < self.peak_r_max <= 1.0:
            raise ValueError(
                "near_axis_grid_oscillation peak_r_max must be finite and in (0, 1]"
            )
        if self.amplitude_min is not None and (
            not math.isfinite(self.amplitude_min)
            or not 0.0 <= self.amplitude_min <= 1.0
        ):
            raise ValueError(
                "near_axis_grid_oscillation amplitude_min must be null or "
                "finite and in [0, 1]"
            )
        if (
            isinstance(self.min_consecutive_sign_flips, bool)
            or not isinstance(self.min_consecutive_sign_flips, int)
            or self.min_consecutive_sign_flips < 1
        ):
            raise ValueError(
                "near_axis_grid_oscillation min_consecutive_sign_flips must "
                "be a positive integer"
            )
        if self.step_l2_min is not None and (
            not math.isfinite(self.step_l2_min) or self.step_l2_min < 0.0
        ):
            raise ValueError(
                "near_axis_grid_oscillation step_l2_min must be null or a "
                "finite nonnegative number"
            )

    @property
    def enabled(self) -> bool:
        """Return whether both magnitude thresholds are configured."""
        return self.amplitude_min is not None and self.step_l2_min is not None


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
    exception_amplitude_max: float | None = DEFAULT_CROSS_WINDOW_EXCEPTION_AMPLITUDE_MAX
    exception_k_max: float | None = DEFAULT_CROSS_WINDOW_EXCEPTION_K_MAX
    exception_half_width_grid: int = DEFAULT_CROSS_WINDOW_EXCEPTION_HALF_WIDTH_GRID
    exception_calibrated_n_radial: int = (
        DEFAULT_CROSS_WINDOW_EXCEPTION_CALIBRATED_N_RADIAL
    )

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
            ("exception_amplitude_max", self.exception_amplitude_max),
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"cross_window {name} must be null or finite and in [0, 1]"
                )
        if self.exception_k_max is not None and (
            not math.isfinite(self.exception_k_max) or self.exception_k_max < 0
        ):
            raise ValueError(
                "cross_window exception_k_max must be null or finite and nonnegative"
            )
        for name, value, minimum in (
            ("exception_half_width_grid", self.exception_half_width_grid, 0),
            ("exception_calibrated_n_radial", self.exception_calibrated_n_radial, 3),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"cross_window {name} must be an integer >= {minimum}")

    @property
    def exception_enabled(self) -> bool:
        return (
            self.exception_amplitude_max is not None
            and self.exception_k_max is not None
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
    ext_df_gap_min_inclusive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ext_df_gap_min_inclusive, bool):
            raise ValueError("ext_df_gap_min_inclusive must be a boolean")
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


@dataclass(frozen=True)
class InteriorHarmonicIncoherenceConfig:
    """Thresholds for incoherent harmonic activity in the core."""

    core_r_max: float = DEFAULT_INTERIOR_HARMONIC_CORE_R_MAX
    active_core_energy_fraction_min: float = (
        DEFAULT_INTERIOR_HARMONIC_ACTIVE_CORE_ENERGY_FRACTION_MIN
    )
    max_lag_grid: int = DEFAULT_INTERIOR_HARMONIC_MAX_LAG_GRID
    score_threshold: float | None = (
        DEFAULT_INTERIOR_HARMONIC_INCOHERENCE_SCORE_THRESHOLD
    )
    calibrated_n_radial: int = DEFAULT_INTERIOR_HARMONIC_CALIBRATED_N_RADIAL

    def __post_init__(self) -> None:
        if not math.isfinite(self.core_r_max) or not 0.0 <= self.core_r_max <= 1.0:
            raise ValueError(
                "interior harmonic incoherence core_r_max must be finite and in [0, 1]"
            )
        if (
            not math.isfinite(self.active_core_energy_fraction_min)
            or not 0.0 < self.active_core_energy_fraction_min <= 1.0
        ):
            raise ValueError(
                "interior harmonic incoherence active_core_energy_fraction_min "
                "must be finite and in (0, 1]"
            )
        if isinstance(self.max_lag_grid, bool) or not isinstance(
            self.max_lag_grid, int
        ):
            raise ValueError(
                "interior harmonic incoherence max_lag_grid must be an integer"
            )
        if self.max_lag_grid < 0:
            raise ValueError(
                "interior harmonic incoherence max_lag_grid must be nonnegative"
            )
        if self.score_threshold is not None and (
            not math.isfinite(self.score_threshold) or self.score_threshold < 0.0
        ):
            raise ValueError(
                "interior harmonic incoherence score_threshold must be null or a "
                "finite nonnegative number"
            )
        if isinstance(self.calibrated_n_radial, bool) or not isinstance(
            self.calibrated_n_radial, int
        ):
            raise ValueError(
                "interior harmonic incoherence calibrated_n_radial must be an integer"
            )
        if self.calibrated_n_radial < 2:
            raise ValueError(
                "interior harmonic incoherence calibrated_n_radial must be at least 2"
            )

    @property
    def enabled(self) -> bool:
        """Return whether the combined-score decision threshold is configured."""
        return self.score_threshold is not None


@dataclass(frozen=True)
class ContinuumCrossingTailConfig:
    """Same-crossing signed roughness and strongest-two-harmonic tail ratio."""

    k_min: float | None = DEFAULT_CONTINUUM_CROSSING_TAIL_K_MIN
    top2_ratio_min: float = DEFAULT_CONTINUUM_CROSSING_TAIL_TOP2_RATIO_MIN
    half_width_grid: int = DEFAULT_CONTINUUM_CROSSING_TAIL_HALF_WIDTH_GRID
    calibrated_n_radial: int = DEFAULT_CONTINUUM_CROSSING_TAIL_CALIBRATED_N_RADIAL

    def __post_init__(self) -> None:
        for name in ("k_min", "top2_ratio_min"):
            value = getattr(self, name)
            if name == "k_min" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"continuum crossing tail {name} must be finite and nonnegative"
                )
        for name, minimum in (("half_width_grid", 0), ("calibrated_n_radial", 3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"continuum crossing tail {name} must be an integer >= {minimum}"
                )

    @property
    def enabled(self) -> bool:
        return self.k_min is not None


def empty_continuum_crossing_tail_features(
    config: ContinuumCrossingTailConfig | None = None,
) -> dict[str, Any]:
    resolved = config or ContinuumCrossingTailConfig()
    return {
        "k_min": resolved.k_min,
        "top2_ratio_min": resolved.top2_ratio_min,
        "half_width_grid": resolved.half_width_grid,
        "calibrated_n_radial": resolved.calibrated_n_radial,
        "n_radial": None,
        "resolution_eligible": None,
        "candidate_found": None,
        "n_qualifying_crossings": 0,
        "energy_peak_r": None,
        "total_energy": None,
        "top2_energy": None,
        "top2_energy_share": None,
        "top2_harmonic_indices": [],
        "records": [],
        "witness": None,
    }


def _crossing_roughness(
    normalized_mode: np.ndarray,
    second_differences: np.ndarray,
    radial_grid: np.ndarray,
    r_cross: float,
    half_width_grid: int,
) -> dict[str, float | int | None]:
    """Shared native-grid K for the window exception and crossing-tail gate."""
    centers = np.abs(radial_grid[1:-1] - r_cross) <= half_width_grid * (
        radial_grid[1] - radial_grid[0]
    )
    denominator = float(np.sum(normalized_mode[:, 1:-1][:, centers] ** 2))
    numerator = float(np.sum(second_differences[:, centers] ** 2))
    return {
        "K_cross": math.sqrt(numerator / denominator) if denominator > 0 else None,
        "n_centers": int(np.count_nonzero(centers)),
        "second_difference_energy": numerator,
        "local_amplitude_energy": denominator,
    }


def extract_continuum_crossing_tail_features(
    mode: np.ndarray,
    crossing_records: list[Mapping[str, Any]],
    *,
    config: ContinuumCrossingTailConfig | None = None,
) -> dict[str, Any]:
    """Measure cumulative tail energy and native-grid roughness at each crossing.

    Energy is trapezoidal integral(sum_h A_h**2, dr), with global peak
    amplitude normalized to one. The reference denominator is the full-domain
    energy of the two strongest individual harmonics (no adjacency constraint).
    The tail is the side opposite the global W peak. K uses unscaled signed
    second differences at complete stencil centers within +/-4 native intervals;
    its stencil can extend one interval beyond that window. Both strict cuts
    must hold at one crossing. Other resolutions retain evidence and fail open.
    """
    resolved = config or ContinuumCrossingTailConfig()
    A = np.asarray(mode, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.isfinite(A).all():
        raise ValueError("mode contains non-finite values")
    peak = float(np.max(np.abs(A)))
    A = A if peak == 0 else A / peak
    r = np.linspace(0.0, 1.0, A.shape[1])
    W = np.sum(A**2, axis=0)
    total = float(np.trapezoid(W, r))
    harmonic_energy = np.trapezoid(A**2, r, axis=1)
    # Stable ties choose the lower stored index; adding zero rows has no effect.
    top2_indices = np.argsort(-harmonic_energy, kind="stable")[:2]
    top2 = float(np.sum(harmonic_energy[top2_indices]))
    r_peak = float(r[np.argmax(W)]) if total > 0 else None
    result = empty_continuum_crossing_tail_features(resolved)
    result.update(
        {
            "n_radial": len(r),
            "resolution_eligible": len(r) == resolved.calibrated_n_radial,
            "candidate_found": False if resolved.enabled else None,
            "energy_peak_r": r_peak,
            "total_energy": total,
            "top2_energy": top2,
            "top2_energy_share": top2 / total if total > 0 else None,
            "top2_harmonic_indices": (
                [int(h) for h in top2_indices] if total > 0 else []
            ),
        }
    )
    d2 = A[:, 2:] - 2 * A[:, 1:-1] + A[:, :-2]
    for crossing in sorted(
        crossing_records, key=lambda row: (row["r_cross"], row["boundary"])
    ):
        rc = float(crossing["r_cross"])
        boundary = crossing["boundary"]
        if boundary not in {"low", "high"} or not math.isfinite(rc) or not 0 <= rc <= 1:
            raise ValueError(
                "continuum crossing tail requires a finite lower/upper crossing in [0, 1]"
            )
        roughness = _crossing_roughness(A, d2, r, rc, resolved.half_width_grid)
        k = roughness["K_cross"]
        inner, outer, fraction, ratio, side = None, None, None, None, None
        if total > 0:
            samples = np.r_[0.0, r[(r > 0) & (r < rc)], rc]
            inner = float(np.trapezoid(np.interp(samples, r, W), samples)) / total
            outer = 1.0 - inner
            side = "inner" if rc < r_peak else "outer"
            fraction = inner if side == "inner" else outer
            ratio = fraction * total / top2
        thresholds_pass = (
            resolved.enabled
            and k is not None
            and ratio is not None
            and k > resolved.k_min
            and ratio > resolved.top2_ratio_min
        )
        result["records"].append(
            {
                "boundary": boundary,
                "r_cross": rc,
                "tail_side": side,
                "energy_fraction_inner": inner,
                "energy_fraction_outer": outer,
                "tail_fraction": fraction,
                "tail_energy": fraction * total if fraction is not None else None,
                "tail_over_top2": ratio,
                **roughness,
                "thresholds_pass": bool(thresholds_pass) if resolved.enabled else None,
                "candidate_found": bool(
                    thresholds_pass and result["resolution_eligible"]
                ),
            }
        )
    qualified = [row for row in result["records"] if row["candidate_found"]]
    result["n_qualifying_crossings"] = len(qualified)
    if qualified:
        result["candidate_found"] = True
        # Select from crossings satisfying BOTH cuts, never independent maxima.
        result["witness"] = dict(max(qualified, key=lambda row: row["K_cross"]))
    return result


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


def empty_near_axis_grid_oscillation_features(
    config: NearAxisGridOscillationConfig | None = None,
    *,
    candidate_found: bool | None = None,
    qualifying_run_count: int | None = None,
    step_l2_qualified_run_count: int | None = None,
) -> dict[str, Any]:
    """Return the stable near-axis sign-flip evidence shape."""
    resolved = config or NearAxisGridOscillationConfig()
    return {
        "candidate_found": candidate_found,
        "peak_r_max_exclusive": resolved.peak_r_max,
        "amplitude_min": resolved.amplitude_min,
        "min_consecutive_sign_flips": resolved.min_consecutive_sign_flips,
        "step_l2_min": resolved.step_l2_min,
        "qualifying_run_count": qualifying_run_count,
        "step_l2_qualified_run_count": step_l2_qualified_run_count,
        "near_axis_peak": None,
        "near_axis_peak_signed_amplitude": None,
        "near_axis_peak_harmonic_index": None,
        "near_axis_peak_r": None,
        "near_axis_peak_amplitude_qualified": None,
        "selected_run_harmonic_index": None,
        "selected_run_start_index": None,
        "selected_run_end_index": None,
        "selected_run_start_r": None,
        "selected_run_end_r": None,
        "selected_run_peak": None,
        "selected_run_peak_signed_amplitude": None,
        "selected_run_peak_r": None,
        "selected_run_consecutive_sign_flip_count": None,
        "selected_run_step_l2": None,
        "selected_run_total_variation": None,
        "selected_run_max_step": None,
        "selected_run_mean_abs_step": None,
        "near_axis_peak_and_run_same_harmonic": None,
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
        "ext_df_gap_min_inclusive": resolved.ext_df_gap_min_inclusive,
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


def empty_interior_harmonic_incoherence_features(
    config: InteriorHarmonicIncoherenceConfig | None = None,
    *,
    candidate_found: bool | None = None,
) -> dict[str, Any]:
    """Return the stable interior harmonic-incoherence audit shape."""
    resolved = config or InteriorHarmonicIncoherenceConfig()
    return {
        "core_r_max": resolved.core_r_max,
        "active_core_energy_fraction_min": (
            resolved.active_core_energy_fraction_min
        ),
        "max_lag_grid": resolved.max_lag_grid,
        "score_threshold": resolved.score_threshold,
        "calibrated_n_radial": resolved.calibrated_n_radial,
        "resolution_eligible": None,
        "candidate_found": candidate_found,
        "core_radial_sample_count": None,
        "core_positive_energy_sample_count": None,
        "core_adjacent_radial_pair_count": None,
        "active_core_harmonic_count": None,
        "active_adjacent_harmonic_pair_count": None,
        "core_energy_fraction": None,
        "core_js_divergence": None,
        "core_effective_harmonic_count_wmean": None,
        "effective_harmonic_count_threshold": (
            HARMONIC_PARTICIPATION_EFFECTIVE_COUNT_THRESHOLD
        ),
        "global_effective_harmonic_count_wmean": None,
        "global_energy_fraction_above_effective_harmonic_count_threshold": None,
        "core_energy_fraction_above_effective_harmonic_count_threshold": None,
        "total_energy_fraction_in_core_above_effective_harmonic_count_threshold": (
            None
        ),
        "core_adjacent_harmonic_coherence": None,
        "incoherence_score": None,
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
    near_axis_grid_oscillation_config: NearAxisGridOscillationConfig | None = None,
    interior_unresolved_envelope_config: InteriorUnresolvedEnvelopeConfig | None = None,
    interior_harmonic_incoherence_config: (
        InteriorHarmonicIncoherenceConfig | None
    ) = None,
    continuum_crossing_tail_config: ContinuumCrossingTailConfig | None = None,
    axis_energy_concentration_config: AxisEnergyConcentrationConfig | None = None,
    continuum_noise_config: ContinuumNoiseThresholds | None = None,
) -> dict[str, Any]:
    """Return the complete rule-feature schema with unavailable values as null."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    packet_config = grid_scale_packet_config or GridScalePacketConfig()
    near_axis_oscillation_config = (
        near_axis_grid_oscillation_config or NearAxisGridOscillationConfig()
    )
    edge_config = edge_artifact_config or EdgeArtifactConfig()
    interior_config = (
        interior_unresolved_envelope_config or InteriorUnresolvedEnvelopeConfig()
    )
    incoherence_config = (
        interior_harmonic_incoherence_config
        or InteriorHarmonicIncoherenceConfig()
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
            "extended_continuum_noise": empty_continuum_noise_features(continuum_noise_config),
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
            "near_axis_grid_oscillation": (
                empty_near_axis_grid_oscillation_features(
                    near_axis_oscillation_config
                )
            ),
            "interior_harmonic_incoherence": (
                empty_interior_harmonic_incoherence_features(
                    incoherence_config
                )
            ),
        },
        "crossing_features": {
            **{name: None for name in EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES},
            "continuum_crossing_tail": empty_continuum_crossing_tail_features(
                continuum_crossing_tail_config
            ),
            **empty_continuum_crossing_window_features(
                cross_window_config.half_width_grid
            ),
            "continuum_crossing_window_exception": empty_crossing_window_exception_features(
                cross_window_config
            ),
        },
        "crossing_records": [],
        "extremum_features": {
            "match_found": None,
            **{name: None for name in EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES},
        },
        "boundary_features": {
            "axis_energy_concentration": empty_axis_energy_concentration_features(
                axis_energy_concentration_config
            ),
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
    near_axis_grid_oscillation_features: Mapping[str, Any],
    edge_artifact_features: Mapping[str, Any],
    continuum_crossing_window_features: Mapping[str, Any],
    interior_unresolved_envelope_features: Mapping[str, Any],
    interior_harmonic_incoherence_features: Mapping[str, Any],
    continuum_crossing_tail_features: Mapping[str, Any],
    axis_energy_concentration_features: Mapping[str, Any],
    continuum_noise_features: Mapping[str, Any],
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
            "extended_continuum_noise": dict(continuum_noise_features),
            "grid_scale_spike": dict(grid_scale_spike_features),
            "grid_scale_packet": dict(grid_scale_packet_features),
            "near_axis_grid_oscillation": dict(
                near_axis_grid_oscillation_features
            ),
            "interior_harmonic_incoherence": dict(
                interior_harmonic_incoherence_features
            ),
        },
        "crossing_features": {
            **{
                name: named_features[name]
                for name in EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES
            },
            **dict(continuum_crossing_window_features),
            "continuum_crossing_tail": dict(continuum_crossing_tail_features),
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
            "axis_energy_concentration": dict(axis_energy_concentration_features),
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


def extract_near_axis_grid_oscillation_features(
    mode: np.ndarray,
    *,
    peak_r_max: float = DEFAULT_NEAR_AXIS_GRID_OSCILLATION_PEAK_R_MAX,
    amplitude_min: float | None = (
        DEFAULT_NEAR_AXIS_GRID_OSCILLATION_AMPLITUDE_MIN
    ),
    min_consecutive_sign_flips: int = (
        DEFAULT_NEAR_AXIS_GRID_OSCILLATION_MIN_CONSECUTIVE_SIGN_FLIPS
    ),
    step_l2_min: float | None = DEFAULT_NEAR_AXIS_GRID_OSCILLATION_STEP_L2_MIN,
) -> dict[str, Any]:
    """Measure strict consecutive sign-flip runs whose peak is near the axis.

    ``near_axis_peak`` is the largest absolute sample from any harmonic at
    strictly ``r < peak_r_max``. Separately, each harmonic is divided into
    maximal runs of strictly consecutive nonzero sign changes. Runs with at
    least ``min_consecutive_sign_flips`` are eligible only when their own
    largest absolute sample is also at strictly ``r < peak_r_max``. The
    selected run has the largest
    ``sqrt(sum((A[i+1] - A[i])**2))`` and is never combined with another
    harmonic or across a missing sign flip.
    """
    config = NearAxisGridOscillationConfig(
        peak_r_max=peak_r_max,
        amplitude_min=amplitude_min,
        min_consecutive_sign_flips=min_consecutive_sign_flips,
        step_l2_min=step_l2_min,
    )
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")

    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    near_axis_indices = np.flatnonzero(radial_grid < config.peak_r_max)
    if near_axis_indices.size == 0:
        return empty_near_axis_grid_oscillation_features(
            config,
            candidate_found=False,
            qualifying_run_count=0,
            step_l2_qualified_run_count=(
                None if config.step_l2_min is None else 0
            ),
        )

    near_axis_values = np.abs(mode_array[:, near_axis_indices])
    flat_peak_index = int(np.argmax(near_axis_values))
    peak_harmonic_index, peak_offset = np.unravel_index(
        flat_peak_index, near_axis_values.shape
    )
    near_axis_peak_index = int(near_axis_indices[peak_offset])
    near_axis_peak_signed_amplitude = float(
        mode_array[peak_harmonic_index, near_axis_peak_index]
    )
    near_axis_peak = abs(near_axis_peak_signed_amplitude)
    amplitude_tolerance = (
        None
        if config.amplitude_min is None
        else 64.0
        * np.finfo(float).eps
        * max(1.0, config.amplitude_min)
    )
    near_axis_peak_amplitude_qualified = (
        None
        if config.amplitude_min is None or amplitude_tolerance is None
        else bool(
            near_axis_peak >= config.amplitude_min - amplitude_tolerance
        )
    )

    runs: list[dict[str, Any]] = []
    for harmonic_index, profile in enumerate(mode_array):
        left = profile[:-1]
        right = profile[1:]
        sign_flips = (
            (left != 0.0)
            & (right != 0.0)
            & (np.signbit(left) != np.signbit(right))
        )
        padded = np.concatenate(([False], sign_flips, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends_exclusive = np.flatnonzero(padded[:-1] & ~padded[1:])
        for raw_start, raw_end_exclusive in zip(starts, ends_exclusive):
            start_index = int(raw_start)
            end_index = int(raw_end_exclusive)
            sign_flip_count = end_index - start_index
            if sign_flip_count < config.min_consecutive_sign_flips:
                continue
            values = profile[start_index : end_index + 1]
            run_peak_offset = int(np.argmax(np.abs(values)))
            run_peak_index = start_index + run_peak_offset
            run_peak_r = float(radial_grid[run_peak_index])
            if run_peak_r >= config.peak_r_max:
                continue
            signed_steps = np.diff(values)
            step_magnitudes = np.abs(signed_steps)
            run_peak_signed_amplitude = float(values[run_peak_offset])
            runs.append(
                {
                    "selected_run_harmonic_index": int(harmonic_index),
                    "selected_run_start_index": start_index,
                    "selected_run_end_index": end_index,
                    "selected_run_start_r": float(radial_grid[start_index]),
                    "selected_run_end_r": float(radial_grid[end_index]),
                    "selected_run_peak": abs(run_peak_signed_amplitude),
                    "selected_run_peak_signed_amplitude": (
                        run_peak_signed_amplitude
                    ),
                    "selected_run_peak_r": run_peak_r,
                    "selected_run_consecutive_sign_flip_count": (
                        sign_flip_count
                    ),
                    "selected_run_step_l2": float(np.linalg.norm(signed_steps)),
                    "selected_run_total_variation": float(
                        np.sum(step_magnitudes)
                    ),
                    "selected_run_max_step": float(np.max(step_magnitudes)),
                    "selected_run_mean_abs_step": float(
                        np.mean(step_magnitudes)
                    ),
                }
            )

    step_l2_tolerance = (
        None
        if config.step_l2_min is None
        else 64.0
        * np.finfo(float).eps
        * max(1.0, config.step_l2_min)
    )
    step_l2_qualified_run_count = (
        None
        if config.step_l2_min is None or step_l2_tolerance is None
        else sum(
            run["selected_run_step_l2"]
            >= config.step_l2_min - step_l2_tolerance
            for run in runs
        )
    )
    base = empty_near_axis_grid_oscillation_features(
        config,
        candidate_found=False,
        qualifying_run_count=len(runs),
        step_l2_qualified_run_count=step_l2_qualified_run_count,
    )
    base.update(
        {
            "near_axis_peak": near_axis_peak,
            "near_axis_peak_signed_amplitude": near_axis_peak_signed_amplitude,
            "near_axis_peak_harmonic_index": int(peak_harmonic_index),
            "near_axis_peak_r": float(radial_grid[near_axis_peak_index]),
            "near_axis_peak_amplitude_qualified": (
                near_axis_peak_amplitude_qualified
            ),
        }
    )
    if not runs:
        return base

    selected = min(
        runs,
        key=lambda run: (
            -run["selected_run_step_l2"],
            -run["selected_run_consecutive_sign_flip_count"],
            -run["selected_run_peak"],
            run["selected_run_harmonic_index"],
            run["selected_run_start_index"],
        ),
    )
    base.update(selected)
    base["near_axis_peak_and_run_same_harmonic"] = bool(
        peak_harmonic_index == selected["selected_run_harmonic_index"]
    )
    if config.enabled and step_l2_tolerance is not None:
        base["candidate_found"] = bool(
            near_axis_peak_amplitude_qualified
            and selected["selected_run_step_l2"]
            >= config.step_l2_min - step_l2_tolerance
        )
    return base


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
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Measure harmonic amplitude and normalized energy near true crossings.

    The inclusive radial window around each interpolated crossing is
    ``abs(r_i - r_cross) <= half_width_grid * delta_r``. Winners are selected
    independently for individual-harmonic absolute amplitude and total radial
    energy so both kinds of evidence remain auditable.
    ``allow_empty`` supports per-crossing diagnostics for zero-width
    calibration windows whose crossing lies between radial samples.
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
        if allow_empty:
            return result
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


def empty_crossing_window_exception_features(
    config: ContinuumCrossingWindowConfig,
) -> dict[str, Any]:
    return {
        "enabled": config.exception_enabled,
        "amplitude_max": config.exception_amplitude_max,
        "k_max": config.exception_k_max,
        "half_width_grid": config.exception_half_width_grid,
        "calibrated_n_radial": config.exception_calibrated_n_radial,
        "n_radial": None,
        "resolution_eligible": None,
        "n_violating_crossings": 0,
        "n_exempted_crossings": 0,
        "n_unexcused_crossings": 0,
        "all_violations_exempted": False,
        "records": [],
    }


def extract_crossing_window_exception_features(
    mode: np.ndarray,
    crossing_records: list[Mapping[str, Any]],
    *,
    config: ContinuumCrossingWindowConfig | None = None,
) -> dict[str, Any]:
    """Excuse only individual violating windows with low point amplitude and K.

    A_cross interpolates every signed harmonic before taking its magnitude.
    The original window maxima remain unchanged. Undefined K or an ineligible
    grid never grants an exception; another offending crossing still rejects.
    """
    resolved = config or ContinuumCrossingWindowConfig()
    A = np.asarray(mode, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 2 or not np.isfinite(A).all():
        raise ValueError("mode must be finite with shape (n_harmonics, n_radial>=2)")
    r = np.linspace(0, 1, A.shape[1])
    peak = float(np.max(np.abs(A)))
    normalized = A if peak == 0 else A / peak
    d2 = normalized[:, 2:] - 2 * normalized[:, 1:-1] + normalized[:, :-2]
    result = empty_crossing_window_exception_features(resolved)
    eligible = A.shape[1] == resolved.exception_calibrated_n_radial
    result.update(n_radial=A.shape[1], resolution_eligible=eligible)
    for crossing in sorted(
        crossing_records, key=lambda x: (x["r_cross"], x["boundary"])
    ):
        window = extract_continuum_crossing_window_features(
            A, [crossing], half_width_grid=resolved.half_width_grid, allow_empty=True
        )
        violation = (
            resolved.amplitude_min is not None
            and window["cross_window_A_max"] is not None
            and window["cross_window_A_max"] >= resolved.amplitude_min
        ) or (
            resolved.w_min is not None
            and window["cross_window_W_max"] is not None
            and window["cross_window_W_max"] >= resolved.w_min
        )
        rc = crossing["r_cross"]
        signed = np.array([np.interp(rc, r, profile) for profile in A])
        amplitude = float(np.max(np.abs(signed)))
        roughness = _crossing_roughness(
            normalized, d2, r, rc, resolved.exception_half_width_grid
        )
        k = roughness["K_cross"]
        passes = bool(
            resolved.exception_enabled
            and eligible
            and k is not None
            and amplitude < resolved.exception_amplitude_max
            and k < resolved.exception_k_max
        )
        exempted = bool(violation and passes)
        result["records"].append(
            {
                "boundary": crossing["boundary"],
                "r_cross": rc,
                "A_cross": amplitude,
                "A_cross_harmonic_index": int(np.argmax(np.abs(signed))),
                **roughness,
                "window_A_max": window["cross_window_A_max"],
                "window_W_max": window["cross_window_W_max"],
                "window_violation": bool(violation),
                "exception_conditions_pass": passes,
                "exempted": exempted,
            }
        )
        result["n_violating_crossings"] += int(violation)
        result["n_exempted_crossings"] += int(exempted)
    result["n_unexcused_crossings"] = (
        result["n_violating_crossings"] - result["n_exempted_crossings"]
    )
    result["all_violations_exempted"] = bool(
        result["n_violating_crossings"] > 0 and result["n_unexcused_crossings"] == 0
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
        and (
            ext_df_gap >= resolved.ext_df_gap_min - ext_df_tolerance
            if resolved.ext_df_gap_min_inclusive
            else ext_df_gap > resolved.ext_df_gap_min
        )
        and ext_df_gap <= resolved.ext_df_gap_max + ext_df_tolerance
    )
    result["extremum_exception_applied"] = bool(
        candidate_found is True and exception_qualified
    )
    return result


def extract_interior_harmonic_incoherence_features(
    mode: np.ndarray,
    *,
    config: InteriorHarmonicIncoherenceConfig | None = None,
) -> dict[str, Any]:
    """Measure incoherent harmonic participation in the inclusive core.

    The combined score is

    ``f_core * J_core * N_eff_core * (1 - C_adj)``.

    Here ``J_core`` is the base-2 Jensen--Shannon divergence of adjacent
    squared-amplitude harmonic distributions, ``N_eff_core`` is their
    energy-weighted inverse participation ratio, and ``C_adj`` is the
    energy-weighted signed-profile coherence of adjacent stored harmonic
    rows after allowing a bounded radial lag.  Whole-radius ``N_eff`` and
    energy fractions above ``N_eff > 3`` are retained only as audit evidence.
    No physical poloidal-mode offset is inferred from the stored row index.
    """
    resolved = config or InteriorHarmonicIncoherenceConfig()
    mode_array = np.asarray(mode, dtype=float)
    if mode_array.ndim != 2 or mode_array.shape[0] < 1 or mode_array.shape[1] < 2:
        raise ValueError(
            "mode must have shape (n_harmonics, n_radial) with n_radial >= 2"
        )
    if not np.all(np.isfinite(mode_array)):
        raise ValueError("mode contains non-finite values")

    radial_grid = np.linspace(0.0, 1.0, mode_array.shape[1])
    radial_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(resolved.core_r_max)
    )
    core_mask = radial_grid <= resolved.core_r_max + radial_tolerance
    amplitude_scale = float(np.max(np.abs(mode_array)))
    scaled_mode = (
        mode_array if amplitude_scale == 0.0 else mode_array / amplitude_scale
    )
    core_mode = scaled_mode[:, core_mask]
    squared_mode = np.square(scaled_mode)
    radial_energy = np.sum(squared_mode, axis=0)
    core_radial_energy = radial_energy[core_mask]
    total_energy = float(np.sum(radial_energy))
    core_energy = float(np.sum(core_radial_energy))

    result = empty_interior_harmonic_incoherence_features(resolved)
    result.update(
        {
            "core_radial_sample_count": int(core_mode.shape[1]),
            "core_positive_energy_sample_count": int(
                np.count_nonzero(core_radial_energy > 0.0)
            ),
            "core_adjacent_radial_pair_count": 0,
            "active_core_harmonic_count": 0,
            "active_adjacent_harmonic_pair_count": 0,
            "resolution_eligible": bool(
                mode_array.shape[1] == resolved.calibrated_n_radial
            ),
            "candidate_found": None if not resolved.enabled else False,
        }
    )
    if total_energy <= 0.0:
        return result

    core_energy_fraction = min(1.0, max(0.0, core_energy / total_energy))
    result["core_energy_fraction"] = float(core_energy_fraction)

    positive_energy = radial_energy > 0.0
    probabilities = np.zeros_like(scaled_mode, dtype=float)
    probabilities[:, positive_energy] = (
        squared_mode[:, positive_energy] / radial_energy[positive_energy]
    )
    inverse_participation = np.sum(
        np.square(probabilities[:, positive_energy]), axis=0
    )
    effective_harmonic_count = np.zeros_like(radial_energy, dtype=float)
    effective_harmonic_count[positive_energy] = 1.0 / inverse_participation
    result["global_effective_harmonic_count_wmean"] = float(
        np.sum(radial_energy * effective_harmonic_count) / total_energy
    )

    above_threshold = (
        effective_harmonic_count
        > HARMONIC_PARTICIPATION_EFFECTIVE_COUNT_THRESHOLD
    )
    global_high_participation_energy = float(
        np.sum(radial_energy[above_threshold])
    )
    core_high_participation_energy = float(
        np.sum(radial_energy[core_mask & above_threshold])
    )
    result[
        "global_energy_fraction_above_effective_harmonic_count_threshold"
    ] = float(
        min(1.0, max(0.0, global_high_participation_energy / total_energy))
    )
    result[
        "total_energy_fraction_in_core_above_effective_harmonic_count_threshold"
    ] = float(
        min(1.0, max(0.0, core_high_participation_energy / total_energy))
    )
    if core_energy <= 0.0:
        return result

    result[
        "core_energy_fraction_above_effective_harmonic_count_threshold"
    ] = float(
        min(1.0, max(0.0, core_high_participation_energy / core_energy))
    )
    core_probabilities = probabilities[:, core_mask]
    core_effective_harmonic_count = float(
        np.sum(core_radial_energy * effective_harmonic_count[core_mask])
        / core_energy
    )
    result["core_effective_harmonic_count_wmean"] = (
        core_effective_harmonic_count
    )

    harmonic_core_energy = np.sum(np.square(core_mode), axis=1)
    harmonic_core_fraction = harmonic_core_energy / core_energy
    active_fraction_tolerance = 64.0 * np.finfo(float).eps
    active_harmonics = (harmonic_core_energy > 0.0) & (
        harmonic_core_fraction
        >= resolved.active_core_energy_fraction_min - active_fraction_tolerance
    )
    result["active_core_harmonic_count"] = int(
        np.count_nonzero(active_harmonics)
    )

    js_weight_sum = 0.0
    js_weighted_total = 0.0
    js_pair_count = 0
    for radial_index in range(core_mode.shape[1] - 1):
        left_energy = float(core_radial_energy[radial_index])
        right_energy = float(core_radial_energy[radial_index + 1])
        pair_weight = math.sqrt(left_energy * right_energy)
        if pair_weight <= 0.0:
            continue
        left = core_probabilities[:, radial_index]
        right = core_probabilities[:, radial_index + 1]
        midpoint = 0.5 * (left + right)
        left_nonzero = left > 0.0
        right_nonzero = right > 0.0
        divergence = 0.5 * (
            float(
                np.sum(
                    left[left_nonzero]
                    * np.log2(left[left_nonzero] / midpoint[left_nonzero])
                )
            )
            + float(
                np.sum(
                    right[right_nonzero]
                    * np.log2(right[right_nonzero] / midpoint[right_nonzero])
                )
            )
        )
        divergence = min(1.0, max(0.0, divergence))
        js_weighted_total += pair_weight * divergence
        js_weight_sum += pair_weight
        js_pair_count += 1
    result["core_adjacent_radial_pair_count"] = js_pair_count
    core_js_divergence: float | None = None
    if js_weight_sum > 0.0:
        core_js_divergence = min(
            1.0, max(0.0, js_weighted_total / js_weight_sum)
        )
        result["core_js_divergence"] = float(core_js_divergence)

    coherence_weight_sum = 0.0
    coherence_weighted_total = 0.0
    active_pair_count = 0
    n_core = core_mode.shape[1]
    for harmonic_index in range(core_mode.shape[0] - 1):
        if not (
            active_harmonics[harmonic_index]
            and active_harmonics[harmonic_index + 1]
        ):
            continue
        left_profile = core_mode[harmonic_index]
        right_profile = core_mode[harmonic_index + 1]
        pair_coherence: float | None = None
        for lag in range(-resolved.max_lag_grid, resolved.max_lag_grid + 1):
            if n_core - abs(lag) < 2:
                continue
            if lag < 0:
                left_overlap = left_profile[-lag:]
                right_overlap = right_profile[: n_core + lag]
            else:
                left_overlap = left_profile[: n_core - lag]
                right_overlap = right_profile[lag:]
            left_norm = float(np.linalg.norm(left_overlap))
            right_norm = float(np.linalg.norm(right_overlap))
            if left_norm <= 0.0 or right_norm <= 0.0:
                continue
            coherence = abs(
                float(np.dot(left_overlap, right_overlap))
                / (left_norm * right_norm)
            )
            coherence = min(1.0, max(0.0, coherence))
            if pair_coherence is None or coherence > pair_coherence:
                pair_coherence = coherence
        if pair_coherence is None:
            continue
        pair_weight = math.sqrt(
            float(harmonic_core_energy[harmonic_index])
            * float(harmonic_core_energy[harmonic_index + 1])
        )
        if pair_weight <= 0.0:
            continue
        coherence_weighted_total += pair_weight * pair_coherence
        coherence_weight_sum += pair_weight
        active_pair_count += 1

    result["active_adjacent_harmonic_pair_count"] = active_pair_count
    if coherence_weight_sum <= 0.0 or core_js_divergence is None:
        return result
    adjacent_coherence = min(
        1.0, max(0.0, coherence_weighted_total / coherence_weight_sum)
    )
    result["core_adjacent_harmonic_coherence"] = float(adjacent_coherence)

    incoherence_score = max(
        0.0,
        core_energy_fraction
        * core_js_divergence
        * core_effective_harmonic_count
        * (1.0 - adjacent_coherence),
    )
    result["incoherence_score"] = float(incoherence_score)
    if resolved.enabled and result["resolution_eligible"]:
        # The calibrated boundary is deliberately strict: equality passes.
        result["candidate_found"] = bool(
            incoherence_score > resolved.score_threshold
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
    axis_energy_concentration_config: AxisEnergyConcentrationConfig | None = None,
    grid_scale_spike_config: GridScaleSpikeConfig | None = None,
    grid_scale_packet_config: GridScalePacketConfig | None = None,
    near_axis_grid_oscillation_config: NearAxisGridOscillationConfig | None = None,
    continuum_crossing_config: ContinuumCrossingConfig | None = None,
    continuum_crossing_window_config: ContinuumCrossingWindowConfig | None = None,
    edge_artifact_config: EdgeArtifactConfig | None = None,
    interior_unresolved_envelope_config: InteriorUnresolvedEnvelopeConfig | None = None,
    interior_harmonic_incoherence_config: (
        InteriorHarmonicIncoherenceConfig | None
    ) = None,
    continuum_crossing_tail_config: ContinuumCrossingTailConfig | None = None,
    continuum_noise_config: ContinuumNoiseThresholds | None = None,
) -> RuleResult:
    """Extract named features and evaluate one valid, preprocessed TAE mode."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
    axis_energy_config = axis_energy_concentration_config or AxisEnergyConcentrationConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    packet_config = grid_scale_packet_config or GridScalePacketConfig()
    near_axis_oscillation_config = (
        near_axis_grid_oscillation_config or NearAxisGridOscillationConfig()
    )
    crossing_config = continuum_crossing_config or ContinuumCrossingConfig()
    cross_window_config = (
        continuum_crossing_window_config or ContinuumCrossingWindowConfig()
    )
    edge_config = edge_artifact_config or EdgeArtifactConfig()
    interior_config = (
        interior_unresolved_envelope_config or InteriorUnresolvedEnvelopeConfig()
    )
    incoherence_config = (
        interior_harmonic_incoherence_config
        or InteriorHarmonicIncoherenceConfig()
    )
    path = str(preprocessed_row.get("path", ""))
    tail_config = continuum_crossing_tail_config or ContinuumCrossingTailConfig()
    noise_config = continuum_noise_config or ContinuumNoiseThresholds()
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
                near_axis_oscillation_config,
                interior_config,
                incoherence_config,
                tail_config,
                axis_energy_config,
                noise_config,
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
        near_axis_grid_oscillation_features = (
            extract_near_axis_grid_oscillation_features(
                mode,
                peak_r_max=near_axis_oscillation_config.peak_r_max,
                amplitude_min=near_axis_oscillation_config.amplitude_min,
                min_consecutive_sign_flips=(
                    near_axis_oscillation_config.min_consecutive_sign_flips
                ),
                step_l2_min=near_axis_oscillation_config.step_l2_min,
            )
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
        cross_window_exception_features = extract_crossing_window_exception_features(
            mode, feature_status["crossing_records"], config=cross_window_config
        )
        cross_window_features["continuum_crossing_window_exception"] = (
            cross_window_exception_features
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
        interior_incoherence_features = (
            extract_interior_harmonic_incoherence_features(
                mode,
                config=incoherence_config,
            )
        )
        features = grouped_rule_features(
            named_features,
            feature_status,
            axis_features,
            grid_scale_features,
            grid_scale_packet_features,
            near_axis_grid_oscillation_features,
            edge_features,
            cross_window_features,
            interior_envelope_features,
            interior_incoherence_features,
            extract_continuum_crossing_tail_features(
                mode, feature_status["crossing_records"], config=tail_config
            ),
            extract_axis_energy_concentration_features(mode, config=axis_energy_config),
            extract_continuum_noise_features(mode, frequency, low2, high2, config=noise_config),
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
                near_axis_oscillation_config,
                interior_config,
                incoherence_config,
                tail_config,
                axis_energy_config,
                noise_config,
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

    if (
        near_axis_oscillation_config.enabled
        and near_axis_grid_oscillation_features["candidate_found"]
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
            primary_reason=BAD_NEAR_AXIS_GRID_OSCILLATION,
            triggered_rules=(BAD_NEAR_AXIS_GRID_OSCILLATION,),
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
        and not cross_window_exception_features["all_violations_exempted"]
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

    if (
        incoherence_config.enabled
        and interior_incoherence_features["candidate_found"]
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
            primary_reason=BAD_INTERIOR_HARMONIC_INCOHERENCE,
            triggered_rules=(BAD_INTERIOR_HARMONIC_INCOHERENCE,),
            features=features,
        )

    if (
        tail_config.enabled
        and features["crossing_features"]["continuum_crossing_tail"]["candidate_found"]
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
            primary_reason=BAD_CONTINUUM_CROSSING_TAIL,
            triggered_rules=(BAD_CONTINUUM_CROSSING_TAIL,),
            features=features,
        )

    # Append the gate so existing BAD primary reasons retain precedence.
    if features["boundary_features"]["axis_energy_concentration"]["candidate_found"]:
        return RuleResult(
            path=path, mode_key=mode_key, shot=shot, ntor=ntor,
            frequency=frequency, input_fingerprint=fingerprint, gap_region=gap_region,
            decision="BAD", primary_reason=BAD_AXIS_ENERGY_CONCENTRATION,
            triggered_rules=(BAD_AXIS_ENERGY_CONCENTRATION,), features=features,
        )

    if features["numerical_structure_features"]["extended_continuum_noise"]["candidate_found"]:
        return RuleResult(
            path=path, mode_key=mode_key, shot=shot, ntor=ntor,
            frequency=frequency, input_fingerprint=fingerprint, gap_region=gap_region,
            decision="BAD", primary_reason=BAD_EXTENDED_CONTINUUM_NOISE,
            triggered_rules=(BAD_EXTENDED_CONTINUUM_NOISE,), features=features,
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
