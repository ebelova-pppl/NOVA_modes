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

RULESET_VERSION = "tae-rules-axis-artifact-v2"
BAD_AXIS_SPIKE = "BAD_AXIS_SPIKE"
NO_GOOD_TEMPLATE = "NO_GOOD_TEMPLATE"
RULE_FEATURE_EXTRACTION_FAILED = "RULE_FEATURE_EXTRACTION_FAILED"
RULE_FEATURE_NAMES = tuple(
    get_feature_names(include_crossing_features=True, include_extremum_features=True)
)
RULE_FEATURE_SCHEMA_VERSION = "tae-rule-features-grouped-v4"
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


def empty_rule_features(
    axis_artifact_config: AxisArtifactConfig | None = None,
) -> dict[str, Any]:
    """Return the complete rule-feature schema with unavailable values as null."""
    config = axis_artifact_config or AxisArtifactConfig()
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "rf_standard_features": {name: None for name in RF_FEATURE_NAMES},
        "resolution_features": {},
        "numerical_structure_features": {},
        "crossing_features": {
            name: None for name in EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES
        },
        "crossing_records": [],
        "extremum_features": {
            "match_found": None,
            **{name: None for name in EXPERIMENTAL_EXTREMUM_RF_FEATURE_NAMES},
        },
        "boundary_features": {
            "axis_artifact": empty_axis_artifact_features(config.r_ax),
        },
    }


def grouped_rule_features(
    named_features: Mapping[str, Any],
    feature_status: Mapping[str, Any],
    axis_artifact_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Organize shared RF31 measurements and deterministic rule evidence."""
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "rf_standard_features": {
            name: named_features[name] for name in RF_FEATURE_NAMES
        },
        "resolution_features": {},
        "numerical_structure_features": {},
        "crossing_features": {
            name: named_features[name]
            for name in EXPERIMENTAL_CROSSING_RF_FEATURE_NAMES
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
) -> RuleResult:
    """Extract named features and evaluate one valid, preprocessed TAE mode."""
    axis_config = axis_artifact_config or AxisArtifactConfig()
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
            features=empty_rule_features(axis_config),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    try:
        if mode is None or low2 is None or high2 is None:
            raise ValueError("mode, low2, and high2 arrays are required")
        axis_features = extract_axis_artifact_features(mode, r_ax=axis_config.r_ax)
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
        features = grouped_rule_features(
            named_features,
            feature_status,
            axis_features,
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
            features=empty_rule_features(axis_config),
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
