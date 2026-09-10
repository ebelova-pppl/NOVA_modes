#!/usr/bin/env python3
"""Load strict, versioned run configurations for deterministic TAE rules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tae_rule_engine import (
    AXIS_ENERGY_RULESET_VERSION,
    CLEARANCE_RULESET_VERSION,
    LEGACY_RULESET_VERSION,
    PREVIOUS_RULESET_VERSION,
    RULESET_VERSION,
    AxisArtifactConfig,
    AxisEnergyConcentrationConfig,
    ContinuumCrossingConfig,
    ContinuumCrossingWindowConfig,
    ContinuumCrossingTailConfig,
    EdgeArtifactConfig,
    GridScalePacketConfig,
    GridScaleSpikeConfig,
    InteriorHarmonicIncoherenceConfig,
    InteriorUnresolvedEnvelopeConfig,
    NearAxisGridOscillationConfig,
)
from continuum_noise import ContinuumNoiseThresholds
from tae_rule_io import sha256_file
from tae_eae_features import validate_routing_thresholds


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "rules"
RULE_CONFIG_SCHEMA_VERSION = "tae-rule-run-config-v11"
PRODUCTION_RULE_CONFIG_NAME = "tae_rules_production_v11"
PRODUCTION_RULE_CONFIG_SHA256 = (
    "da5019505f7e7a8025215e78dd41b0ed93dbbb3e41a470a0a0cb22b771bac9da"
)
FROZEN_CONFIGURATION_SHA256 = {
    "tae_rules_production_v10": "51932bc9d402a99e6b015cbb72cca11e42a175f7edd46315246d6b965860843e",
    "tae_rules_production_v9": "e5d3ae4bac8cea9b9606a7e6337180e205f9294ea56529ec4a25f0a55d2f6bf7",
    "tae_rules_production_v8": "86436a3486cd2d3bd9fd8a16d3af51f2127cb0325017de392ae7d1d36d8647e0",
    "tae_rules_production_v7": "10980f26b800d597de343e7d1fde173d5b749c56b9b15c5d98f3e8ac03a16429",
    "tae_rules_production_v5": (
        "982cc0ba3f17aae03a9fc6a4b662104200df0ff2897bda4de21131ce71c5bc9f"
    ),
    "tae_rules_production_v6": "b611a7554e61e3a16311d4fcdb0ff4854953fce769f70b6267308bfa46c1e398",
    PRODUCTION_RULE_CONFIG_NAME: PRODUCTION_RULE_CONFIG_SHA256,
}


@dataclass(frozen=True)
class RuleRunConfiguration:
    """Validated named configuration plus exact runtime keyword arguments."""

    name: str
    schema_version: str
    rule_set_version: str
    source_path: Path
    sha256: str
    run_kwargs: Mapping[str, Any]


def _require_exact_keys(
    values: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    missing = sorted(expected - set(values))
    extra = sorted(set(values) - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise ValueError(f"{context} keys are invalid: {'; '.join(details)}")


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _string(values: Mapping[str, Any], key: str, *, context: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def _bool(values: Mapping[str, Any], key: str, *, context: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be true or false")
    return value


def _float(values: Mapping[str, Any], key: str, *, context: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}.{key} must be a finite number")
    return result


def _int(values: Mapping[str, Any], key: str, *, context: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def resolve_rule_config_path(value: str | Path) -> Path:
    """Resolve a preset name under configs/rules or an explicit file path."""
    raw = Path(value).expanduser()
    if raw.parent == Path(".") and not raw.suffix:
        raw = DEFAULT_CONFIG_DIR / f"{raw.name}.yaml"
    elif not raw.is_absolute():
        raw = Path.cwd() / raw
    path = raw.resolve()
    if not path.is_file():
        raise ValueError(f"rule configuration does not exist: {path}")
    return path


def load_rule_run_configuration(value: str | Path) -> RuleRunConfiguration:
    """Load strict JSON-compatible YAML without adding a YAML dependency."""
    path = resolve_rule_config_path(value)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"rule configuration must be valid JSON-compatible YAML: {path}"
        ) from exc
    document = _mapping(raw, context="configuration")
    _require_exact_keys(
        document,
        {"schema_version", "name", "rule_set_version", "routing", "deduplication", "gates"},
        context="configuration",
    )
    schema_version = _string(document, "schema_version", context="configuration")
    if schema_version not in {
        RULE_CONFIG_SCHEMA_VERSION,
        "tae-rule-run-config-v10",
        "tae-rule-run-config-v9",
        "tae-rule-run-config-v8",
        "tae-rule-run-config-v7",
        "tae-rule-run-config-v6",
        "tae-rule-run-config-v5",
    }:
        raise ValueError(
            f"unsupported rule configuration schema {schema_version!r}; "
            f"expected {RULE_CONFIG_SCHEMA_VERSION!r}, v10, v9, v8, v7, v6, or v5"
        )
    name = _string(document, "name", context="configuration")
    rule_set_version = _string(document, "rule_set_version", context="configuration")
    expected_ruleset = {
        RULE_CONFIG_SCHEMA_VERSION: RULESET_VERSION,
        "tae-rule-run-config-v10": RULESET_VERSION,
        "tae-rule-run-config-v9": AXIS_ENERGY_RULESET_VERSION,
        "tae-rule-run-config-v8": CLEARANCE_RULESET_VERSION,
        "tae-rule-run-config-v7": PREVIOUS_RULESET_VERSION,
        "tae-rule-run-config-v6": LEGACY_RULESET_VERSION,
        "tae-rule-run-config-v5": LEGACY_RULESET_VERSION,
    }[schema_version]
    if rule_set_version != expected_ruleset:
        raise ValueError(
            f"configuration {name!r} pins ruleset {rule_set_version!r}, but "
            f"this schema requires {expected_ruleset!r}"
        )

    routing = _mapping(document["routing"], context="routing")
    _require_exact_keys(
        routing,
        {
            "fraction_tae_threshold",
            "fraction_eae_threshold",
            "signed_delta_eae_threshold",
            "include_mixed_in_tae_like",
        } | (
            {"fraction_direct_eae_threshold"}
            if schema_version != "tae-rule-run-config-v5"
            else set()
        ),
        context="routing",
    )
    fraction_tae_threshold = _float(
        routing, "fraction_tae_threshold", context="routing"
    )
    fraction_eae_threshold = _float(
        routing, "fraction_eae_threshold", context="routing"
    )
    # Frozen v5 predates the unconditional branch; never inherit the new default.
    fraction_direct_eae_threshold = (
        _float(routing, "fraction_direct_eae_threshold", context="routing")
        if schema_version != "tae-rule-run-config-v5"
        else 0.0
    )
    signed_delta_eae_threshold = _float(
        routing, "signed_delta_eae_threshold", context="routing"
    )
    validate_routing_thresholds(
        fraction_tae_threshold=fraction_tae_threshold,
        fraction_eae_threshold=fraction_eae_threshold,
        fraction_direct_eae_threshold=fraction_direct_eae_threshold,
        signed_delta_eae_threshold=signed_delta_eae_threshold,
    )
    if not _bool(routing, "include_mixed_in_tae_like", context="routing"):
        raise ValueError("include_mixed_in_tae_like must remain true")

    deduplication = _mapping(document["deduplication"], context="deduplication")
    _require_exact_keys(
        deduplication, {"rel_freq_tol"} | ({"rank_method"} if schema_version == RULE_CONFIG_SCHEMA_VERSION else set()), context="deduplication"
    )
    rank_method = (_string(deduplication, "rank_method", context="deduplication")
                   if schema_version == RULE_CONFIG_SCHEMA_VERSION else "rf_p_good")
    if rank_method not in {"rule_severity", "rf_p_good"}:
        raise ValueError("deduplication.rank_method must be rule_severity or rf_p_good")
    rel_freq_tol = _float(deduplication, "rel_freq_tol", context="deduplication")
    if rel_freq_tol <= 0.0:
        raise ValueError("deduplication.rel_freq_tol must be positive")

    gates = _mapping(document["gates"], context="gates")
    gate_names = {
        "axis_artifact",
        "grid_scale_spike",
        "grid_scale_packet",
        "near_axis_grid_oscillation",
        "continuum_crossing",
        "continuum_crossing_window",
        "edge_artifact",
        "interior_unresolved_envelope",
        "interior_harmonic_incoherence",
        "continuum_crossing_tail",
    }
    if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10", "tae-rule-run-config-v9"}:
        gate_names.add("axis_energy_concentration")
    if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10"}:
        gate_names.add("extended_continuum_noise")
    _require_exact_keys(gates, gate_names, context="gates")
    # Older frozen configurations never inherit the newly enabled gate.
    axis_energy_config = AxisEnergyConcentrationConfig(
        amplitude_min=None, energy_fraction_min=None
    )
    if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10", "tae-rule-run-config-v9"}:
        context = "gates.axis_energy_concentration"
        axis_energy = _mapping(gates["axis_energy_concentration"], context=context)
        _require_exact_keys(axis_energy, {
            "enabled", "amplitude_r_max", "amplitude_min", "energy_r_max", "energy_fraction_min"
        }, context=context)
        axis_energy_enabled = _bool(axis_energy, "enabled", context=context)
        axis_energy_values = {
            name: _float(axis_energy, name, context=context)
            for name in ("amplitude_r_max", "amplitude_min", "energy_r_max", "energy_fraction_min")
        }
        AxisEnergyConcentrationConfig(**axis_energy_values)
        if not axis_energy_enabled:
            axis_energy_values.update(amplitude_min=None, energy_fraction_min=None)
        axis_energy_config = AxisEnergyConcentrationConfig(**axis_energy_values)

    noise_config = ContinuumNoiseThresholds(top2_min=None)
    if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10"}:
        context = "gates.extended_continuum_noise"
        noise = _mapping(gates["extended_continuum_noise"], context=context)
        _require_exact_keys(noise, {"enabled", "top2_min", "local_min", "radial_length_min"}, context=context)
        noise_values = {key: _float(noise, key, context=context)
                        for key in ("top2_min", "local_min", "radial_length_min")}
        ContinuumNoiseThresholds(**noise_values)
        if not _bool(noise, "enabled", context=context):
            noise_values["top2_min"] = None
        noise_config = ContinuumNoiseThresholds(**noise_values)

    axis = _mapping(gates["axis_artifact"], context="gates.axis_artifact")
    _require_exact_keys(
        axis,
        {"enabled", "r_ax", "amplitude_min", "width_max_grid"},
        context="gates.axis_artifact",
    )
    axis_enabled = _bool(axis, "enabled", context="gates.axis_artifact")
    axis_r_ax = _float(axis, "r_ax", context="gates.axis_artifact")
    axis_amplitude = _float(axis, "amplitude_min", context="gates.axis_artifact")
    axis_width = _float(axis, "width_max_grid", context="gates.axis_artifact")
    AxisArtifactConfig(
        r_ax=axis_r_ax,
        axis_amplitude_min=axis_amplitude,
        axis_width_max_grid=axis_width,
    )

    grid = _mapping(gates["grid_scale_spike"], context="gates.grid_scale_spike")
    _require_exact_keys(
        grid,
        {
            "enabled",
            "amplitude_min",
            "width_max_grid",
            "high_r_cutoff_r",
            "high_r_width_max_grid",
        },
        context="gates.grid_scale_spike",
    )
    grid_enabled = _bool(grid, "enabled", context="gates.grid_scale_spike")
    grid_amplitude = _float(grid, "amplitude_min", context="gates.grid_scale_spike")
    grid_width = _float(grid, "width_max_grid", context="gates.grid_scale_spike")
    grid_cutoff = _float(grid, "high_r_cutoff_r", context="gates.grid_scale_spike")
    grid_high_width = _float(
        grid, "high_r_width_max_grid", context="gates.grid_scale_spike"
    )
    GridScaleSpikeConfig(
        amplitude_min=grid_amplitude,
        width_max_grid=grid_width,
        high_r_cutoff_r=grid_cutoff,
        high_r_width_max_grid=grid_high_width,
    )

    packet = _mapping(
        gates["grid_scale_packet"], context="gates.grid_scale_packet"
    )
    _require_exact_keys(
        packet,
        {
            "enabled",
            "amplitude_min",
            "step_min",
            "min_large_turns",
            "window_span_grid",
            "peak_r_max",
        },
        context="gates.grid_scale_packet",
    )
    packet_enabled = _bool(packet, "enabled", context="gates.grid_scale_packet")
    packet_amplitude = _float(
        packet, "amplitude_min", context="gates.grid_scale_packet"
    )
    packet_step = _float(packet, "step_min", context="gates.grid_scale_packet")
    packet_turns = _int(
        packet, "min_large_turns", context="gates.grid_scale_packet"
    )
    packet_span = _int(
        packet, "window_span_grid", context="gates.grid_scale_packet"
    )
    packet_peak_r = _float(packet, "peak_r_max", context="gates.grid_scale_packet")
    GridScalePacketConfig(
        amplitude_min=packet_amplitude,
        step_min=packet_step,
        min_large_turns=packet_turns,
        window_span_grid=packet_span,
        peak_r_max=packet_peak_r,
    )

    near_axis_oscillation = _mapping(
        gates["near_axis_grid_oscillation"],
        context="gates.near_axis_grid_oscillation",
    )
    _require_exact_keys(
        near_axis_oscillation,
        {
            "enabled",
            "peak_r_max",
            "amplitude_min",
            "min_consecutive_sign_flips",
            "step_l2_min",
        },
        context="gates.near_axis_grid_oscillation",
    )
    near_axis_oscillation_enabled = _bool(
        near_axis_oscillation,
        "enabled",
        context="gates.near_axis_grid_oscillation",
    )
    near_axis_oscillation_peak_r = _float(
        near_axis_oscillation,
        "peak_r_max",
        context="gates.near_axis_grid_oscillation",
    )
    near_axis_oscillation_amplitude = _float(
        near_axis_oscillation,
        "amplitude_min",
        context="gates.near_axis_grid_oscillation",
    )
    near_axis_oscillation_min_flips = _int(
        near_axis_oscillation,
        "min_consecutive_sign_flips",
        context="gates.near_axis_grid_oscillation",
    )
    near_axis_oscillation_step_l2 = _float(
        near_axis_oscillation,
        "step_l2_min",
        context="gates.near_axis_grid_oscillation",
    )
    NearAxisGridOscillationConfig(
        peak_r_max=near_axis_oscillation_peak_r,
        amplitude_min=near_axis_oscillation_amplitude,
        min_consecutive_sign_flips=near_axis_oscillation_min_flips,
        step_l2_min=near_axis_oscillation_step_l2,
    )

    crossing = _mapping(
        gates["continuum_crossing"], context="gates.continuum_crossing"
    )
    _require_exact_keys(
        crossing,
        {"enabled", "w_cross_threshold"},
        context="gates.continuum_crossing",
    )
    crossing_enabled = _bool(
        crossing, "enabled", context="gates.continuum_crossing"
    )
    crossing_threshold = _float(
        crossing, "w_cross_threshold", context="gates.continuum_crossing"
    )
    ContinuumCrossingConfig(w_cross_threshold=crossing_threshold)

    window = _mapping(
        gates["continuum_crossing_window"],
        context="gates.continuum_crossing_window",
    )
    _require_exact_keys(
        window,
        {"enabled", "half_width_grid", "amplitude_min", "w_min"}
        | (
            {"smooth_exception"}
            if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10", "tae-rule-run-config-v9", "tae-rule-run-config-v8", "tae-rule-run-config-v7"}
            else set()
        ),
        context="gates.continuum_crossing_window",
    )
    window_enabled = _bool(
        window, "enabled", context="gates.continuum_crossing_window"
    )
    window_half_width = _int(
        window, "half_width_grid", context="gates.continuum_crossing_window"
    )
    window_amplitude = _float(
        window, "amplitude_min", context="gates.continuum_crossing_window"
    )
    window_w = _float(window, "w_min", context="gates.continuum_crossing_window")
    exception_kwargs = {"exception_amplitude_max": None, "exception_k_max": None}
    if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10", "tae-rule-run-config-v9", "tae-rule-run-config-v8", "tae-rule-run-config-v7"}:
        context = "gates.continuum_crossing_window.smooth_exception"
        exception = _mapping(window["smooth_exception"], context=context)
        _require_exact_keys(
            exception,
            {
                "enabled",
                "amplitude_max",
                "k_max",
                "half_width_grid",
                "calibrated_n_radial",
            },
            context=context,
        )
        enabled = _bool(exception, "enabled", context=context)
        exception_kwargs = {
            "exception_amplitude_max": _float(
                exception, "amplitude_max", context=context
            ),
            "exception_k_max": _float(exception, "k_max", context=context),
            "exception_half_width_grid": _int(
                exception, "half_width_grid", context=context
            ),
            "exception_calibrated_n_radial": _int(
                exception, "calibrated_n_radial", context=context
            ),
        }
        # Validate stored thresholds even when the named exception is disabled.
        ContinuumCrossingWindowConfig(**exception_kwargs)
        if not enabled:
            exception_kwargs["exception_amplitude_max"] = None
    window_config = ContinuumCrossingWindowConfig(
        half_width_grid=window_half_width,
        amplitude_min=window_amplitude,
        w_min=window_w,
        **exception_kwargs,
    )

    edge = _mapping(gates["edge_artifact"], context="gates.edge_artifact")
    _require_exact_keys(
        edge,
        {"enabled", "r_edge_min", "width_max_grid"},
        context="gates.edge_artifact",
    )
    edge_enabled = _bool(edge, "enabled", context="gates.edge_artifact")
    edge_r = _float(edge, "r_edge_min", context="gates.edge_artifact")
    edge_width = _float(edge, "width_max_grid", context="gates.edge_artifact")
    EdgeArtifactConfig(
        r_edge_min=edge_r,
        edge_width_max_grid=edge_width,
    )

    interior = _mapping(
        gates["interior_unresolved_envelope"],
        context="gates.interior_unresolved_envelope",
    )
    _require_exact_keys(
        interior,
        {
            "enabled",
            "peak_r_max",
            "width_max_grid",
            "extremum_r_min",
            "extremum_r_max",
            "ext_dr_max",
            "ext_df_gap_min",
            "ext_df_gap_max",
        } | ({"ext_df_gap_min_inclusive"} if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10", "tae-rule-run-config-v9", "tae-rule-run-config-v8"} else set()),
        context="gates.interior_unresolved_envelope",
    )
    interior_enabled = _bool(
        interior, "enabled", context="gates.interior_unresolved_envelope"
    )
    interior_peak_r = _float(
        interior, "peak_r_max", context="gates.interior_unresolved_envelope"
    )
    interior_width = _float(
        interior, "width_max_grid", context="gates.interior_unresolved_envelope"
    )
    interior_extremum_r_min = _float(
        interior, "extremum_r_min", context="gates.interior_unresolved_envelope"
    )
    interior_extremum_r_max = _float(
        interior, "extremum_r_max", context="gates.interior_unresolved_envelope"
    )
    interior_ext_dr = _float(
        interior, "ext_dr_max", context="gates.interior_unresolved_envelope"
    )
    interior_ext_df_min = _float(
        interior, "ext_df_gap_min", context="gates.interior_unresolved_envelope"
    )
    interior_ext_df_max = _float(
        interior, "ext_df_gap_max", context="gates.interior_unresolved_envelope"
    )
    # Frozen v5-v7 use the inclusive lower comparison, including tangency.
    interior_ext_df_min_inclusive = (
        _bool(interior, "ext_df_gap_min_inclusive", context="gates.interior_unresolved_envelope")
        if schema_version in {RULE_CONFIG_SCHEMA_VERSION, "tae-rule-run-config-v10", "tae-rule-run-config-v9", "tae-rule-run-config-v8"} else True
    )
    InteriorUnresolvedEnvelopeConfig(
        peak_r_max=interior_peak_r,
        width_max_grid=interior_width,
        extremum_r_min=interior_extremum_r_min,
        extremum_r_max=interior_extremum_r_max,
        ext_dr_max=interior_ext_dr,
        ext_df_gap_min=interior_ext_df_min,
        ext_df_gap_max=interior_ext_df_max,
        ext_df_gap_min_inclusive=interior_ext_df_min_inclusive,
    )

    incoherence = _mapping(
        gates["interior_harmonic_incoherence"],
        context="gates.interior_harmonic_incoherence",
    )
    _require_exact_keys(
        incoherence,
        {
            "enabled",
            "core_r_max",
            "active_core_energy_fraction_min",
            "max_lag_grid",
            "score_threshold",
            "calibrated_n_radial",
        },
        context="gates.interior_harmonic_incoherence",
    )
    incoherence_enabled = _bool(
        incoherence,
        "enabled",
        context="gates.interior_harmonic_incoherence",
    )
    incoherence_core_r_max = _float(
        incoherence,
        "core_r_max",
        context="gates.interior_harmonic_incoherence",
    )
    incoherence_active_fraction = _float(
        incoherence,
        "active_core_energy_fraction_min",
        context="gates.interior_harmonic_incoherence",
    )
    incoherence_max_lag = _int(
        incoherence,
        "max_lag_grid",
        context="gates.interior_harmonic_incoherence",
    )
    incoherence_score_threshold = _float(
        incoherence,
        "score_threshold",
        context="gates.interior_harmonic_incoherence",
    )
    incoherence_calibrated_n_radial = _int(
        incoherence,
        "calibrated_n_radial",
        context="gates.interior_harmonic_incoherence",
    )
    InteriorHarmonicIncoherenceConfig(
        core_r_max=incoherence_core_r_max,
        active_core_energy_fraction_min=incoherence_active_fraction,
        max_lag_grid=incoherence_max_lag,
        score_threshold=incoherence_score_threshold,
        calibrated_n_radial=incoherence_calibrated_n_radial,
    )

    tail_context = "gates.continuum_crossing_tail"
    tail = _mapping(gates["continuum_crossing_tail"], context=tail_context)
    _require_exact_keys(
        tail,
        {
            "enabled",
            "k_min",
            "top2_ratio_min",
            "half_width_grid",
            "calibrated_n_radial",
        },
        context=tail_context,
    )
    tail_enabled = _bool(tail, "enabled", context=tail_context)
    tail_config = ContinuumCrossingTailConfig(
        k_min=_float(tail, "k_min", context=tail_context),
        top2_ratio_min=_float(tail, "top2_ratio_min", context=tail_context),
        half_width_grid=_int(tail, "half_width_grid", context=tail_context),
        calibrated_n_radial=_int(tail, "calibrated_n_radial", context=tail_context),
    )

    run_kwargs = {
        "duplicate_rank_method": rank_method,
        "continuum_noise_top2_min": noise_config.top2_min,
        "continuum_noise_local_min": noise_config.local_min,
        "continuum_noise_radial_length_min": noise_config.radial_length_min,
        "continuum_crossing_tail_k_min": tail_config.k_min if tail_enabled else None,
        "continuum_crossing_tail_top2_ratio_min": tail_config.top2_ratio_min,
        "continuum_crossing_tail_half_width_grid": tail_config.half_width_grid,
        "continuum_crossing_tail_calibrated_n_radial": tail_config.calibrated_n_radial,
        "fraction_tae_threshold": fraction_tae_threshold,
        "fraction_eae_threshold": fraction_eae_threshold,
        "fraction_direct_eae_threshold": fraction_direct_eae_threshold,
        "signed_delta_eae_threshold": signed_delta_eae_threshold,
        "rel_freq_tol": rel_freq_tol,
        "axis_energy_amplitude_r_max": axis_energy_config.amplitude_r_max,
        "axis_energy_amplitude_min": axis_energy_config.amplitude_min,
        "axis_energy_r_max": axis_energy_config.energy_r_max,
        "axis_energy_fraction_min": axis_energy_config.energy_fraction_min,
        "axis_r_ax": axis_r_ax,
        "axis_amplitude_min": axis_amplitude if axis_enabled else None,
        "axis_width_max_grid": axis_width if axis_enabled else None,
        "grid_scale_amplitude_min": grid_amplitude if grid_enabled else None,
        "grid_scale_width_max_grid": grid_width,
        "grid_scale_high_r_cutoff_r": grid_cutoff,
        "grid_scale_high_r_width_max_grid": grid_high_width,
        "grid_scale_packet_amplitude_min": (
            packet_amplitude if packet_enabled else None
        ),
        "grid_scale_packet_step_min": packet_step,
        "grid_scale_packet_min_large_turns": packet_turns,
        "grid_scale_packet_window_span_grid": packet_span,
        "grid_scale_packet_peak_r_max": packet_peak_r,
        "near_axis_grid_oscillation_peak_r_max": (
            near_axis_oscillation_peak_r
        ),
        "near_axis_grid_oscillation_amplitude_min": (
            near_axis_oscillation_amplitude
            if near_axis_oscillation_enabled
            else None
        ),
        "near_axis_grid_oscillation_min_consecutive_sign_flips": (
            near_axis_oscillation_min_flips
        ),
        "near_axis_grid_oscillation_step_l2_min": (
            near_axis_oscillation_step_l2
        ),
        "w_cross_threshold": crossing_threshold if crossing_enabled else None,
        "cross_window_half_width_grid": window_half_width,
        "cross_window_amplitude_min": (
            window_amplitude if window_enabled else None
        ),
        "cross_window_w_min": window_w if window_enabled else None,
        "cross_window_exception_amplitude_max": window_config.exception_amplitude_max,
        "cross_window_exception_k_max": window_config.exception_k_max,
        "cross_window_exception_half_width_grid": window_config.exception_half_width_grid,
        "cross_window_exception_calibrated_n_radial": window_config.exception_calibrated_n_radial,
        "edge_r_min": edge_r,
        "edge_width_max_grid": edge_width if edge_enabled else None,
        "interior_envelope_peak_r_max": interior_peak_r,
        "interior_envelope_width_max_grid": (
            interior_width if interior_enabled else None
        ),
        "interior_envelope_extremum_r_min": interior_extremum_r_min,
        "interior_envelope_extremum_r_max": interior_extremum_r_max,
        "interior_envelope_ext_dr_max": interior_ext_dr,
        "interior_envelope_ext_df_gap_min": interior_ext_df_min,
        "interior_envelope_ext_df_gap_max": interior_ext_df_max,
        "interior_envelope_ext_df_gap_min_inclusive": interior_ext_df_min_inclusive,
        "interior_harmonic_core_r_max": incoherence_core_r_max,
        "interior_harmonic_active_core_energy_fraction_min": (
            incoherence_active_fraction
        ),
        "interior_harmonic_max_lag_grid": incoherence_max_lag,
        "interior_harmonic_incoherence_score_threshold": (
            incoherence_score_threshold if incoherence_enabled else None
        ),
        "interior_harmonic_calibrated_n_radial": (
            incoherence_calibrated_n_radial
        ),
    }
    digest = sha256_file(path)
    expected_digest = FROZEN_CONFIGURATION_SHA256.get(name)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            f"frozen configuration {name!r} has SHA-256 {digest}, expected "
            f"{expected_digest}"
        )
    return RuleRunConfiguration(
        name=name,
        schema_version=schema_version,
        rule_set_version=rule_set_version,
        source_path=path,
        sha256=digest,
        run_kwargs=run_kwargs,
    )
