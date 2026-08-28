#!/usr/bin/env python3
"""Load strict, versioned run configurations for deterministic TAE rules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tae_rule_engine import (
    RULESET_VERSION,
    AxisArtifactConfig,
    ContinuumCrossingConfig,
    ContinuumCrossingWindowConfig,
    EdgeArtifactConfig,
    GridScalePacketConfig,
    GridScaleSpikeConfig,
)
from tae_rule_io import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "rules"
RULE_CONFIG_SCHEMA_VERSION = "tae-rule-run-config-v1"
PRODUCTION_RULE_CONFIG_NAME = "tae_rules_production_v1"
PRODUCTION_RULE_CONFIG_SHA256 = (
    "a2c85d958eeebe4396a9ce0d2f52c3dbf157f1630d344279801c00bb826e6f39"
)


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
    if schema_version != RULE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported rule configuration schema {schema_version!r}; "
            f"expected {RULE_CONFIG_SCHEMA_VERSION!r}"
        )
    name = _string(document, "name", context="configuration")
    rule_set_version = _string(document, "rule_set_version", context="configuration")
    if rule_set_version != RULESET_VERSION:
        raise ValueError(
            f"configuration {name!r} pins ruleset {rule_set_version!r}, but "
            f"this checkout provides {RULESET_VERSION!r}"
        )

    routing = _mapping(document["routing"], context="routing")
    _require_exact_keys(
        routing,
        {
            "fraction_tae_threshold",
            "fraction_eae_threshold",
            "signed_delta_eae_threshold",
            "include_mixed_in_tae_like",
        },
        context="routing",
    )
    fraction_tae_threshold = _float(
        routing, "fraction_tae_threshold", context="routing"
    )
    fraction_eae_threshold = _float(
        routing, "fraction_eae_threshold", context="routing"
    )
    signed_delta_eae_threshold = _float(
        routing, "signed_delta_eae_threshold", context="routing"
    )
    if not 0.0 <= fraction_eae_threshold <= fraction_tae_threshold <= 1.0:
        raise ValueError(
            "routing thresholds must satisfy 0 <= fraction_eae_threshold "
            "<= fraction_tae_threshold <= 1"
        )
    if not _bool(routing, "include_mixed_in_tae_like", context="routing"):
        raise ValueError("include_mixed_in_tae_like must remain true")

    deduplication = _mapping(document["deduplication"], context="deduplication")
    _require_exact_keys(
        deduplication, {"rel_freq_tol"}, context="deduplication"
    )
    rel_freq_tol = _float(deduplication, "rel_freq_tol", context="deduplication")
    if rel_freq_tol <= 0.0:
        raise ValueError("deduplication.rel_freq_tol must be positive")

    gates = _mapping(document["gates"], context="gates")
    gate_names = {
        "axis_artifact",
        "grid_scale_spike",
        "grid_scale_packet",
        "continuum_crossing",
        "continuum_crossing_window",
        "edge_artifact",
    }
    _require_exact_keys(gates, gate_names, context="gates")

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
        {"enabled", "half_width_grid", "amplitude_min", "w_min"},
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
    ContinuumCrossingWindowConfig(
        half_width_grid=window_half_width,
        amplitude_min=window_amplitude,
        w_min=window_w,
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

    run_kwargs = {
        "fraction_tae_threshold": fraction_tae_threshold,
        "fraction_eae_threshold": fraction_eae_threshold,
        "signed_delta_eae_threshold": signed_delta_eae_threshold,
        "rel_freq_tol": rel_freq_tol,
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
        "w_cross_threshold": crossing_threshold if crossing_enabled else None,
        "cross_window_half_width_grid": window_half_width,
        "cross_window_amplitude_min": (
            window_amplitude if window_enabled else None
        ),
        "cross_window_w_min": window_w if window_enabled else None,
        "edge_r_min": edge_r,
        "edge_width_max_grid": edge_width if edge_enabled else None,
    }
    digest = sha256_file(path)
    if (
        name == PRODUCTION_RULE_CONFIG_NAME
        and digest != PRODUCTION_RULE_CONFIG_SHA256
    ):
        raise ValueError(
            f"frozen configuration {name!r} has SHA-256 {digest}, expected "
            f"{PRODUCTION_RULE_CONFIG_SHA256}"
        )
    return RuleRunConfiguration(
        name=name,
        schema_version=schema_version,
        rule_set_version=rule_set_version,
        source_path=path,
        sha256=digest,
        run_kwargs=run_kwargs,
    )
