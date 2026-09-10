"""Dimensionless gate margins; decisions and equality semantics stay authoritative.

Use max for OR/across candidates and min for AND within one candidate.
Absent prerequisites have zero severity; missing/disabled evidence is null.
"""

from dataclasses import asdict
import hashlib
import json
import math

SCHEMA_VERSION = "rule-severity-v1"
GATE_NAMES = (
    "BAD_AXIS_SPIKE", "BAD_GRID_SCALE_SPIKE", "BAD_GRID_SCALE_PACKET",
    "BAD_NEAR_AXIS_GRID_OSCILLATION", "BAD_CONT_CROSS", "BAD_CONT_CROSS_WINDOW",
    "BAD_EDGE_SPIKE", "BAD_INTERIOR_UNRESOLVED_ENVELOPE",
    "BAD_INTERIOR_HARMONIC_INCOHERENCE", "BAD_CONTINUUM_CROSSING_TAIL",
    "BAD_AXIS_ENERGY_CONCENTRATION", "BAD_EXTENDED_CONTINUUM_NOISE",
)
SEVERITY_FIELDS = [*("gate_severity_" + gate for gate in GATE_NAMES),
                   "overall_rule_severity", "rule_margin", "nearest_gate",
                   "severity_complete", "severity_schema_version", "severity_config_sha256"]


def component(value, threshold, *, small_is_bad=False, comparison=">="):
    ratio = None
    if value is not None and threshold is not None and math.isfinite(value) and threshold > 0:
        ratio = threshold / max(value, 1e-12) if small_is_bad else value / threshold
        if not math.isfinite(ratio):
            ratio = None
    return dict(value=value, threshold=threshold, ratio=ratio, comparison=comparison)


def combined(components, operation="AND"):
    ratios = [c["ratio"] for c in components.values()]
    if not ratios or any(q is None for q in ratios):
        return None
    return float(min(ratios) if operation == "AND" else max(ratios))


def gate_record(enabled, fired, candidates=(), *, operation="AND", unavailable=None):
    candidates = list(candidates)
    scored = [(combined(parts, operation), parts, witness) for parts, witness in candidates]
    known = [r for r in scored if r[0] is not None]
    winner = max(known, key=lambda r: r[0]) if known else (None, {}, None)
    missing = unavailable or ("UNDEFINED_NORMALIZATION" if len(known) != len(scored) else None)
    severity = None if missing else (winner[0] if scored else 0.0)
    status = "MEASURED" if scored else "NOT_APPLICABLE"
    if missing:
        status = missing
    if not enabled:
        severity, status = None, "DISABLED"
    return dict(enabled=bool(enabled), fired=bool(fired), severity=severity,
                status=status, operation=operation, components=winner[1], witness=winner[2])


def empty_severity_features():
    return dict(schema_version=SCHEMA_VERSION, configuration_sha256=None,
                gates={g: dict(enabled=None, fired=None, severity=None,
                               status="UNAVAILABLE", operation=None, components={}, witness=None)
                       for g in GATE_NAMES},
                overall_rule_severity=None, rule_margin=None, nearest_gate=None,
                complete=False, unavailable_gates=list(GATE_NAMES))


def severity_columns(report):
    row = {"gate_severity_" + g: report["gates"][g]["severity"] for g in GATE_NAMES}
    row.update({k: report[k] for k in ("overall_rule_severity", "rule_margin", "nearest_gate")})
    row.update(severity_complete=report["complete"], severity_schema_version=report["schema_version"],
               severity_config_sha256=report["configuration_sha256"])
    return row


def extract_rule_severities(features, configs, candidates, fired):
    """Consume the same candidate geometry and gate flags as the rule engine."""
    gates = {}
    def add(name, rows=(), **kwargs):
        gates[name] = gate_record(configs[name].enabled, fired[name], rows, **kwargs)
    def q(value, threshold, **kwargs):
        return component(value, threshold, **kwargs)

    c = configs["BAD_AXIS_SPIKE"]
    add("BAD_AXIS_SPIKE", [({"amplitude": q(r["axis_peak"], c.axis_amplitude_min),
                            "width": q(r["axis_halfmax_width_grid"], c.axis_width_max_grid,
                                       small_is_bad=True, comparison="<=")}, r)
                           for r in candidates["axis"]])
    c = configs["BAD_GRID_SCALE_SPIKE"]
    add("BAD_GRID_SCALE_SPIKE", [({"amplitude": q(r["amplitude"], c.amplitude_min),
                                  "width": q(r["width_grid"], r["width_limit"],
                                             small_is_bad=True, comparison="<=")}, r)
                                 for r in candidates["grid"]])
    c = configs["BAD_GRID_SCALE_PACKET"]
    add("BAD_GRID_SCALE_PACKET", [({"amplitude": q(r["amplitude"], c.amplitude_min),
                                   "turn_step": q(r["turn_step"], c.step_min)}, r)
                                  for r in candidates["packet"]])
    c = configs["BAD_NEAR_AXIS_GRID_OSCILLATION"]
    f = features["numerical_structure_features"]["near_axis_grid_oscillation"]
    add("BAD_NEAR_AXIS_GRID_OSCILLATION", [] if not f["qualifying_run_count"] else [(
        {"axis_amplitude": q(f["near_axis_peak"], c.amplitude_min),
         "run_step_l2": q(f["selected_run_step_l2"], c.step_l2_min)},
        {k: f[k] for k in ("selected_run_harmonic_index", "selected_run_start_index",
                          "selected_run_end_index", "selected_run_consecutive_sign_flip_count")})])
    c = configs["BAD_CONT_CROSS"]
    add("BAD_CONT_CROSS", [({"energy_at_crossing": q(r["W_peak"], c.w_cross_threshold,
                                                     comparison=">")}, r)
                           for r in features["crossing_records"]])
    c = configs["BAD_CONT_CROSS_WINDOW"]
    f = features["crossing_features"]["continuum_crossing_window_exception"]
    rows = []
    for r in f["records"]:
        if r["exception_conditions_pass"]:
            continue
        parts = {}
        if c.amplitude_min is not None:
            parts["window_amplitude"] = q(r["window_A_max"], c.amplitude_min)
        if c.w_min is not None:
            parts["window_energy"] = q(r["window_W_max"], c.w_min)
        rows.append((parts, dict(boundary=r["boundary"], r_cross=r["r_cross"])))
    add("BAD_CONT_CROSS_WINDOW", rows, operation="OR")
    gates["BAD_CONT_CROSS_WINDOW"]["exempted_candidates"] = sum(
        bool(r["exception_conditions_pass"]) for r in f["records"])
    c = configs["BAD_EDGE_SPIKE"]
    f = features["boundary_features"]["edge_artifact"]
    add("BAD_EDGE_SPIKE", [] if not f["edge_energy_peak_in_window"] else [(
        {"width": q(f["edge_energy_halfmax_width_grid"], c.edge_width_max_grid,
                    small_is_bad=True, comparison="<=")}, {"peak_r": f["edge_energy_peak_r"]})])
    c = configs["BAD_INTERIOR_UNRESOLVED_ENVELOPE"]
    f = features["resolution_features"]["interior_unresolved_envelope"]
    applicable = (f["energy_peak_r"] is not None and f["energy_peak_r"] <= c.peak_r_max + 64 * math.ulp(1.)
                  and not candidates["interior_exception"])
    add("BAD_INTERIOR_UNRESOLVED_ENVELOPE", [] if not applicable else [(
        {"width": q(f["energy_halfmax_width_grid"], c.width_max_grid,
                    small_is_bad=True, comparison="<=")}, {"peak_r": f["energy_peak_r"]})])
    gates["BAD_INTERIOR_UNRESOLVED_ENVELOPE"]["exception_qualified"] = candidates["interior_exception"]
    c = configs["BAD_INTERIOR_HARMONIC_INCOHERENCE"]
    f = features["numerical_structure_features"]["interior_harmonic_incoherence"]
    add("BAD_INTERIOR_HARMONIC_INCOHERENCE", [] if not f["active_adjacent_harmonic_pair_count"] else [(
        {"incoherence_score": q(f["incoherence_score"], c.score_threshold, comparison=">")}, None)],
        unavailable="UNSUPPORTED_RESOLUTION" if not f["resolution_eligible"] else None)
    c = configs["BAD_CONTINUUM_CROSSING_TAIL"]
    f = features["crossing_features"]["continuum_crossing_tail"]
    add("BAD_CONTINUUM_CROSSING_TAIL", [(
        {"roughness": q(r["K_cross"], c.k_min, comparison=">"),
         "tail_top2_ratio": q(r["tail_over_top2"], c.top2_ratio_min, comparison=">")},
        dict(boundary=r["boundary"], r_cross=r["r_cross"])) for r in f["records"]],
        unavailable="UNSUPPORTED_RESOLUTION" if f["records"] and not f["resolution_eligible"] else None)
    c = configs["BAD_AXIS_ENERGY_CONCENTRATION"]
    f = features["boundary_features"]["axis_energy_concentration"]
    add("BAD_AXIS_ENERGY_CONCENTRATION", [(
        {"axis_amplitude": q(f["axis_amplitude"], c.amplitude_min, comparison=">"),
         "inner_energy_fraction": q(f["inner_energy_fraction"], c.energy_fraction_min, comparison=">")}, None)])
    c = configs["BAD_EXTENDED_CONTINUUM_NOISE"]
    f = features["numerical_structure_features"]["extended_continuum_noise"]
    add("BAD_EXTENDED_CONTINUUM_NOISE", [(
        {"top2_ratio": q(r["hf_out_top2_ratio"], c.top2_min),
         "local_fraction": q(r["hf_out_local_fraction"], c.local_min),
         "radial_length": q(r["hf_out_radial_length"], c.radial_length_min)},
        dict(region_id=r["region_id"], side=r["side"], r_start=r["r_start"], r_end=r["r_end"]))
        for r in f["records"] if r["hf_out_energy"] > 0])
    # Explicit zero-HF regions cannot satisfy positive production cuts.
    unknown = [g for g in GATE_NAMES if gates[g]["enabled"] and gates[g]["severity"] is None]
    known = [g for g in GATE_NAMES if gates[g]["enabled"] and gates[g]["severity"] is not None]
    nearest = max(known, key=lambda g: gates[g]["severity"]) if known else None
    overall = gates[nearest]["severity"] if nearest else 0.0
    encoded = json.dumps({g: asdict(configs[g]) for g in GATE_NAMES}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return dict(schema_version=SCHEMA_VERSION, configuration_sha256=hashlib.sha256(encoded).hexdigest(),
                gates=gates, complete=not unknown, unavailable_gates=unknown,
                overall_rule_severity=None if unknown else overall,
                rule_margin=None if unknown else 1.0 - overall,
                nearest_gate=nearest if not unknown and overall > 0 else None)
