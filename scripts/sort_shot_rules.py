#!/usr/bin/env python3
"""Deterministically preprocess and rule-sort one NOVA shot.

Example:
    python scripts/sort_shot_rules.py --shot_dir /path/to/SHOT \
        --out_dir /path/to/output \
        --manual_overrides /path/to/manual_overrides.csv \
        --rf_model /path/to/nova_mode_classifier.joblib

RF is optional and is used only to rank representatives among final-GOOD
close-frequency modes. This workflow never loads or runs a CNN model.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from _repo_bootstrap import ensure_repo_src_on_path


ensure_repo_src_on_path()

from make_tae_like_list import (  # noqa: E402
    DEFAULT_FRACTION_EAE_THRESHOLD,
    DEFAULT_FRACTION_TAE_THRESHOLD,
    DEFAULT_SIGNED_DELTA_EAE_THRESHOLD,
    PreprocessResult,
    preprocess_shot,
)
from tae_rule_engine import (  # noqa: E402
    DEFAULT_AXIS_AMPLITUDE_MIN,
    DEFAULT_AXIS_R_AX,
    DEFAULT_AXIS_WIDTH_MAX_GRID,
    DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN,
    DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID,
    DEFAULT_CROSS_WINDOW_W_MIN,
    DEFAULT_EDGE_R_MIN,
    DEFAULT_EDGE_WIDTH_MAX_GRID,
    DEFAULT_GRID_SCALE_AMPLITUDE_MIN,
    DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R,
    DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID,
    DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN,
    DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS,
    DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX,
    DEFAULT_GRID_SCALE_PACKET_STEP_MIN,
    DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID,
    DEFAULT_GRID_SCALE_WIDTH_MAX_GRID,
    DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX,
    DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN,
    DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX,
    DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN,
    DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX,
    DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX,
    DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID,
    DEFAULT_W_CROSS_THRESHOLD,
    NO_GOOD_TEMPLATE,
    RULESET_VERSION,
    AxisArtifactConfig,
    ContinuumCrossingConfig,
    ContinuumCrossingWindowConfig,
    EdgeArtifactConfig,
    GridScalePacketConfig,
    GridScaleSpikeConfig,
    InteriorUnresolvedEnvelopeConfig,
    evaluate_mode,
)
from tae_rule_config import (  # noqa: E402
    PRODUCTION_RULE_CONFIG_NAME,
    load_rule_run_configuration,
)
from tae_rule_io import (  # noqa: E402
    ALLOWED_FINAL_DECISIONS,
    MANUAL_OVERRIDE_FIELDS,
    RULE_OUTPUT_FIELDS,
    read_dict_csv,
    rule_row_sort_key,
    sha256_file,
    stable_json,
    write_dict_csv,
    write_text,
    write_vertical_summary,
)


SIMILARITY_THRESHOLD = 0.90
RADIAL_LOCATION_TOLERANCE = 0.10
RADIAL_WIDTH_TOLERANCE = 0.05

RULE_SURVIVOR_POLICY_REVIEW = "review"
RULE_SURVIVOR_POLICY_ACCEPT = "accept-as-good-v1"
RULE_SURVIVOR_POLICIES = {
    RULE_SURVIVOR_POLICY_REVIEW,
    RULE_SURVIVOR_POLICY_ACCEPT,
}

CLUSTER_FIELDS = [
    "ntor",
    "cluster_index",
    "cluster_status",
    "diagnostic_message",
    "type_group",
    "path",
    "mode_key",
    "omega",
    "action",
    "retained_mode_key",
    "duplicate_rank_score",
    "duplicate_rank_source",
    "selection_reason",
]

SHOT_SUMMARY_FIELDS = [
    "shot",
    "method",
    "n_total_files",
    "n_invalid",
    "n_tae_like",
    "n_mixed",
    "n_eae_like",
    "n_rule_evaluated",
    "n_preliminary_bad",
    "n_preliminary_review",
    "n_preliminary_good",
    "n_preliminary_invalid",
    "rule_survivor_policy",
    "n_rule_survivors_accepted",
    "n_final_bad",
    "n_final_review",
    "n_final_good_before_clustering",
    "n_final_good",
    "n_good_removed_as_duplicates",
    "duplicate_processing_status",
    "n_manual_override_rows",
    "n_modes_manually_inspected",
    "n_overrides_applied",
    "n_stale_overrides",
    "n_ambiguous_overrides",
    "n_ineligible_overrides",
    "n_unmatched_overrides",
    "n_decisions_changed",
    "transition_counts_json",
    "manual_adjudication_performed",
    "manual_override_sha256",
    "primary_reason_counts_json",
    "rule_set_version",
    "rule_configuration_name",
    "rule_configuration_schema_version",
    "rule_configuration_sha256",
    "fraction_tae_threshold",
    "fraction_eae_threshold",
    "signed_delta_eae_threshold",
    "include_mixed_in_tae_like",
    "rel_freq_tol",
    "similarity_threshold",
    "radial_location_tolerance",
    "radial_width_tolerance",
    "axis_artifact_gate_enabled",
    "axis_artifact_r_ax",
    "axis_artifact_amplitude_min",
    "axis_artifact_width_max_grid",
    "grid_scale_spike_gate_enabled",
    "grid_scale_spike_amplitude_min",
    "grid_scale_spike_width_max_grid",
    "grid_scale_spike_high_r_cutoff_r",
    "grid_scale_spike_high_r_width_max_grid",
    "grid_scale_packet_gate_enabled",
    "grid_scale_packet_amplitude_min",
    "grid_scale_packet_step_min",
    "grid_scale_packet_min_large_turns",
    "grid_scale_packet_window_span_grid",
    "grid_scale_packet_peak_r_max",
    "continuum_crossing_gate_enabled",
    "continuum_crossing_w_threshold",
    "continuum_crossing_window_gate_enabled",
    "continuum_crossing_window_half_width_grid",
    "continuum_crossing_window_amplitude_min",
    "continuum_crossing_window_w_min",
    "edge_artifact_gate_enabled",
    "edge_artifact_r_min",
    "edge_artifact_width_max_grid",
    "interior_envelope_gate_enabled",
    "interior_envelope_peak_r_max",
    "interior_envelope_width_max_grid",
    "interior_envelope_extremum_r_min",
    "interior_envelope_extremum_r_max",
    "interior_envelope_ext_dr_max",
    "interior_envelope_ext_df_gap_min",
    "interior_envelope_ext_df_gap_max",
]

SUMMARY_BY_N_FIELDS = ["shot", "n", *[field for field in SHOT_SUMMARY_FIELDS if field != "shot"]]

RULE_CONFIG_OVERRIDE_OPTIONS = frozenset(
    {
        "--rel_freq_tol",
        "--fraction_tae_threshold",
        "--fraction_eae_threshold",
        "--signed_delta_eae_threshold",
        "--axis_r_ax",
        "--axis_amplitude_min",
        "--axis_width_max_grid",
        "--disable_axis_artifact",
        "--grid_scale_amplitude_min",
        "--grid_scale_width_max_grid",
        "--grid_scale_high_r_cutoff_r",
        "--grid_scale_high_r_width_max_grid",
        "--disable_grid_scale_spike",
        "--grid_scale_packet_amplitude_min",
        "--grid_scale_packet_step_min",
        "--grid_scale_packet_min_large_turns",
        "--grid_scale_packet_window_span_grid",
        "--grid_scale_packet_peak_r_max",
        "--disable_grid_scale_packet",
        "--w_cross_threshold",
        "--disable_cont_cross",
        "--cross_window_half_width_grid",
        "--cross_window_amplitude_min",
        "--cross_window_w_min",
        "--disable_cont_cross_window",
        "--edge_r_min",
        "--edge_width_max_grid",
        "--disable_edge_artifact",
        "--interior_envelope_peak_r_max",
        "--interior_envelope_width_max_grid",
        "--interior_envelope_extremum_r_min",
        "--interior_envelope_extremum_r_max",
        "--interior_envelope_ext_dr_max",
        "--interior_envelope_ext_df_gap_min",
        "--interior_envelope_ext_df_gap_max",
        "--disable_interior_unresolved_envelope",
    }
)


@dataclass(frozen=True)
class OverrideAudit:
    supplied_rows: int = 0
    inspected_rows: int = 0
    applied: int = 0
    stale: int = 0
    ambiguous: int = 0
    ineligible: int = 0
    unmatched: int = 0
    decisions_changed: int = 0


@dataclass(frozen=True)
class DuplicateResult:
    selected_paths: frozenset[str]
    cluster_records: tuple[dict[str, Any], ...]
    status: str


@dataclass(frozen=True)
class ShotRunResult:
    preprocess: PreprocessResult
    preliminary_rows: tuple[dict[str, Any], ...]
    final_rows: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]
    duplicate_result: DuplicateResult


def apply_rule_survivor_policy(
    rows: Sequence[Mapping[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    """Apply a workflow decision to modes that passed every BAD gate.

    The underlying rule verdict remains REVIEW/NO_GOOD_TEMPLATE. Production
    sorting may accept those survivors as final GOOD without presenting that
    workflow policy as a positive rule-engine decision.
    """
    if policy not in RULE_SURVIVOR_POLICIES:
        allowed = ", ".join(sorted(RULE_SURVIVOR_POLICIES))
        raise ValueError(f"rule_survivor_policy must be one of: {allowed}")

    output: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        if row.get("rule_decision") in ALLOWED_FINAL_DECISIONS:
            row["rule_survivor_policy"] = policy
            row["rule_survivor_accepted"] = False
        if (
            policy == RULE_SURVIVOR_POLICY_ACCEPT
            and row.get("processing_status") == "RULE_EVALUATED"
            and row.get("rule_decision") == "REVIEW"
            and row.get("rule_primary_reason") == NO_GOOD_TEMPLATE
        ):
            row["rule_survivor_accepted"] = True
            row["final_decision"] = "GOOD"
            row["decision_source"] = "rule_survivor_policy"
        output.append(row)
    return output


def normalize_manual_decision(value: str) -> str:
    """Normalize a nonempty manual decision to GOOD, BAD, or REVIEW."""
    normalized = value.strip().upper()
    aliases = {"G": "GOOD", "B": "BAD", "R": "REVIEW"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALLOWED_FINAL_DECISIONS:
        raise ValueError(
            f"manual_decision must be GOOD, BAD, or REVIEW, got {value!r}"
        )
    return normalized


def load_manual_overrides(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the canonical reusable manual-override schema."""
    fields, rows = read_dict_csv(path)
    if fields != MANUAL_OVERRIDE_FIELDS:
        raise ValueError(
            "manual override columns must exactly match the stable schema in this order: "
            + ", ".join(MANUAL_OVERRIDE_FIELDS)
        )

    validated: list[dict[str, str]] = []
    for line_number, raw_row in enumerate(rows, start=2):
        row = dict(raw_row)
        decision_text = row["manual_decision"].strip()
        if not decision_text:
            if row["manual_reason"].strip():
                raise ValueError(
                    f"{path}:{line_number}: manual_reason requires manual_decision"
                )
            validated.append(row)
            continue
        row["manual_decision"] = normalize_manual_decision(decision_text)
        if not row["manual_reason"].strip():
            raise ValueError(
                f"{path}:{line_number}: manual_reason must be nonempty when "
                "manual_decision is present"
            )
        if not row["mode_key"].strip():
            raise ValueError(f"{path}:{line_number}: mode_key must not be blank")
        fingerprint = row["input_fingerprint"].strip().lower()
        if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
            raise ValueError(
                f"{path}:{line_number}: input_fingerprint must be a SHA-256 hex digest"
            )
        row["input_fingerprint"] = fingerprint
        if not row["reviewer"].strip():
            raise ValueError(
                f"{path}:{line_number}: reviewer must not be blank for a manual decision"
            )
        if not row["adjudication_timestamp"].strip():
            raise ValueError(
                f"{path}:{line_number}: adjudication_timestamp must not be blank "
                "for a manual decision"
            )
        validated.append(row)
    return validated


def apply_manual_overrides(
    rows: Sequence[Mapping[str, Any]],
    overrides: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], OverrideAudit]:
    """Apply only unique, eligible overrides whose input fingerprint matches."""
    overrides_by_key: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    inspected_rows = 0
    for override in overrides:
        if override.get("manual_decision", "").strip():
            normalize_manual_decision(str(override["manual_decision"]))
            if not str(override.get("manual_reason", "")).strip():
                raise ValueError(
                    "manual_reason must be nonempty when manual_decision is present"
                )
            inspected_rows += 1
            overrides_by_key[str(override.get("mode_key", "")).strip()].append(override)

    current_key_counts = Counter(str(row.get("mode_key", "")) for row in rows)
    current_keys = {key for key in current_key_counts if key}
    unmatched = sum(
        len(candidates)
        for key, candidates in overrides_by_key.items()
        if key not in current_keys
    )

    applied = stale = ambiguous = ineligible = decisions_changed = 0
    counted_ambiguous_keys: set[str] = set()
    output: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        key = str(row.get("mode_key", ""))
        candidates = overrides_by_key.get(key, [])
        rule_decision = str(row.get("rule_decision", ""))

        if not candidates:
            if rule_decision in ALLOWED_FINAL_DECISIONS:
                row["override_status"] = "NOT_PRESENT"
            output.append(row)
            continue
        if current_key_counts[key] != 1 or len(candidates) != 1:
            row["override_status"] = "AMBIGUOUS_OVERRIDE"
            row["override_message"] = (
                "override not applied because mode_key is not unique in the "
                "current rows or override file"
            )
            if row.get("rule_survivor_accepted") is True:
                row["rule_survivor_accepted"] = False
                row["final_decision"] = "REVIEW"
                row["decision_source"] = "override_review_required"
            if key not in counted_ambiguous_keys:
                ambiguous += len(candidates)
                counted_ambiguous_keys.add(key)
            output.append(row)
            continue
        override = candidates[0]
        if str(override.get("input_fingerprint", "")) != str(
            row.get("input_fingerprint", "")
        ):
            row["override_status"] = "STALE_FINGERPRINT"
            row["override_message"] = (
                "stored override fingerprint does not match current mode/datcon inputs; "
                "review this mode again"
            )
            if row.get("rule_survivor_accepted") is True:
                row["rule_survivor_accepted"] = False
                row["final_decision"] = "REVIEW"
                row["decision_source"] = "override_review_required"
            stale += 1
            output.append(row)
            continue
        if rule_decision not in ALLOWED_FINAL_DECISIONS:
            row["override_status"] = "INELIGIBLE_DECISION"
            row["override_message"] = (
                "manual overrides apply only to preliminary GOOD, BAD, or REVIEW rows"
            )
            ineligible += 1
            output.append(row)
            continue

        decision = normalize_manual_decision(str(override["manual_decision"]))
        row.update(
            {
                "manual_decision": decision,
                "manual_reason": str(override["manual_reason"]),
                "reviewer": str(override["reviewer"]),
                "adjudication_timestamp": str(override["adjudication_timestamp"]),
                "override_status": "APPLIED",
                "override_message": "",
                "final_decision": decision,
                "decision_source": "manual_override",
                "rule_survivor_accepted": False,
            }
        )
        applied += 1
        decisions_changed += int(decision != rule_decision)
        output.append(row)

    audit = OverrideAudit(
        supplied_rows=len(overrides),
        inspected_rows=inspected_rows,
        applied=applied,
        stale=stale,
        ambiguous=ambiguous,
        ineligible=ineligible,
        unmatched=unmatched,
        decisions_changed=decisions_changed,
    )
    return output, audit


def _fallback_cluster_record(
    cluster: Sequence[Mapping[str, Any]],
    *,
    ntor: int,
    cluster_index: int,
    status: str,
    diagnostic_message: str,
    selection_reason: str,
) -> dict[str, Any]:
    members = [
        {
            "path": str(row["path"]),
            "mode_key": str(row["mode_key"]),
            "omega": float(row["omega"]),
            "action": "KEEP",
            "type_group": index,
            "retained_mode_key": str(row["mode_key"]),
            "duplicate_rank_score": "",
            "duplicate_rank_source": "",
            "selection_reason": selection_reason,
        }
        for index, row in enumerate(cluster, start=1)
    ]
    return {
        "ntor": ntor,
        "cluster_index": cluster_index,
        "status": status,
        "diagnostic_message": diagnostic_message,
        "members": members,
    }


def _cluster_rows_by_frequency(
    rows: Sequence[dict[str, Any]], rel_freq_tol: float
) -> list[list[dict[str, Any]]]:
    """Match ``sort_shot.cluster_modes_by_frequency`` without model imports."""
    ordered = sorted(rows, key=lambda row: float(row["omega"]))
    if not ordered:
        return []
    clusters: list[list[dict[str, Any]]] = [[ordered[0]]]
    for row in ordered[1:]:
        anchor = clusters[-1][0]
        relative_delta = abs(float(row["omega"]) - float(anchor["omega"])) / max(
            abs(float(anchor["omega"])), 1e-12
        )
        if relative_delta < rel_freq_tol:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def deduplicate_final_good(
    rows: Sequence[dict[str, Any]],
    *,
    rf_model_path: str | Path | None,
    rel_freq_tol: float,
    rf_loader: Callable[[str | Path], Any] | None = None,
    rf_scorer: Callable[[Any, str], tuple[Any, ...]] | None = None,
) -> DuplicateResult:
    """Apply existing frequency/structure clustering only to final GOOD rows."""
    good_rows = [row for row in rows if row.get("final_decision") == "GOOD"]
    if not good_rows:
        return DuplicateResult(
            selected_paths=frozenset(),
            cluster_records=tuple(),
            status="SKIPPED_NO_GOOD_MODES",
        )

    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in good_rows:
        by_n[int(row["ntor"])].append(row)

    frequency_clusters: list[tuple[int, int, list[dict[str, Any]]]] = []
    for ntor in sorted(by_n):
        stubs = [
            {
                "path": row["path"],
                "mode_key": row["mode_key"],
                "ntor": int(row["ntor"]),
                "omega": float(row["omega"]),
                "row": row,
            }
            for row in sorted(by_n[ntor], key=rule_row_sort_key)
        ]
        for cluster_index, cluster in enumerate(
            _cluster_rows_by_frequency(stubs, rel_freq_tol), start=1
        ):
            frequency_clusters.append((ntor, cluster_index, list(cluster)))

    classifier = None
    unavailable_message = ""
    has_close_cluster = any(len(cluster) > 1 for _n, _index, cluster in frequency_clusters)
    if not has_close_cluster:
        unavailable_message = ""
    elif rf_model_path is None:
        unavailable_message = "no RF checkpoint was supplied"
    else:
        try:
            if rf_loader is None:
                import joblib

                rf_loader = joblib.load
            classifier = rf_loader(rf_model_path)
        except Exception as exc:
            unavailable_message = (
                f"RF checkpoint could not be loaded: {type(exc).__name__}: {exc}"
            )

    build_mode_dict = resolve_cluster = None
    if classifier is not None and has_close_cluster:
        # Reuse the established structure calculations and representative
        # selection only after a usable RF checkpoint is available.
        from sort_shot import build_mode_dict as existing_build_mode_dict
        from sort_shot import classify_mode_rf, resolve_cluster as existing_resolve_cluster

        build_mode_dict = existing_build_mode_dict
        resolve_cluster = existing_resolve_cluster
        if rf_scorer is None:
            rf_scorer = classify_mode_rf

    selected_paths = {str(row["path"]) for row in good_rows}
    records: list[dict[str, Any]] = []
    saw_close_cluster = False
    saw_scoring_failure = False

    for ntor, cluster_index, cluster in frequency_clusters:
        if len(cluster) == 1:
            records.append(
                _fallback_cluster_record(
                    cluster,
                    ntor=ntor,
                    cluster_index=cluster_index,
                    status="NOT_APPLICABLE_SINGLETON",
                    diagnostic_message="one final-GOOD mode in frequency cluster",
                    selection_reason="SINGLETON_RETAINED",
                )
            )
            continue

        saw_close_cluster = True
        if classifier is None:
            records.append(
                _fallback_cluster_record(
                    cluster,
                    ntor=ntor,
                    cluster_index=cluster_index,
                    status="SKIPPED_NO_RF_CHECKPOINT",
                    diagnostic_message=unavailable_message,
                    selection_reason="RETAIN_ALL_NO_RF_CHECKPOINT",
                )
            )
            continue

        scored_modes: list[dict[str, Any]] = []
        score_by_path: dict[str, float] = {}
        scoring_error = ""
        assert build_mode_dict is not None
        assert resolve_cluster is not None
        assert rf_scorer is not None
        for member in cluster:
            try:
                score_result = rf_scorer(classifier, str(member["path"]))
                score = float(score_result[0])
                mode = score_result[1]
                omega = float(score_result[2])
                ntor_scored = int(score_result[4])
                if not math.isfinite(score):
                    raise ValueError(f"non-finite RF p_good {score}")
                score_by_path[str(member["path"])] = score
                scored_modes.append(
                    build_mode_dict(
                        path=str(member["path"]),
                        shot=str(member["row"]["shot"]),
                        ntor=ntor_scored,
                        omega=omega,
                        score=score,
                        mode=mode,
                        dm_band=1,
                        center_power=2.0,
                        median_k=3,
                        max_step=2,
                    )
                )
            except Exception as exc:
                scoring_error = (
                    f"RF scoring failed for {member['mode_key']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

        if scoring_error:
            saw_scoring_failure = True
            records.append(
                _fallback_cluster_record(
                    cluster,
                    ntor=ntor,
                    cluster_index=cluster_index,
                    status="SKIPPED_RF_SCORING_FAILED",
                    diagnostic_message=scoring_error,
                    selection_reason="RETAIN_ALL_RF_SCORING_FAILED",
                )
            )
            continue

        kept, type_groups = resolve_cluster(
            scored_modes,
            rel_freq_tol=rel_freq_tol,
            sim_threshold=SIMILARITY_THRESHOLD,
            r_tol=RADIAL_LOCATION_TOLERANCE,
            width_tol=RADIAL_WIDTH_TOLERANCE,
        )
        kept_paths = {str(mode["path"]) for mode in kept}
        selected_paths.difference_update(
            str(member["path"])
            for member in cluster
            if str(member["path"]) not in kept_paths
        )
        group_by_path: dict[str, tuple[int, str]] = {}
        for group_index, group in enumerate(type_groups, start=1):
            representative_key = next(
                str(member["mode_key"])
                for member in cluster
                if str(member["path"]) == str(group["rep"]["path"])
            )
            for mode in group["members"]:
                group_by_path[str(mode["path"])] = (group_index, representative_key)

        record_members: list[dict[str, Any]] = []
        for member in cluster:
            path = str(member["path"])
            group_index, representative_key = group_by_path[path]
            record_members.append(
                {
                    "path": path,
                    "mode_key": str(member["mode_key"]),
                    "omega": float(member["omega"]),
                    "action": "KEEP" if path in kept_paths else "DROP",
                    "type_group": group_index,
                    "retained_mode_key": representative_key,
                    "duplicate_rank_score": score_by_path[path],
                    "duplicate_rank_source": "rf_p_good",
                    "selection_reason": "HIGHEST_RF_P_GOOD_WITHIN_MATCHED_MODE_TYPE",
                }
            )
        records.append(
            {
                "ntor": ntor,
                "cluster_index": cluster_index,
                "status": "PROCESSED_RF",
                "diagnostic_message": (
                    "representatives selected by highest RF p_good under the "
                    "preserved frequency and structural-similarity procedure"
                ),
                "members": record_members,
            }
        )

    if saw_scoring_failure:
        status = "COMPLETED_WITH_RF_SCORING_FALLBACK"
    elif unavailable_message and saw_close_cluster:
        status = "SKIPPED_NO_RF_CHECKPOINT"
    elif not saw_close_cluster:
        status = "NO_CLOSE_FREQUENCY_CLUSTERS"
    else:
        status = "COMPLETED_RF"
    return DuplicateResult(
        selected_paths=frozenset(selected_paths),
        cluster_records=tuple(records),
        status=status,
    )


def apply_duplicate_result(
    rows: Sequence[Mapping[str, Any]], result: DuplicateResult
) -> list[dict[str, Any]]:
    """Attach duplicate audit fields without changing any classification field."""
    member_audit: dict[str, Mapping[str, Any]] = {}
    for record in result.cluster_records:
        for member in record["members"]:
            member_audit[str(member["path"])] = member

    output: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        if row.get("final_decision") == "GOOD":
            path = str(row["path"])
            row["selected_final"] = path in result.selected_paths
            audit = member_audit.get(path)
            if audit is not None:
                row["duplicate_rank_score"] = audit["duplicate_rank_score"]
                row["duplicate_rank_source"] = audit["duplicate_rank_source"]
        output.append(row)
    return output


def write_cluster_outputs(out_dir: Path, result: DuplicateResult, rel_freq_tol: float) -> None:
    """Write stable, headered cluster CSV and a deterministic text report."""
    csv_rows: list[dict[str, Any]] = []
    report = [
        "Close-frequency cluster report for final-GOOD NOVA modes",
        (
            f"Parameters: rel_freq_tol={rel_freq_tol} "
            f"sim_tol={SIMILARITY_THRESHOLD} "
            f"r_tol={RADIAL_LOCATION_TOLERANCE} "
            f"width_tol={RADIAL_WIDTH_TOLERANCE}"
        ),
        f"Overall status: {result.status}",
        "=" * 78,
    ]
    multi_records = [record for record in result.cluster_records if len(record["members"]) > 1]
    if not multi_records:
        report.append("No multi-mode close-frequency clusters were present.")
    for record in multi_records:
        members = record["members"]
        report.extend(
            [
                "",
                (
                    f"n={record['ntor']} cluster={record['cluster_index']} "
                    f"size={len(members)} status={record['status']}"
                ),
                f"Diagnostic: {record['diagnostic_message']}",
            ]
        )
        for member in members:
            score = member["duplicate_rank_score"]
            score_text = "" if score == "" else f" score={float(score):.12g}"
            report.append(
                f"  [{member['action']}] {member['mode_key']} "
                f"omega={float(member['omega']):.12g}{score_text} "
                f"retained={member['retained_mode_key']} "
                f"reason={member['selection_reason']}"
            )
            csv_rows.append(
                {
                    "ntor": record["ntor"],
                    "cluster_index": record["cluster_index"],
                    "cluster_status": record["status"],
                    "diagnostic_message": record["diagnostic_message"],
                    **member,
                }
            )
    write_text(out_dir / "frequency_cluster_report.txt", "\n".join(report) + "\n")
    write_dict_csv(out_dir / "frequency_clusters.csv", CLUSTER_FIELDS, csv_rows)


def _primary_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = ""
        if row.get("processing_status") == "INVALID" and row.get(
            "preprocessing_primary_reason"
        ):
            reason = str(row["preprocessing_primary_reason"])
        elif row.get("rule_primary_reason"):
            reason = str(row["rule_primary_reason"])
        if reason:
            counts[reason] += 1
    return dict(sorted(counts.items()))


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    shot: str,
    selected_paths: frozenset[str],
    duplicate_status: str,
    override_audit: OverrideAudit,
    override_sha256: str,
    fraction_tae_threshold: float,
    fraction_eae_threshold: float,
    signed_delta_eae_threshold: float,
    rel_freq_tol: float,
    axis_artifact_config: AxisArtifactConfig | None = None,
    grid_scale_spike_config: GridScaleSpikeConfig | None = None,
    grid_scale_packet_config: GridScalePacketConfig | None = None,
    continuum_crossing_config: ContinuumCrossingConfig | None = None,
    continuum_crossing_window_config: ContinuumCrossingWindowConfig | None = None,
    edge_artifact_config: EdgeArtifactConfig | None = None,
    interior_unresolved_envelope_config: (
        InteriorUnresolvedEnvelopeConfig | None
    ) = None,
    rule_survivor_policy: str = RULE_SURVIVOR_POLICY_REVIEW,
    rule_configuration_name: str = "",
    rule_configuration_schema_version: str = "",
    rule_configuration_sha256: str = "",
) -> dict[str, Any]:
    axis_config = axis_artifact_config or AxisArtifactConfig()
    grid_config = grid_scale_spike_config or GridScaleSpikeConfig()
    packet_config = grid_scale_packet_config or GridScalePacketConfig()
    crossing_config = continuum_crossing_config or ContinuumCrossingConfig()
    cross_window_config = (
        continuum_crossing_window_config or ContinuumCrossingWindowConfig()
    )
    edge_config = edge_artifact_config or EdgeArtifactConfig()
    interior_config = (
        interior_unresolved_envelope_config
        or InteriorUnresolvedEnvelopeConfig()
    )
    rule_rows = [row for row in rows if row.get("rule_version") == RULESET_VERSION]
    transitions = Counter(
        f"{row['rule_decision']}->{row['final_decision']}"
        for row in rows
        if row.get("override_status") == "APPLIED"
        and row.get("rule_decision") != row.get("final_decision")
    )
    final_good_before = sum(row.get("final_decision") == "GOOD" for row in rows)
    summary = {
        "shot": shot,
        "method": "rules",
        "n_total_files": len(rows),
        "n_invalid": sum(row.get("final_decision") == "INVALID" for row in rows),
        "n_tae_like": sum(row.get("gap_region") == "tae_like" for row in rows),
        "n_mixed": sum(row.get("gap_region") == "mixed" for row in rows),
        "n_eae_like": sum(row.get("gap_region") == "eae_like" for row in rows),
        "n_rule_evaluated": len(rule_rows),
        "n_preliminary_bad": sum(row.get("rule_decision") == "BAD" for row in rule_rows),
        "n_preliminary_review": sum(
            row.get("rule_decision") == "REVIEW" for row in rule_rows
        ),
        "n_preliminary_good": sum(row.get("rule_decision") == "GOOD" for row in rule_rows),
        "n_preliminary_invalid": sum(
            row.get("rule_decision") == "INVALID" for row in rule_rows
        ),
        "rule_survivor_policy": rule_survivor_policy,
        "n_rule_survivors_accepted": sum(
            row.get("rule_survivor_accepted") is True for row in rows
        ),
        "n_final_bad": sum(row.get("final_decision") == "BAD" for row in rows),
        "n_final_review": sum(row.get("final_decision") == "REVIEW" for row in rows),
        "n_final_good_before_clustering": final_good_before,
        "n_final_good": len(selected_paths),
        "n_good_removed_as_duplicates": final_good_before - len(selected_paths),
        "duplicate_processing_status": duplicate_status,
        "n_manual_override_rows": override_audit.supplied_rows,
        "n_modes_manually_inspected": override_audit.inspected_rows,
        "n_overrides_applied": override_audit.applied,
        "n_stale_overrides": override_audit.stale,
        "n_ambiguous_overrides": override_audit.ambiguous,
        "n_ineligible_overrides": override_audit.ineligible,
        "n_unmatched_overrides": override_audit.unmatched,
        "n_decisions_changed": override_audit.decisions_changed,
        "transition_counts_json": stable_json(dict(sorted(transitions.items()))),
        "manual_adjudication_performed": override_audit.inspected_rows > 0,
        "manual_override_sha256": override_sha256,
        "primary_reason_counts_json": stable_json(_primary_reason_counts(rows)),
        "rule_set_version": RULESET_VERSION,
        "rule_configuration_name": rule_configuration_name,
        "rule_configuration_schema_version": rule_configuration_schema_version,
        "rule_configuration_sha256": rule_configuration_sha256,
        "fraction_tae_threshold": fraction_tae_threshold,
        "fraction_eae_threshold": fraction_eae_threshold,
        "signed_delta_eae_threshold": signed_delta_eae_threshold,
        "include_mixed_in_tae_like": True,
        "rel_freq_tol": rel_freq_tol,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "radial_location_tolerance": RADIAL_LOCATION_TOLERANCE,
        "radial_width_tolerance": RADIAL_WIDTH_TOLERANCE,
        "axis_artifact_gate_enabled": axis_config.enabled,
        "axis_artifact_r_ax": axis_config.r_ax,
        "axis_artifact_amplitude_min": axis_config.axis_amplitude_min,
        "axis_artifact_width_max_grid": axis_config.axis_width_max_grid,
        "grid_scale_spike_gate_enabled": grid_config.enabled,
        "grid_scale_spike_amplitude_min": grid_config.amplitude_min,
        "grid_scale_spike_width_max_grid": grid_config.width_max_grid,
        "grid_scale_spike_high_r_cutoff_r": grid_config.high_r_cutoff_r,
        "grid_scale_spike_high_r_width_max_grid": (
            grid_config.high_r_width_max_grid
        ),
        "grid_scale_packet_gate_enabled": packet_config.enabled,
        "grid_scale_packet_amplitude_min": packet_config.amplitude_min,
        "grid_scale_packet_step_min": packet_config.step_min,
        "grid_scale_packet_min_large_turns": packet_config.min_large_turns,
        "grid_scale_packet_window_span_grid": packet_config.window_span_grid,
        "grid_scale_packet_peak_r_max": packet_config.peak_r_max,
        "continuum_crossing_gate_enabled": crossing_config.enabled,
        "continuum_crossing_w_threshold": crossing_config.w_cross_threshold,
        "continuum_crossing_window_gate_enabled": cross_window_config.enabled,
        "continuum_crossing_window_half_width_grid": (
            cross_window_config.half_width_grid
        ),
        "continuum_crossing_window_amplitude_min": (
            cross_window_config.amplitude_min
        ),
        "continuum_crossing_window_w_min": cross_window_config.w_min,
        "edge_artifact_gate_enabled": edge_config.enabled,
        "edge_artifact_r_min": edge_config.r_edge_min,
        "edge_artifact_width_max_grid": edge_config.edge_width_max_grid,
        "interior_envelope_gate_enabled": interior_config.enabled,
        "interior_envelope_peak_r_max": interior_config.peak_r_max,
        "interior_envelope_width_max_grid": interior_config.width_max_grid,
        "interior_envelope_extremum_r_min": interior_config.extremum_r_min,
        "interior_envelope_extremum_r_max": interior_config.extremum_r_max,
        "interior_envelope_ext_dr_max": interior_config.ext_dr_max,
        "interior_envelope_ext_df_gap_min": interior_config.ext_df_gap_min,
        "interior_envelope_ext_df_gap_max": interior_config.ext_df_gap_max,
    }
    return summary


def _summary_by_n(
    rows: Sequence[Mapping[str, Any]],
    *,
    shot: str,
    selected_paths: frozenset[str],
    duplicate_status: str,
    override_sha256: str,
    fraction_tae_threshold: float,
    fraction_eae_threshold: float,
    signed_delta_eae_threshold: float,
    rel_freq_tol: float,
    axis_artifact_config: AxisArtifactConfig | None = None,
    grid_scale_spike_config: GridScaleSpikeConfig | None = None,
    grid_scale_packet_config: GridScalePacketConfig | None = None,
    continuum_crossing_config: ContinuumCrossingConfig | None = None,
    continuum_crossing_window_config: ContinuumCrossingWindowConfig | None = None,
    edge_artifact_config: EdgeArtifactConfig | None = None,
    interior_unresolved_envelope_config: (
        InteriorUnresolvedEnvelopeConfig | None
    ) = None,
    rule_survivor_policy: str = RULE_SURVIVOR_POLICY_REVIEW,
    rule_configuration_name: str = "",
    rule_configuration_schema_version: str = "",
    rule_configuration_sha256: str = "",
) -> list[dict[str, Any]]:
    by_n: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_n = row.get("ntor") or row.get("n")
        try:
            by_n[int(raw_n)].append(row)
        except (TypeError, ValueError):
            continue
    summaries: list[dict[str, Any]] = []
    for ntor in sorted(by_n):
        n_rows = by_n[ntor]
        n_selected = frozenset(
            str(row["path"])
            for row in n_rows
            if str(row.get("path", "")) in selected_paths
        )
        n_audit = OverrideAudit(
            supplied_rows=sum(bool(row.get("manual_decision")) for row in n_rows),
            inspected_rows=sum(bool(row.get("manual_decision")) for row in n_rows),
            applied=sum(row.get("override_status") == "APPLIED" for row in n_rows),
            stale=sum(row.get("override_status") == "STALE_FINGERPRINT" for row in n_rows),
            ambiguous=sum(
                row.get("override_status") == "AMBIGUOUS_OVERRIDE" for row in n_rows
            ),
            ineligible=sum(
                row.get("override_status") == "INELIGIBLE_DECISION" for row in n_rows
            ),
            decisions_changed=sum(
                row.get("override_status") == "APPLIED"
                and row.get("rule_decision") != row.get("final_decision")
                for row in n_rows
            ),
        )
        summary = build_summary(
            n_rows,
            shot=shot,
            selected_paths=n_selected,
            duplicate_status=duplicate_status,
            override_audit=n_audit,
            override_sha256=override_sha256,
            fraction_tae_threshold=fraction_tae_threshold,
            fraction_eae_threshold=fraction_eae_threshold,
            signed_delta_eae_threshold=signed_delta_eae_threshold,
            rel_freq_tol=rel_freq_tol,
            axis_artifact_config=axis_artifact_config,
            grid_scale_spike_config=grid_scale_spike_config,
            grid_scale_packet_config=grid_scale_packet_config,
            continuum_crossing_config=continuum_crossing_config,
            continuum_crossing_window_config=continuum_crossing_window_config,
            edge_artifact_config=edge_artifact_config,
            interior_unresolved_envelope_config=(
                interior_unresolved_envelope_config
            ),
            rule_survivor_policy=rule_survivor_policy,
            rule_configuration_name=rule_configuration_name,
            rule_configuration_schema_version=rule_configuration_schema_version,
            rule_configuration_sha256=rule_configuration_sha256,
        )
        summaries.append({"shot": shot, "n": ntor, **summary})
    return summaries


def write_outputs(
    *,
    out_dir: Path,
    preliminary_rows: Sequence[dict[str, Any]],
    final_rows: Sequence[dict[str, Any]],
    summary: Mapping[str, Any],
    summary_by_n: Sequence[dict[str, Any]],
    duplicate_result: DuplicateResult,
    rel_freq_tol: float,
    override_source: Path | None,
) -> None:
    """Write the complete stable output set, including header-only empty lists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preliminary_sorted = sorted(preliminary_rows, key=rule_row_sort_key)
    final_sorted = sorted(final_rows, key=rule_row_sort_key)
    final_classifications = [
        row for row in final_sorted if row.get("processing_status") != "ROUTED_EAE"
    ]
    tae_rows = [
        row for row in final_sorted if row.get("gap_region") in {"tae_like", "mixed"}
    ]
    eae_rows = [row for row in final_sorted if row.get("processing_status") == "ROUTED_EAE"]
    rejected_rows = [row for row in final_sorted if row.get("final_decision") == "INVALID"]
    bad_rows = [row for row in final_sorted if row.get("final_decision") == "BAD"]
    review_rows = [row for row in final_sorted if row.get("final_decision") == "REVIEW"]
    good_rows = [row for row in final_sorted if row.get("final_decision") == "GOOD"]
    final_good_rows = [row for row in good_rows if row.get("selected_final") is True]
    rule_results = [row for row in preliminary_sorted if row.get("rule_version")]

    write_dict_csv(out_dir / "all_modes_rules.csv", RULE_OUTPUT_FIELDS, final_sorted)
    write_dict_csv(out_dir / "tae_like_all.csv", RULE_OUTPUT_FIELDS, tae_rows)
    write_dict_csv(out_dir / "eae_like.csv", RULE_OUTPUT_FIELDS, eae_rows)
    write_dict_csv(out_dir / "rejected_modes.csv", RULE_OUTPUT_FIELDS, rejected_rows)
    write_dict_csv(out_dir / "rule_results.csv", RULE_OUTPUT_FIELDS, rule_results)
    write_dict_csv(
        out_dir / "final_classifications.csv", RULE_OUTPUT_FIELDS, final_classifications
    )
    write_dict_csv(out_dir / "bad_tae_like.csv", RULE_OUTPUT_FIELDS, bad_rows)
    write_dict_csv(out_dir / "review_tae_like.csv", RULE_OUTPUT_FIELDS, review_rows)
    write_dict_csv(out_dir / "good_tae_unchecked.csv", RULE_OUTPUT_FIELDS, good_rows)
    write_dict_csv(out_dir / "good_tae_final.csv", RULE_OUTPUT_FIELDS, final_good_rows)
    write_vertical_summary(out_dir / "shot_summary.csv", SHOT_SUMMARY_FIELDS, summary)
    write_dict_csv(out_dir / "shot_summary_wide.csv", SHOT_SUMMARY_FIELDS, [summary])
    write_dict_csv(
        out_dir / "shot_summary_by_n.csv", SUMMARY_BY_N_FIELDS, summary_by_n
    )
    write_cluster_outputs(out_dir, duplicate_result, rel_freq_tol)

    override_output = out_dir / "manual_overrides.csv"
    if override_source is None:
        write_dict_csv(override_output, MANUAL_OVERRIDE_FIELDS, [])
    elif override_source.resolve() != override_output.resolve():
        shutil.copyfile(override_source, override_output)


def run_shot(
    shot_dir: str | Path,
    out_dir: str | Path,
    *,
    manual_overrides: str | Path | None = None,
    rf_model: str | Path | None = None,
    n_min: int = 1,
    n_max: int = 10,
    pattern: str = "egn*",
    fraction_tae_threshold: float = DEFAULT_FRACTION_TAE_THRESHOLD,
    fraction_eae_threshold: float = DEFAULT_FRACTION_EAE_THRESHOLD,
    signed_delta_eae_threshold: float = DEFAULT_SIGNED_DELTA_EAE_THRESHOLD,
    rel_freq_tol: float = 0.02,
    axis_r_ax: float = DEFAULT_AXIS_R_AX,
    axis_amplitude_min: float | None = DEFAULT_AXIS_AMPLITUDE_MIN,
    axis_width_max_grid: float | None = DEFAULT_AXIS_WIDTH_MAX_GRID,
    grid_scale_amplitude_min: float | None = DEFAULT_GRID_SCALE_AMPLITUDE_MIN,
    grid_scale_width_max_grid: float | None = DEFAULT_GRID_SCALE_WIDTH_MAX_GRID,
    grid_scale_high_r_cutoff_r: float = DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R,
    grid_scale_high_r_width_max_grid: float | None = (
        DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID
    ),
    grid_scale_packet_amplitude_min: float | None = (
        DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN
    ),
    grid_scale_packet_step_min: float = DEFAULT_GRID_SCALE_PACKET_STEP_MIN,
    grid_scale_packet_min_large_turns: int = (
        DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS
    ),
    grid_scale_packet_window_span_grid: int = (
        DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID
    ),
    grid_scale_packet_peak_r_max: float | None = (
        DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX
    ),
    w_cross_threshold: float | None = DEFAULT_W_CROSS_THRESHOLD,
    cross_window_half_width_grid: int = DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID,
    cross_window_amplitude_min: float | None = DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN,
    cross_window_w_min: float | None = DEFAULT_CROSS_WINDOW_W_MIN,
    edge_r_min: float = DEFAULT_EDGE_R_MIN,
    edge_width_max_grid: float | None = DEFAULT_EDGE_WIDTH_MAX_GRID,
    interior_envelope_peak_r_max: float = DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX,
    interior_envelope_width_max_grid: float | None = (
        DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID
    ),
    interior_envelope_extremum_r_min: float = (
        DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN
    ),
    interior_envelope_extremum_r_max: float = (
        DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX
    ),
    interior_envelope_ext_dr_max: float = DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX,
    interior_envelope_ext_df_gap_min: float = (
        DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN
    ),
    interior_envelope_ext_df_gap_max: float = (
        DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX
    ),
    rule_survivor_policy: str = RULE_SURVIVOR_POLICY_REVIEW,
    rule_configuration_name: str = "",
    rule_configuration_schema_version: str = "",
    rule_configuration_sha256: str = "",
) -> ShotRunResult:
    """Run the complete noninteractive deterministic workflow for one shot."""
    if rel_freq_tol <= 0.0 or not math.isfinite(rel_freq_tol):
        raise ValueError("rel_freq_tol must be a finite positive number")
    if rule_survivor_policy not in RULE_SURVIVOR_POLICIES:
        allowed = ", ".join(sorted(RULE_SURVIVOR_POLICIES))
        raise ValueError(f"rule_survivor_policy must be one of: {allowed}")
    axis_config = AxisArtifactConfig(
        r_ax=axis_r_ax,
        axis_amplitude_min=axis_amplitude_min,
        axis_width_max_grid=axis_width_max_grid,
    )
    grid_config = GridScaleSpikeConfig(
        amplitude_min=grid_scale_amplitude_min,
        width_max_grid=grid_scale_width_max_grid,
        high_r_cutoff_r=grid_scale_high_r_cutoff_r,
        high_r_width_max_grid=grid_scale_high_r_width_max_grid,
    )
    packet_config = GridScalePacketConfig(
        amplitude_min=grid_scale_packet_amplitude_min,
        step_min=grid_scale_packet_step_min,
        min_large_turns=grid_scale_packet_min_large_turns,
        window_span_grid=grid_scale_packet_window_span_grid,
        peak_r_max=grid_scale_packet_peak_r_max,
    )
    crossing_config = ContinuumCrossingConfig(
        w_cross_threshold=w_cross_threshold,
    )
    cross_window_config = ContinuumCrossingWindowConfig(
        half_width_grid=cross_window_half_width_grid,
        amplitude_min=cross_window_amplitude_min,
        w_min=cross_window_w_min,
    )
    edge_config = EdgeArtifactConfig(
        r_edge_min=edge_r_min,
        edge_width_max_grid=edge_width_max_grid,
    )
    interior_config = InteriorUnresolvedEnvelopeConfig(
        peak_r_max=interior_envelope_peak_r_max,
        width_max_grid=interior_envelope_width_max_grid,
        extremum_r_min=interior_envelope_extremum_r_min,
        extremum_r_max=interior_envelope_extremum_r_max,
        ext_dr_max=interior_envelope_ext_dr_max,
        ext_df_gap_min=interior_envelope_ext_df_gap_min,
        ext_df_gap_max=interior_envelope_ext_df_gap_max,
    )
    output_dir = Path(out_dir).expanduser()
    existing_override_output = output_dir / "manual_overrides.csv"
    if manual_overrides is None and existing_override_output.exists():
        try:
            _fields, existing_override_rows = read_dict_csv(existing_override_output)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "existing manual_overrides.csv cannot be safely replaced; pass it "
                "with --manual_overrides, choose another output directory, or repair it"
            ) from exc
        if existing_override_rows:
            raise ValueError(
                "refusing to erase existing manual override rows during a run without "
                "--manual_overrides; pass that file explicitly or choose another output directory"
            )
    preprocess = preprocess_shot(
        shot_dir,
        n_min=n_min,
        n_max=n_max,
        pattern=pattern,
        fraction_tae_threshold=fraction_tae_threshold,
        fraction_eae_threshold=fraction_eae_threshold,
        signed_delta_eae_threshold=signed_delta_eae_threshold,
    )

    rule_by_key: dict[str, dict[str, Any]] = {}
    for row in preprocess.tae_rows:
        key = str(row["mode_key"])
        feature_data = preprocess.rule_feature_data.get(key)
        result = evaluate_mode(
            row,
            mode=None if feature_data is None else feature_data.mode,
            low2=None if feature_data is None else feature_data.low2,
            high2=None if feature_data is None else feature_data.high2,
            axis_artifact_config=axis_config,
            grid_scale_spike_config=grid_config,
            grid_scale_packet_config=packet_config,
            continuum_crossing_config=crossing_config,
            continuum_crossing_window_config=cross_window_config,
            edge_artifact_config=edge_config,
            interior_unresolved_envelope_config=interior_config,
        )
        rule_by_key[key] = result.as_output_row(row)

    preliminary_rows: list[dict[str, Any]] = []
    for source_row in preprocess.rows:
        key = str(source_row["mode_key"])
        preliminary_rows.append(rule_by_key.get(key, dict(source_row)))

    override_path = Path(manual_overrides).expanduser() if manual_overrides else None
    if override_path is not None:
        overrides = load_manual_overrides(override_path)
        override_digest = sha256_file(override_path)
    else:
        overrides = []
        override_digest = ""
    automatic_rows = apply_rule_survivor_policy(
        preliminary_rows,
        rule_survivor_policy,
    )
    final_rows, override_audit = apply_manual_overrides(automatic_rows, overrides)
    duplicate_result = deduplicate_final_good(
        final_rows,
        rf_model_path=rf_model,
        rel_freq_tol=rel_freq_tol,
    )
    final_rows = apply_duplicate_result(final_rows, duplicate_result)
    final_rows = sorted(final_rows, key=rule_row_sort_key)

    summary = build_summary(
        final_rows,
        shot=preprocess.shot,
        selected_paths=duplicate_result.selected_paths,
        duplicate_status=duplicate_result.status,
        override_audit=override_audit,
        override_sha256=override_digest,
        fraction_tae_threshold=fraction_tae_threshold,
        fraction_eae_threshold=fraction_eae_threshold,
        signed_delta_eae_threshold=signed_delta_eae_threshold,
        rel_freq_tol=rel_freq_tol,
        axis_artifact_config=axis_config,
        grid_scale_spike_config=grid_config,
        grid_scale_packet_config=packet_config,
        continuum_crossing_config=crossing_config,
        continuum_crossing_window_config=cross_window_config,
        edge_artifact_config=edge_config,
        interior_unresolved_envelope_config=interior_config,
        rule_survivor_policy=rule_survivor_policy,
        rule_configuration_name=rule_configuration_name,
        rule_configuration_schema_version=rule_configuration_schema_version,
        rule_configuration_sha256=rule_configuration_sha256,
    )
    summary_by_n = _summary_by_n(
        final_rows,
        shot=preprocess.shot,
        selected_paths=duplicate_result.selected_paths,
        duplicate_status=duplicate_result.status,
        override_sha256=override_digest,
        fraction_tae_threshold=fraction_tae_threshold,
        fraction_eae_threshold=fraction_eae_threshold,
        signed_delta_eae_threshold=signed_delta_eae_threshold,
        rel_freq_tol=rel_freq_tol,
        axis_artifact_config=axis_config,
        grid_scale_spike_config=grid_config,
        grid_scale_packet_config=packet_config,
        continuum_crossing_config=crossing_config,
        continuum_crossing_window_config=cross_window_config,
        edge_artifact_config=edge_config,
        interior_unresolved_envelope_config=interior_config,
        rule_survivor_policy=rule_survivor_policy,
        rule_configuration_name=rule_configuration_name,
        rule_configuration_schema_version=rule_configuration_schema_version,
        rule_configuration_sha256=rule_configuration_sha256,
    )
    write_outputs(
        out_dir=output_dir,
        preliminary_rows=preliminary_rows,
        final_rows=final_rows,
        summary=summary,
        summary_by_n=summary_by_n,
        duplicate_result=duplicate_result,
        rel_freq_tol=rel_freq_tol,
        override_source=override_path,
    )
    return ShotRunResult(
        preprocess=preprocess,
        preliminary_rows=tuple(preliminary_rows),
        final_rows=tuple(final_rows),
        summary=summary,
        duplicate_result=duplicate_result,
    )


def run_configured_shot(
    shot_dir: str | Path,
    out_dir: str | Path,
    *,
    rule_config: str | Path = PRODUCTION_RULE_CONFIG_NAME,
    manual_overrides: str | Path | None = None,
    rf_model: str | Path | None = None,
    n_min: int = 1,
    n_max: int = 10,
    pattern: str = "egn*",
    rule_survivor_policy: str = RULE_SURVIVOR_POLICY_REVIEW,
) -> ShotRunResult:
    """Run one shot from a strict named rule configuration."""
    configuration = load_rule_run_configuration(rule_config)
    return run_shot(
        shot_dir,
        out_dir,
        manual_overrides=manual_overrides,
        rf_model=rf_model,
        n_min=n_min,
        n_max=n_max,
        pattern=pattern,
        rule_survivor_policy=rule_survivor_policy,
        rule_configuration_name=configuration.name,
        rule_configuration_schema_version=configuration.schema_version,
        rule_configuration_sha256=configuration.sha256,
        **configuration.run_kwargs,
    )


def _explicit_rule_config_overrides(argv: Sequence[str]) -> list[str]:
    """Return config-owned CLI options explicitly supplied by the caller."""
    supplied = {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    }
    return sorted(supplied & RULE_CONFIG_OVERRIDE_OPTIONS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess and deterministically rule-sort one NOVA shot. RF is "
            "optional and used only to rank final-GOOD duplicate representatives."
        )
    )
    parser.add_argument("--shot_dir", required=True, help="Shot containing N1, N2, ...")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--rule_config",
        help=(
            "Frozen rule-run configuration name under configs/rules, or an "
            "explicit JSON-compatible YAML path. Config-owned threshold and "
            "gate flags cannot be combined with this option."
        ),
    )
    parser.add_argument(
        "--manual_overrides",
        help="Reusable manual_overrides.csv to validate and apply",
    )
    parser.add_argument(
        "--rf_model",
        help="Optional RF checkpoint used only for final-GOOD duplicate ranking",
    )
    parser.add_argument("--n_min", type=int, default=1, help="Smallest N# to scan")
    parser.add_argument("--n_max", type=int, default=10, help="Largest N# to scan")
    parser.add_argument("--pattern", default="egn*", help="Mode-file glob")
    parser.add_argument("--rel_freq_tol", type=float, default=0.02)
    parser.add_argument(
        "--fraction_tae_threshold", type=float, default=DEFAULT_FRACTION_TAE_THRESHOLD
    )
    parser.add_argument(
        "--fraction_eae_threshold", type=float, default=DEFAULT_FRACTION_EAE_THRESHOLD
    )
    parser.add_argument(
        "--signed_delta_eae_threshold",
        type=float,
        default=DEFAULT_SIGNED_DELTA_EAE_THRESHOLD,
    )
    parser.add_argument(
        "--axis_r_ax",
        type=float,
        default=DEFAULT_AXIS_R_AX,
        help=(
            "Inclusive normalized radial window r <= r_ax used to find all "
            "axis local peaks"
        ),
    )
    parser.add_argument(
        "--axis_amplitude_min",
        type=float,
        default=DEFAULT_AXIS_AMPLITUDE_MIN,
        help=(
            "Minimum normalized harmonic amplitude for BAD_AXIS_SPIKE "
            f"(default: {DEFAULT_AXIS_AMPLITUDE_MIN})"
        ),
    )
    parser.add_argument(
        "--axis_width_max_grid",
        type=float,
        default=DEFAULT_AXIS_WIDTH_MAX_GRID,
        help=(
            "Maximum full-grid half-maximum width in radial intervals for "
            f"BAD_AXIS_SPIKE (default: {DEFAULT_AXIS_WIDTH_MAX_GRID:g})"
        ),
    )
    parser.add_argument(
        "--disable_axis_artifact",
        action="store_true",
        help="Calculate axis features but disable the BAD_AXIS_SPIKE decision gate",
    )
    parser.add_argument(
        "--grid_scale_amplitude_min",
        type=float,
        default=DEFAULT_GRID_SCALE_AMPLITUDE_MIN,
        help=(
            "Minimum absolute normalized signed-lobe amplitude for "
            f"BAD_GRID_SCALE_SPIKE (default: {DEFAULT_GRID_SCALE_AMPLITUDE_MIN})"
        ),
    )
    parser.add_argument(
        "--grid_scale_width_max_grid",
        type=float,
        default=DEFAULT_GRID_SCALE_WIDTH_MAX_GRID,
        help=(
            "Maximum signed-lobe half-maximum width in radial intervals for "
            f"BAD_GRID_SCALE_SPIKE (default: {DEFAULT_GRID_SCALE_WIDTH_MAX_GRID:g})"
        ),
    )
    parser.add_argument(
        "--grid_scale_high_r_cutoff_r",
        type=float,
        default=DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R,
        help=(
            "Peaks strictly above this normalized radius use the high-r "
            "grid-scale width limit "
            f"(default: {DEFAULT_GRID_SCALE_HIGH_R_CUTOFF_R:g})"
        ),
    )
    parser.add_argument(
        "--grid_scale_high_r_width_max_grid",
        type=float,
        default=DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID,
        help=(
            "Maximum signed-lobe half-maximum width in radial intervals for "
            "BAD_GRID_SCALE_SPIKE peaks strictly above the high-r cutoff "
            f"(default: {DEFAULT_GRID_SCALE_HIGH_R_WIDTH_MAX_GRID:g})"
        ),
    )
    parser.add_argument(
        "--disable_grid_scale_spike",
        action="store_true",
        help=(
            "Calculate grid-scale features at the configured width but disable "
            "the BAD_GRID_SCALE_SPIKE decision gate"
        ),
    )
    parser.add_argument(
        "--grid_scale_packet_amplitude_min",
        type=float,
        default=DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN,
        help=(
            "Minimum absolute amplitude in a qualifying short window for "
            "BAD_GRID_SCALE_PACKET "
            f"(default: {DEFAULT_GRID_SCALE_PACKET_AMPLITUDE_MIN})"
        ),
    )
    parser.add_argument(
        "--grid_scale_packet_step_min",
        type=float,
        default=DEFAULT_GRID_SCALE_PACKET_STEP_MIN,
        help=(
            "Minimum absolute signed-amplitude difference between adjacent "
            "radial samples counted as a large packet step "
            f"(default: {DEFAULT_GRID_SCALE_PACKET_STEP_MIN})"
        ),
    )
    parser.add_argument(
        "--grid_scale_packet_min_large_turns",
        type=int,
        default=DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS,
        help=(
            "Minimum number of sharp turning points whose two adjacent "
            "steps are large and oppositely directed in one packet window "
            f"(default: {DEFAULT_GRID_SCALE_PACKET_MIN_LARGE_TURNS})"
        ),
    )
    parser.add_argument(
        "--grid_scale_packet_window_span_grid",
        type=int,
        default=DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID,
        help=(
            "Packet window span in radial grid intervals; the number of "
            "samples is one larger "
            f"(default: {DEFAULT_GRID_SCALE_PACKET_WINDOW_SPAN_GRID})"
        ),
    )
    parser.add_argument(
        "--grid_scale_packet_peak_r_max",
        type=float,
        default=DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX,
        help=(
            "Inclusive maximum radius of a qualifying packet window's "
            "largest absolute sample "
            f"(default: {DEFAULT_GRID_SCALE_PACKET_PEAK_R_MAX})"
        ),
    )
    parser.add_argument(
        "--disable_grid_scale_packet",
        action="store_true",
        help=(
            "Calculate short-window packet features but disable the "
            "BAD_GRID_SCALE_PACKET decision gate"
        ),
    )
    parser.add_argument(
        "--w_cross_threshold",
        type=float,
        default=DEFAULT_W_CROSS_THRESHOLD,
        help=(
            "Reject a mode as BAD_CONT_CROSS when n_cross > 0 and "
            "W_star_max is strictly greater than this peak-normalized radial "
            f"energy (default: {DEFAULT_W_CROSS_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--disable_cont_cross",
        action="store_true",
        help=(
            "Retain continuum-crossing measurements but disable the "
            "BAD_CONT_CROSS decision gate"
        ),
    )
    parser.add_argument(
        "--cross_window_half_width_grid",
        type=int,
        default=DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID,
        help=(
            "Inclusive half-width in radial grid intervals around every true "
            "crossing for BAD_CONT_CROSS_WINDOW "
            f"(default: {DEFAULT_CROSS_WINDOW_HALF_WIDTH_GRID})"
        ),
    )
    parser.add_argument(
        "--cross_window_amplitude_min",
        type=float,
        default=DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN,
        help=(
            "Minimum individual-harmonic absolute amplitude within the crossing "
            "window for BAD_CONT_CROSS_WINDOW "
            f"(default: {DEFAULT_CROSS_WINDOW_AMPLITUDE_MIN})"
        ),
    )
    parser.add_argument(
        "--cross_window_w_min",
        type=float,
        default=DEFAULT_CROSS_WINDOW_W_MIN,
        help=(
            "Minimum peak-normalized radial energy within the crossing window "
            "for BAD_CONT_CROSS_WINDOW "
            f"(default: {DEFAULT_CROSS_WINDOW_W_MIN})"
        ),
    )
    parser.add_argument(
        "--disable_cont_cross_window",
        action="store_true",
        help=(
            "Calculate crossing-window measurements but disable the "
            "BAD_CONT_CROSS_WINDOW decision gate"
        ),
    )
    parser.add_argument(
        "--edge_r_min",
        type=float,
        default=DEFAULT_EDGE_R_MIN,
        help=(
            "Inclusive normalized radius at which a global energy peak is "
            f"edge-localized for BAD_EDGE_SPIKE (default: {DEFAULT_EDGE_R_MIN})"
        ),
    )
    parser.add_argument(
        "--edge_width_max_grid",
        type=float,
        default=DEFAULT_EDGE_WIDTH_MAX_GRID,
        help=(
            "Maximum full-grid total-energy half-maximum width in radial "
            f"intervals for BAD_EDGE_SPIKE (default: {DEFAULT_EDGE_WIDTH_MAX_GRID:g})"
        ),
    )
    parser.add_argument(
        "--disable_edge_artifact",
        action="store_true",
        help=(
            "Calculate edge-envelope and harmonic audit features but disable "
            "the BAD_EDGE_SPIKE decision gate"
        ),
    )
    parser.add_argument(
        "--interior_envelope_peak_r_max",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX,
        help=(
            "Inclusive maximum total-energy peak radius for "
            "BAD_INTERIOR_UNRESOLVED_ENVELOPE "
            f"(default: {DEFAULT_INTERIOR_ENVELOPE_PEAK_R_MAX:g})"
        ),
    )
    parser.add_argument(
        "--interior_envelope_width_max_grid",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID,
        help=(
            "Inclusive maximum connected total-energy FWHM in radial-grid "
            "intervals for BAD_INTERIOR_UNRESOLVED_ENVELOPE "
            f"(default: {DEFAULT_INTERIOR_ENVELOPE_WIDTH_MAX_GRID:g})"
        ),
    )
    parser.add_argument(
        "--interior_envelope_extremum_r_min",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN,
        help=(
            "Inclusive minimum radius for the gate-specific continuum-extremum "
            f"exception search (default: {DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MIN:g})"
        ),
    )
    parser.add_argument(
        "--interior_envelope_extremum_r_max",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX,
        help=(
            "Inclusive maximum radius for the gate-specific continuum-extremum "
            f"exception search (default: {DEFAULT_INTERIOR_ENVELOPE_EXTREMUM_R_MAX:g})"
        ),
    )
    parser.add_argument(
        "--interior_envelope_ext_dr_max",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX,
        help=(
            "Inclusive maximum radial mismatch for the narrow-envelope "
            f"extremum exception (default: {DEFAULT_INTERIOR_ENVELOPE_EXT_DR_MAX:g})"
        ),
    )
    parser.add_argument(
        "--interior_envelope_ext_df_gap_min",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN,
        help=(
            "Inclusive minimum signed gap-side frequency clearance for the "
            "narrow-envelope extremum exception "
            f"(default: {DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MIN:g})"
        ),
    )
    parser.add_argument(
        "--interior_envelope_ext_df_gap_max",
        type=float,
        default=DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX,
        help=(
            "Inclusive maximum signed gap-side frequency clearance for the "
            "narrow-envelope extremum exception "
            f"(default: {DEFAULT_INTERIOR_ENVELOPE_EXT_DF_GAP_MAX:g})"
        ),
    )
    parser.add_argument(
        "--disable_interior_unresolved_envelope",
        action="store_true",
        help=(
            "Calculate interior-envelope and extremum evidence but disable "
            "the BAD_INTERIOR_UNRESOLVED_ENVELOPE decision gate"
        ),
    )
    args = parser.parse_args(raw_args)
    args.rule_configuration_name = ""
    args.rule_configuration_schema_version = ""
    args.rule_configuration_sha256 = ""
    if args.rule_config:
        conflicts = _explicit_rule_config_overrides(raw_args)
        if conflicts:
            parser.error(
                "--rule_config freezes its gate and threshold values; remove "
                "these conflicting options: " + ", ".join(conflicts)
            )
        try:
            configuration = load_rule_run_configuration(args.rule_config)
        except ValueError as exc:
            parser.error(str(exc))
        for key, value in configuration.run_kwargs.items():
            setattr(args, key, value)
        args.rule_configuration_name = configuration.name
        args.rule_configuration_schema_version = configuration.schema_version
        args.rule_configuration_sha256 = configuration.sha256
    return args


def main() -> None:
    args = parse_args()
    result = run_shot(
        args.shot_dir,
        args.out_dir,
        manual_overrides=args.manual_overrides,
        rf_model=args.rf_model,
        n_min=args.n_min,
        n_max=args.n_max,
        pattern=args.pattern,
        fraction_tae_threshold=args.fraction_tae_threshold,
        fraction_eae_threshold=args.fraction_eae_threshold,
        signed_delta_eae_threshold=args.signed_delta_eae_threshold,
        rel_freq_tol=args.rel_freq_tol,
        axis_r_ax=args.axis_r_ax,
        axis_amplitude_min=(
            None if args.disable_axis_artifact else args.axis_amplitude_min
        ),
        axis_width_max_grid=(
            None if args.disable_axis_artifact else args.axis_width_max_grid
        ),
        grid_scale_amplitude_min=(
            None
            if args.disable_grid_scale_spike
            else args.grid_scale_amplitude_min
        ),
        grid_scale_width_max_grid=args.grid_scale_width_max_grid,
        grid_scale_high_r_cutoff_r=args.grid_scale_high_r_cutoff_r,
        grid_scale_high_r_width_max_grid=(
            args.grid_scale_high_r_width_max_grid
        ),
        grid_scale_packet_amplitude_min=(
            None
            if args.disable_grid_scale_packet
            else args.grid_scale_packet_amplitude_min
        ),
        grid_scale_packet_step_min=args.grid_scale_packet_step_min,
        grid_scale_packet_min_large_turns=(
            args.grid_scale_packet_min_large_turns
        ),
        grid_scale_packet_window_span_grid=(
            args.grid_scale_packet_window_span_grid
        ),
        grid_scale_packet_peak_r_max=args.grid_scale_packet_peak_r_max,
        w_cross_threshold=(
            None if args.disable_cont_cross else args.w_cross_threshold
        ),
        cross_window_half_width_grid=args.cross_window_half_width_grid,
        cross_window_amplitude_min=(
            None
            if args.disable_cont_cross_window
            else args.cross_window_amplitude_min
        ),
        cross_window_w_min=(
            None if args.disable_cont_cross_window else args.cross_window_w_min
        ),
        edge_r_min=args.edge_r_min,
        edge_width_max_grid=(
            None if args.disable_edge_artifact else args.edge_width_max_grid
        ),
        interior_envelope_peak_r_max=args.interior_envelope_peak_r_max,
        interior_envelope_width_max_grid=(
            None
            if args.disable_interior_unresolved_envelope
            else args.interior_envelope_width_max_grid
        ),
        interior_envelope_extremum_r_min=(
            args.interior_envelope_extremum_r_min
        ),
        interior_envelope_extremum_r_max=(
            args.interior_envelope_extremum_r_max
        ),
        interior_envelope_ext_dr_max=args.interior_envelope_ext_dr_max,
        interior_envelope_ext_df_gap_min=(
            args.interior_envelope_ext_df_gap_min
        ),
        interior_envelope_ext_df_gap_max=(
            args.interior_envelope_ext_df_gap_max
        ),
        rule_configuration_name=args.rule_configuration_name,
        rule_configuration_schema_version=(
            args.rule_configuration_schema_version
        ),
        rule_configuration_sha256=args.rule_configuration_sha256,
    )
    summary = result.summary
    print(f"Shot: {summary['shot']}")
    print(f"Discovered modes: {summary['n_total_files']}")
    print(
        "Frequency routing: "
        f"tae_like={summary['n_tae_like']} mixed={summary['n_mixed']} "
        f"eae_like={summary['n_eae_like']} invalid={summary['n_invalid']}"
    )
    print(
        "Final TAE decisions: "
        f"BAD={summary['n_final_bad']} REVIEW={summary['n_final_review']} "
        f"GOOD before clustering={summary['n_final_good_before_clustering']} "
        f"GOOD final={summary['n_final_good']}"
    )
    if summary["rule_configuration_name"]:
        print(
            "Rule configuration: "
            f"{summary['rule_configuration_name']} "
            f"schema={summary['rule_configuration_schema_version']} "
            f"sha256={summary['rule_configuration_sha256']}"
        )
    print(
        "Axis artifact gate: "
        f"enabled={summary['axis_artifact_gate_enabled']} "
        f"r_ax={summary['axis_artifact_r_ax']} "
        f"amplitude_min={summary['axis_artifact_amplitude_min']} "
        f"width_max_grid={summary['axis_artifact_width_max_grid']}"
    )
    print(
        "Grid-scale spike gate: "
        f"enabled={summary['grid_scale_spike_gate_enabled']} "
        f"amplitude_min={summary['grid_scale_spike_amplitude_min']} "
        f"width_max_grid={summary['grid_scale_spike_width_max_grid']} "
        f"high_r_cutoff_r={summary['grid_scale_spike_high_r_cutoff_r']} "
        "high_r_width_max_grid="
        f"{summary['grid_scale_spike_high_r_width_max_grid']}"
    )
    print(
        "Grid-scale packet gate: "
        f"enabled={summary['grid_scale_packet_gate_enabled']} "
        f"amplitude_min={summary['grid_scale_packet_amplitude_min']} "
        f"step_min={summary['grid_scale_packet_step_min']} "
        f"min_large_turns={summary['grid_scale_packet_min_large_turns']} "
        "window_span_grid="
        f"{summary['grid_scale_packet_window_span_grid']} "
        f"peak_r_max={summary['grid_scale_packet_peak_r_max']}"
    )
    print(
        "Continuum-crossing gate: "
        f"enabled={summary['continuum_crossing_gate_enabled']} "
        f"w_cross_threshold={summary['continuum_crossing_w_threshold']}"
    )
    print(
        "Continuum-crossing window gate: "
        f"enabled={summary['continuum_crossing_window_gate_enabled']} "
        "half_width_grid="
        f"{summary['continuum_crossing_window_half_width_grid']} "
        "amplitude_min="
        f"{summary['continuum_crossing_window_amplitude_min']} "
        f"w_min={summary['continuum_crossing_window_w_min']}"
    )
    print(
        "Edge artifact gate: "
        f"enabled={summary['edge_artifact_gate_enabled']} "
        f"r_edge_min={summary['edge_artifact_r_min']} "
        f"width_max_grid={summary['edge_artifact_width_max_grid']}"
    )
    print(
        "Interior unresolved-envelope gate: "
        f"enabled={summary['interior_envelope_gate_enabled']} "
        f"peak_r_max={summary['interior_envelope_peak_r_max']} "
        f"width_max_grid={summary['interior_envelope_width_max_grid']} "
        "extremum_r_range="
        f"[{summary['interior_envelope_extremum_r_min']}, "
        f"{summary['interior_envelope_extremum_r_max']}] "
        f"ext_dr_max={summary['interior_envelope_ext_dr_max']} "
        "ext_df_gap_range="
        f"[{summary['interior_envelope_ext_df_gap_min']}, "
        f"{summary['interior_envelope_ext_df_gap_max']}]"
    )
    print(f"Duplicate processing: {summary['duplicate_processing_status']}")
    if args.manual_overrides:
        print(
            "Manual overrides: "
            f"applied={summary['n_overrides_applied']} "
            f"stale={summary['n_stale_overrides']} "
            f"ambiguous={summary['n_ambiguous_overrides']} "
            f"ineligible={summary['n_ineligible_overrides']} "
            f"unmatched={summary['n_unmatched_overrides']}"
        )
        print(f"Manual override SHA-256: {summary['manual_override_sha256']}")
    print(f"Wrote outputs to: {Path(args.out_dir).expanduser()}")


if __name__ == "__main__":
    main()
