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
    compute_named_features_for_mode,
    get_feature_names,
    get_feature_schema_version,
)
from tae_rule_io import empty_rule_row, stable_json


RULESET_VERSION = "tae-rules-placeholder-v1"
RULESET_NOT_IMPLEMENTED = "RULESET_NOT_IMPLEMENTED"
RULE_FEATURE_EXTRACTION_FAILED = "RULE_FEATURE_EXTRACTION_FAILED"
RULE_FEATURE_NAMES = tuple(
    get_feature_names(include_crossing_features=True, include_extremum_features=True)
)
RULE_FEATURE_SCHEMA_VERSION = "tae-rule-features-rf31-v1"
RULE_FEATURE_SOURCE_SCHEMA_VERSION = get_feature_schema_version(
    include_crossing_features=True,
    include_extremum_features=True,
)
RULE_FEATURE_METADATA_NAMES = (
    "feature_schema_version",
    "source_feature_schema_version",
    "extremum_match_found",
)


def empty_rule_features() -> dict[str, Any]:
    """Return the complete rule-feature schema with unavailable values as null."""
    return {
        "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
        "extremum_match_found": None,
        **{name: None for name in RULE_FEATURE_NAMES},
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
) -> RuleResult:
    """Extract named features and evaluate one valid, preprocessed TAE mode."""
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
            features=empty_rule_features(),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    try:
        if mode is None or low2 is None or high2 is None:
            raise ValueError("mode, low2, and high2 arrays are required")
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
            null_missing_extremum=True,
            return_feature_status=True,
        )
        if tuple(named_features) != RULE_FEATURE_NAMES:
            raise ValueError("named feature order does not match the rule schema")
        if not all(
            value is None or math.isfinite(value)
            for value in named_features.values()
        ):
            raise ValueError("one or more rule features are non-finite")
        features = {
            "feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
            "source_feature_schema_version": RULE_FEATURE_SOURCE_SCHEMA_VERSION,
            **feature_status,
            **named_features,
        }
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
            features=empty_rule_features(),
            processing_status="INVALID",
            diagnostic_message=f"{type(exc).__name__}: {exc}",
        )

    # Deliberately conservative until quantitative rejection rules and positive
    # GOOD templates are approved. Not rejected is not equivalent to GOOD.
    return RuleResult(
        path=path,
        mode_key=mode_key,
        shot=shot,
        ntor=ntor,
        frequency=frequency,
        input_fingerprint=fingerprint,
        gap_region=gap_region,
        decision="REVIEW",
        primary_reason=RULESET_NOT_IMPLEMENTED,
        triggered_rules=(RULESET_NOT_IMPLEMENTED,),
        features=features,
    )
