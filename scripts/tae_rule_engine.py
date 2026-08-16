#!/usr/bin/env python3
"""Pure per-mode interface for deterministic NOVA TAE rule evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from tae_rule_io import empty_rule_row, stable_json


RULESET_VERSION = "tae-rules-placeholder-v1"
RULESET_NOT_IMPLEMENTED = "RULESET_NOT_IMPLEMENTED"


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
    features: Mapping[str, Any] = field(
        default_factory=lambda: {"future_feature_placeholder": None}
    )
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


def evaluate_mode(preprocessed_row: Mapping[str, Any]) -> RuleResult:
    """Evaluate one valid TAE-side row without filesystem or model access."""
    path = str(preprocessed_row.get("path", ""))
    mode_key = str(preprocessed_row.get("mode_key", ""))
    shot = str(preprocessed_row.get("shot", ""))
    fingerprint = str(preprocessed_row.get("input_fingerprint", ""))
    gap_region = str(preprocessed_row.get("gap_region", ""))
    try:
        ntor = int(preprocessed_row.get("ntor"))
        frequency = float(preprocessed_row.get("omega"))
        if not math.isfinite(frequency):
            raise ValueError("frequency is non-finite")
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
            features={"future_feature_placeholder": None},
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
    )
