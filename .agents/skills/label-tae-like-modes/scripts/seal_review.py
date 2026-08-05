#!/usr/bin/env python3
"""Validate and seal a complete independent TAE-like mode review."""

from __future__ import annotations

import argparse
from pathlib import Path

from _blind_review_io import (
    atomic_write_csv,
    atomic_write_text,
    ensure_new_targets,
    normalize_validity,
    parse_bool,
    read_dict_csv,
    sha256_file,
    stable_mode_key,
)


MANIFEST_FIELDS = {
    "blind_id",
    "path",
    "shot",
    "n",
    "signed_delta",
    "fraction_below_upper2",
    "gap_region",
    "error",
}
DECISION_FIELDS = {
    "blind_id",
    "path",
    "validity",
    "confidence",
    "reason",
    "prior_seen",
}
OUTPUT_FIELDS = [
    "blind_id",
    "path",
    "validity",
    "confidence",
    "reason",
    "prior_seen",
    "reviewer",
]
CONFIDENCE_VALUES = {"high", "medium", "low"}


def index_unique(rows: list[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        key = stable_mode_key(row.get("path", ""))
        if key in indexed:
            raise ValueError(f"{source}:{line_number}: duplicate mode {key}")
        indexed[key] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact blind-manifest coverage and seal independent "
            "good/bad/skip decisions with a SHA-256 sidecar."
        )
    )
    parser.add_argument("--manifest", required=True, help="Prepared blind manifest")
    parser.add_argument("--decisions", required=True, help="Completed decision CSV")
    parser.add_argument("--out", required=True, help="Sealed output CSV")
    parser.add_argument("--reviewer", required=True, help="Reviewer identifier")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite explicitly named outputs"
    )
    args = parser.parse_args()

    reviewer = args.reviewer.strip()
    if not reviewer:
        raise ValueError("--reviewer must not be blank")

    manifest_fields, manifest_rows = read_dict_csv(args.manifest)
    decision_fields, decision_rows = read_dict_csv(args.decisions)
    if not manifest_rows:
        raise ValueError("blind manifest has no modes")
    if set(manifest_fields) - MANIFEST_FIELDS or "path" not in manifest_fields:
        raise ValueError(
            "manifest has unsafe/unsupported columns: "
            + ", ".join(sorted(set(manifest_fields) - MANIFEST_FIELDS))
        )
    unsupported = set(decision_fields) - DECISION_FIELDS
    if unsupported:
        raise ValueError(
            "decision CSV contains forbidden extra columns: "
            + ", ".join(sorted(unsupported))
        )
    required = {"path", "validity", "confidence", "reason"}
    missing_fields = required - set(decision_fields)
    if missing_fields:
        raise ValueError(
            "decision CSV is missing required columns: "
            + ", ".join(sorted(missing_fields))
        )

    manifest_by_key = index_unique(manifest_rows, str(args.manifest))
    decision_by_key = index_unique(decision_rows, str(args.decisions))
    manifest_keys = set(manifest_by_key)
    decision_keys = set(decision_by_key)
    if manifest_keys != decision_keys:
        missing = sorted(manifest_keys - decision_keys)
        extra = sorted(decision_keys - manifest_keys)
        raise ValueError(
            f"coverage mismatch: missing={len(missing)} extra={len(extra)}; "
            f"missing sample={missing[:3]} extra sample={extra[:3]}"
        )

    sealed_rows: list[dict[str, str]] = []
    prior_seen_count = 0
    label_counts = {"good": 0, "bad": 0, "skip": 0}
    for manifest_row in manifest_rows:
        key = stable_mode_key(manifest_row["path"])
        decision = decision_by_key[key]
        validity = normalize_validity(decision["validity"])
        confidence = decision["confidence"].strip().lower()
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(
                f"{key}: confidence must be high, medium, or low, "
                f"got {decision['confidence']!r}"
            )
        reason = decision["reason"].strip()
        if not reason:
            raise ValueError(f"{key}: reason must not be blank")
        prior_seen = parse_bool(
            decision.get("prior_seen", "false"), field=f"{key}: prior_seen"
        )
        if (
            decision.get("blind_id", "").strip()
            and decision["blind_id"].strip() != manifest_row.get("blind_id", "").strip()
        ):
            raise ValueError(f"{key}: blind_id differs from manifest")
        label_counts[validity] += 1
        prior_seen_count += int(prior_seen)
        sealed_rows.append(
            {
                "blind_id": manifest_row.get("blind_id", ""),
                "path": manifest_row["path"],
                "validity": validity,
                "confidence": confidence,
                "reason": reason,
                "prior_seen": str(prior_seen).lower(),
                "reviewer": reviewer,
            }
        )

    output = Path(args.out).expanduser()
    sidecar = Path(str(output) + ".sha256")
    ensure_new_targets([output, sidecar], args.force)
    atomic_write_csv(output, OUTPUT_FIELDS, sealed_rows)
    digest = sha256_file(output)
    atomic_write_text(sidecar, f"{digest}  {output.name}\n")

    print(f"Sealed {len(sealed_rows)} decisions: {output}")
    print(
        "Labels: "
        + ", ".join(f"{label}={label_counts[label]}" for label in ("good", "bad", "skip"))
    )
    print(f"prior_seen={prior_seen_count}; clean={len(sealed_rows) - prior_seen_count}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
