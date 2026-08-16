#!/usr/bin/env python3
"""Compare a verified sealed blind review with a post-seal reference list."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from _blind_review_io import (
    atomic_write_csv,
    ensure_new_targets,
    normalize_validity,
    parse_bool,
    read_dict_csv,
    sha256_file,
    stable_mode_key,
)


PATH_FIELDS = ("path", "filepath", "mode_path")
LABEL_FIELDS = ("validity", "label", "class", "target", "manual_label")
OUTPUT_FIELDS = [
    "blind_id",
    "path",
    "blind_label",
    "reference_label",
    "agreement",
    "confidence",
    "reason",
    "prior_seen",
]


def read_reference(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser()
    with source.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and any(c.strip() for c in row)]
    if not rows:
        raise ValueError(f"reference CSV is empty: {source}")

    header = [cell.strip().lower() for cell in rows[0]]
    path_fields = [field for field in PATH_FIELDS if field in header]
    label_fields = [field for field in LABEL_FIELDS if field in header]
    parsed: list[dict[str, str]] = []

    if len(path_fields) == 1 and len(label_fields) == 1:
        path_index = header.index(path_fields[0])
        label_index = header.index(label_fields[0])
        for line_number, row in enumerate(rows[1:], start=2):
            if max(path_index, label_index) >= len(row):
                raise ValueError(f"{source}:{line_number}: missing path or label")
            parsed.append(
                {"path": row[path_index].strip(), "validity": row[label_index].strip()}
            )
        return parsed

    for line_number, row in enumerate(rows, start=1):
        if len(row) < 2:
            raise ValueError(
                f"{source}:{line_number}: headerless reference requires path,label"
            )
        parsed.append({"path": row[0].strip(), "validity": row[1].strip()})
    return parsed


def index_labels(
    rows: list[dict[str, str]], source: str
) -> dict[str, tuple[str, dict[str, str]]]:
    indexed: dict[str, tuple[str, dict[str, str]]] = {}
    for line_number, row in enumerate(rows, start=2):
        key = stable_mode_key(row["path"])
        if key in indexed:
            raise ValueError(f"{source}:{line_number}: duplicate mode {key}")
        indexed[key] = (normalize_validity(row["validity"]), row)
    return indexed


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return math.nan
    labels = sorted({label for pair in pairs for label in pair})
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _right in pairs)
    right_counts = Counter(right for _left, right in pairs)
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else math.nan
    return (observed - expected) / (1.0 - expected)


def summary(name: str, rows: list[dict[str, str]]) -> None:
    pairs = [(row["blind_label"], row["reference_label"]) for row in rows]
    agreements = sum(left == right for left, right in pairs)
    rate = agreements / len(rows) if rows else math.nan
    kappa = cohen_kappa(pairs)
    print(
        f"{name}: n={len(rows)} agreement={agreements}/{len(rows)} "
        f"({100.0 * rate:.2f}%) kappa={kappa:.4f}"
        if rows
        else f"{name}: n=0 agreement=n/a kappa=n/a"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a sealed blind-review hash, then compare it with an "
            "independent human/reference list."
        )
    )
    parser.add_argument("--sealed", required=True, help="Sealed blind CSV")
    parser.add_argument("--reference", required=True, help="Post-seal reference CSV")
    parser.add_argument("--out", required=True, help="Detailed comparison CSV")
    parser.add_argument(
        "--sha256",
        help="SHA-256 sidecar (default: SEALED.csv.sha256)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the comparison output"
    )
    args = parser.parse_args()

    sealed_path = Path(args.sealed).expanduser()
    sidecar = (
        Path(args.sha256).expanduser()
        if args.sha256
        else Path(str(sealed_path) + ".sha256")
    )
    if not sidecar.is_file():
        raise FileNotFoundError(f"sealed-review SHA-256 sidecar not found: {sidecar}")
    expected = sidecar.read_text().strip().split()[0]
    actual = sha256_file(sealed_path)
    if expected != actual:
        raise ValueError(
            f"sealed review hash mismatch: expected {expected}, calculated {actual}"
        )
    print(f"Verified sealed SHA-256: {actual}")

    sealed_fields, sealed_rows = read_dict_csv(sealed_path)
    required = {
        "blind_id",
        "path",
        "validity",
        "confidence",
        "reason",
        "prior_seen",
        "reviewer",
    }
    if not required.issubset(sealed_fields):
        raise ValueError(
            "sealed CSV is missing required fields: "
            + ", ".join(sorted(required - set(sealed_fields)))
        )

    reference_rows = read_reference(args.reference)
    blind_by_key = index_labels(sealed_rows, str(sealed_path))
    reference_by_key = index_labels(reference_rows, str(args.reference))
    if set(blind_by_key) != set(reference_by_key):
        missing = sorted(set(blind_by_key) - set(reference_by_key))
        extra = sorted(set(reference_by_key) - set(blind_by_key))
        raise ValueError(
            f"coverage mismatch: reference missing={len(missing)} extra={len(extra)}; "
            f"missing sample={missing[:3]} extra sample={extra[:3]}"
        )

    comparison: list[dict[str, str]] = []
    for row in sealed_rows:
        key = stable_mode_key(row["path"])
        blind_label, _blind_source = blind_by_key[key]
        reference_label, _reference_source = reference_by_key[key]
        prior_seen = parse_bool(row["prior_seen"], field=f"{key}: prior_seen")
        comparison.append(
            {
                "blind_id": row["blind_id"],
                "path": row["path"],
                "blind_label": blind_label,
                "reference_label": reference_label,
                "agreement": str(blind_label == reference_label).lower(),
                "confidence": row["confidence"],
                "reason": row["reason"],
                "prior_seen": str(prior_seen).lower(),
            }
        )

    output = Path(args.out).expanduser()
    ensure_new_targets([output], args.force)
    atomic_write_csv(output, OUTPUT_FIELDS, comparison)

    clean = [row for row in comparison if row["prior_seen"] == "false"]
    summary("overall", comparison)
    summary("clean", clean)
    directions = Counter(
        f"{row['blind_label']}->{row['reference_label']}"
        for row in comparison
        if row["blind_label"] != row["reference_label"]
    )
    if directions:
        print(
            "Disagreements: "
            + ", ".join(f"{key}={value}" for key, value in sorted(directions.items()))
        )
    else:
        print("Disagreements: none")
    print(f"Wrote comparison: {output}")


if __name__ == "__main__":
    main()
