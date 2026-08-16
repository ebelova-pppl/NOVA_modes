#!/usr/bin/env python3
"""Create a strictly label-free manifest for blind TAE-like mode review."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from _blind_review_io import atomic_write_csv, ensure_new_targets, stable_mode_key


PATH_FIELDS = ("path", "filepath", "mode_path")
SAFE_INPUT_FIELDS = {
    *PATH_FIELDS,
    "shot",
    "n",
    "signed_delta",
    "fraction_below_upper2",
    "gap_region",
    "error",
}
OUTPUT_FIELDS = [
    "blind_id",
    "path",
    "shot",
    "n",
    "signed_delta",
    "fraction_below_upper2",
    "gap_region",
    "error",
]
DECISION_FIELDS = [
    "blind_id",
    "path",
    "validity",
    "confidence",
    "reason",
    "prior_seen",
]


def read_label_free_source(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and any(c.strip() for c in row)]
    if not rows:
        raise ValueError(f"input CSV is empty: {path}")

    first = [cell.strip().lower() for cell in rows[0]]
    headered = any(field in first for field in PATH_FIELDS)

    if not headered:
        if any(len(row) != 1 for row in rows):
            raise ValueError(
                "headerless blind manifests must contain exactly one path column; "
                "refusing a possible labeled or prediction-bearing list"
            )
        return [{"path": row[0].strip()} for row in rows]

    if not all(first) or len(first) != len(set(first)):
        raise ValueError("input has blank or duplicate column names")
    unsupported = sorted(set(first) - SAFE_INPUT_FIELDS)
    if unsupported:
        raise ValueError(
            "input is not demonstrably label-free; unsupported/sensitive "
            f"columns: {', '.join(unsupported)}"
        )
    path_fields = [field for field in PATH_FIELDS if field in first]
    if len(path_fields) != 1:
        raise ValueError("input must contain exactly one path column")
    path_field = path_fields[0]

    parsed: list[dict[str, str]] = []
    for line_number, raw in enumerate(rows[1:], start=2):
        if len(raw) != len(first):
            raise ValueError(
                f"{path}:{line_number}: expected {len(first)} columns, "
                f"found {len(raw)}"
            )
        record = {field: value.strip() for field, value in zip(first, raw)}
        record["path"] = record.pop(path_field)
        parsed.append(record)
    return parsed


def portable_path(raw_path: str, data_root: Path | None) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or data_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path.resolve())


def path_metadata(path: str) -> tuple[str, str]:
    key = stable_mode_key(path)
    shot, n_dir, _filename = key.split("/", 2)
    match = re.fullmatch(r"N(\d+)", n_dir, flags=re.IGNORECASE)
    return shot, match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a label-free blind-review manifest. Inputs containing "
            "labels, predictions, probabilities, or unknown columns are rejected."
        )
    )
    parser.add_argument("input_csv", help="Label-free TAE-like split manifest")
    parser.add_argument("output_csv", help="Output blind manifest")
    parser.add_argument(
        "--data-root",
        help="Convert absolute paths below this data root to relative paths",
    )
    parser.add_argument(
        "--decisions-template",
        help="Optional empty decision template to create alongside the manifest",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite explicitly named outputs"
    )
    args = parser.parse_args()

    source = Path(args.input_csv).expanduser()
    output = Path(args.output_csv).expanduser()
    template = (
        Path(args.decisions_template).expanduser()
        if args.decisions_template
        else None
    )
    targets = [output] + ([template] if template else [])
    ensure_new_targets(targets, args.force)

    source_rows = read_label_free_source(source)
    if not source_rows:
        raise ValueError("input contains no mode rows")

    data_root = Path(args.data_root).expanduser() if args.data_root else None
    width = max(3, len(str(len(source_rows))))
    seen: set[str] = set()
    manifest: list[dict[str, str]] = []

    for index, source_row in enumerate(source_rows, start=1):
        raw_path = source_row.get("path", "").strip()
        if not raw_path:
            raise ValueError(f"input row {index} has an empty path")
        path = portable_path(raw_path, data_root)
        key = stable_mode_key(path)
        if key in seen:
            raise ValueError(f"duplicate stable mode path: {key}")
        seen.add(key)
        derived_shot, derived_n = path_metadata(path)
        manifest.append(
            {
                "blind_id": f"B{index:0{width}d}",
                "path": path,
                "shot": source_row.get("shot", "") or derived_shot,
                "n": source_row.get("n", "") or derived_n,
                "signed_delta": source_row.get("signed_delta", ""),
                "fraction_below_upper2": source_row.get(
                    "fraction_below_upper2", ""
                ),
                "gap_region": source_row.get("gap_region", ""),
                "error": source_row.get("error", ""),
            }
        )

    atomic_write_csv(output, OUTPUT_FIELDS, manifest)

    if template is not None:
        decision_rows = [
            {
                "blind_id": row["blind_id"],
                "path": row["path"],
                "validity": "",
                "confidence": "",
                "reason": "",
                "prior_seen": "false",
            }
            for row in manifest
        ]
        atomic_write_csv(template, DECISION_FIELDS, decision_rows)

    print(f"Prepared {len(manifest)} label-free modes: {output}")
    if template is not None:
        print(f"Created decision template: {template}")


if __name__ == "__main__":
    main()
