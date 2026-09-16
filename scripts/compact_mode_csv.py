#!/usr/bin/env python3
"""Copy a wide rules-mode CSV into the fixed, minimal sharing format.

Example: python scripts/compact_mode_csv.py /path/to/good_tae_final.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


OUTPUT_FIELDS = (
    "path", "shot", "n", "ntor", "nr", "nhar", "omega", "gamma_d",
    "rad_loc", "rad_width", "gap_region", "rule_decision",
    "overall_rule_severity", "manual_decision",
)
# Older sorter exports did not record severity; blank means unavailable.
OPTIONAL_FIELDS = frozenset({"overall_rule_severity"})


def compact_mode_csv(input_path: Path, output_path: Path) -> int:
    """Preserve selected cell text and row order; refuse to replace a file."""
    # Full rule-feature JSON can exceed the CSV parser's default field limit.
    csv.field_size_limit(2**31 - 1)
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        header = reader.fieldnames
        if not header:
            raise ValueError("Input CSV has no header.")
        missing = [
            field for field in OUTPUT_FIELDS
            if field not in header and field not in OPTIONAL_FIELDS
        ]
        if missing:
            raise ValueError("Input CSV is missing columns: " + ", ".join(missing))
        duplicated = [field for field in OUTPUT_FIELDS if header.count(field) > 1]
        if duplicated:
            raise ValueError("Input CSV repeats columns: " + ", ".join(duplicated))

        # Validate before creating the output so bad input leaves no partial CSV.
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Input CSV row ending at line {reader.line_num} has a "
                    "different number of columns from the header."
                )
            rows.append({field: row.get(field, "") for field in OUTPUT_FIELDS})

    # Exclusive creation also protects the input if it is given as the output.
    with output_path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for field in OUTPUT_FIELDS:
        if field in OPTIONAL_FIELDS and field not in header:
            print(
                f"Note: Input CSV has no {field} column; left blank in output.",
                file=sys.stderr,
            )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the 14 sharing columns from a header-bearing rules-mode CSV, "
            "preserving all rows and original cell values. Missing "
            "overall_rule_severity in older exports is left blank. "
            "Uses only the Python standard library."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Wide CSV, e.g. good_tae_final.csv")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="New output CSV (default: INPUT_STEM_minimal.csv beside the input).",
    )
    args = parser.parse_args()
    output_path = args.output or args.input_csv.with_name(
        args.input_csv.stem + "_minimal.csv"
    )
    try:
        count = compact_mode_csv(args.input_csv, output_path)
    except FileExistsError:
        parser.error(f"Output already exists: {output_path}. Choose a new output path.")
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {count} rows and {len(OUTPUT_FIELDS)} columns to {output_path}")


if __name__ == "__main__":
    main()
