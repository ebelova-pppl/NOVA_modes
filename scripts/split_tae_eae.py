#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from cont_features import warn_once_per_dir
from path_utils import resolve_mode_csv_path
from tae_eae_features import load_upper2_scalars_for_mode

PATH_HEADER_NAMES = {"path", "filepath", "mode_path"}
SUMMARY_COLUMNS = ("label", "validity", "family")
KNOWN_VALIDITY_VALUES = {"good", "bad", "skip"}
KNOWN_FAMILY_VALUES = {"tae", "eae", "none"}
EXTRA_OUTPUT_COLUMNS = [
    "signed_delta",
    "fraction_below_upper2",
    "gap_region",
    "error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Split NOVA modes into TAE-like vs EAE-like groups using the upper "
            "TAE gap boundary from datcon files."
        )
    )
    ap.add_argument("--input_csv", required=True, help="CSV list of mode files")
    ap.add_argument(
        "--out_below_csv",
        required=True,
        help="Output CSV for modes classified as below_upper2 (TAE-like)",
    )
    ap.add_argument(
        "--out_above_csv",
        required=True,
        help="Output CSV for modes classified as above_upper2 (EAE-like)",
    )
    ap.add_argument(
        "--out_all_csv",
        default=None,
        help="Full output CSV with all rows and errors (default: <input>_tae_eae_split.csv)",
    )
    ap.add_argument(
        "--signed_delta_threshold",
        type=float,
        default=0.0,
        help="Threshold for signed_delta; above this counts as below_upper2",
    )
    ap.add_argument(
        "--fraction_threshold",
        type=float,
        default=0.5,
        help="Threshold for fraction_below_upper2; above this counts as below_upper2",
    )
    return ap.parse_args()


def _normalize_header_name(name: str, fallback_idx: int) -> str:
    clean = name.strip()
    return clean if clean else f"col{fallback_idx}"


def _make_unique_header(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f"{name}_{count}")
    return unique


def _observed_values(rows: list[list[str]], idx: int) -> set[str]:
    values = set()
    for row in rows:
        if idx < len(row):
            value = row[idx].strip().lower()
            if value:
                values.add(value)
    return values


def _infer_default_header(raw_rows: list[list[str]], max_width: int) -> list[str]:
    if max_width <= 0:
        return []

    header = ["path"]
    if max_width == 1:
        return header

    second_values = _observed_values(raw_rows, 1)
    if second_values and second_values <= KNOWN_VALIDITY_VALUES:
        header.append("validity")
    else:
        header.append("col2")

    if max_width >= 3:
        third_values = _observed_values(raw_rows, 2)
        if third_values and third_values <= KNOWN_FAMILY_VALUES:
            header.append("family")
        else:
            header.append("col3")

    if max_width > 3:
        header.extend(f"col{idx}" for idx in range(4, max_width + 1))

    return header


def read_input_rows(csv_path: str) -> tuple[list[str], str, list[dict[str, str]]]:
    with open(csv_path, "r", newline="") as fp:
        reader = csv.reader(fp)
        raw_rows = []
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            first = row[0].strip()
            if first.startswith("#"):
                continue
            raw_rows.append(row)

    if not raw_rows:
        raise ValueError(f"No data rows found in {csv_path}")

    first_row = raw_rows[0]
    header_idx = next(
        (idx for idx, cell in enumerate(first_row) if cell.strip().lower() in PATH_HEADER_NAMES),
        None,
    )

    if header_idx is not None:
        header = [_normalize_header_name(name, idx + 1) for idx, name in enumerate(first_row)]
        data_rows = raw_rows[1:]
    else:
        max_width = max(len(row) for row in raw_rows)
        header = _infer_default_header(raw_rows, max_width)
        header_idx = 0
        data_rows = raw_rows

    max_width = max([len(header)] + [len(row) for row in data_rows]) if data_rows else len(header)
    if max_width > len(header):
        header = header + [f"col{idx}" for idx in range(len(header) + 1, max_width + 1)]

    header = _make_unique_header(header)
    path_column = header[header_idx]

    rows: list[dict[str, str]] = []
    for row in data_rows:
        padded = list(row) + [""] * (len(header) - len(row))
        record = {header[idx]: padded[idx].strip() for idx in range(len(header))}
        rows.append(record)

    return header, path_column, rows


def classify_gap_region(
    signed_delta: float,
    fraction_below_upper2: float,
    *,
    signed_delta_threshold: float,
    fraction_threshold: float,
) -> str:
    if signed_delta > signed_delta_threshold and fraction_below_upper2 > fraction_threshold:
        return "below_upper2"
    return "above_upper2"


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.16g}"


def build_output_row(
    input_row: dict[str, str],
    *,
    signed_delta: float | None,
    fraction_below_upper2: float | None,
    gap_region: str,
    error: str = "",
) -> dict[str, str]:
    row = dict(input_row)
    row["signed_delta"] = fmt_float(signed_delta)
    row["fraction_below_upper2"] = fmt_float(fraction_below_upper2)
    row["gap_region"] = gap_region
    row["error"] = error
    return row


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_group_summary(group_name: str, rows: list[dict[str, str]], header: list[str]) -> None:
    if not rows:
        return

    for column in SUMMARY_COLUMNS:
        if column not in header:
            continue
        counts = Counter((row.get(column, "") or "<blank>") for row in rows)
        print(f"{group_name} by {column}:")
        for value, count in sorted(counts.items()):
            print(f"  {value}: {count}")


def format_named_counts(rows: list[dict[str, str]], column: str, order: list[str]) -> str:
    counts = Counter((row.get(column, "") or "<blank>") for row in rows)
    return ", ".join(f"{counts.get(name, 0)} {name}" for name in order)


def print_label_column_check(
    below_rows: list[dict[str, str]],
    above_rows: list[dict[str, str]],
    header: list[str],
    below_name: str,
    above_name: str,
) -> None:
    if "family" not in header:
        return

    print("Check using label columns:")
    print(f"  {below_name}: {format_named_counts(below_rows, 'family', ['tae', 'none', 'eae'])}")
    print(f"  {above_name}: {format_named_counts(above_rows, 'family', ['eae', 'tae', 'none'])}")


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    out_below_csv = Path(args.out_below_csv)
    out_above_csv = Path(args.out_above_csv)
    out_all_csv = (
        Path(args.out_all_csv)
        if args.out_all_csv
        else input_csv.with_name(f"{input_csv.stem}_tae_eae_split.csv")
    )

    header, path_column, input_rows = read_input_rows(str(input_csv))
    output_header = header + [name for name in EXTRA_OUTPUT_COLUMNS if name not in header]

    all_rows: list[dict[str, str]] = []
    below_rows: list[dict[str, str]] = []
    above_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []

    for input_row in input_rows:
        raw_path = input_row.get(path_column, "").strip()
        if not raw_path:
            row = build_output_row(
                input_row,
                signed_delta=None,
                fraction_below_upper2=None,
                gap_region="error",
                error="Empty mode path",
            )
            all_rows.append(row)
            error_rows.append(row)
            continue

        try:
            mode_path = resolve_mode_csv_path(raw_path)
        except Exception as exc:
            row = build_output_row(
                input_row,
                signed_delta=None,
                fraction_below_upper2=None,
                gap_region="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            all_rows.append(row)
            error_rows.append(row)
            continue

        try:
            if not Path(mode_path).is_file():
                raise FileNotFoundError(f"Missing mode file: {mode_path}")
            scalars = load_upper2_scalars_for_mode(mode_path)
            gap_region = classify_gap_region(
                scalars["signed_delta"],
                scalars["fraction_below_upper2"],
                signed_delta_threshold=args.signed_delta_threshold,
                fraction_threshold=args.fraction_threshold,
            )
            row = build_output_row(
                input_row,
                signed_delta=scalars["signed_delta"],
                fraction_below_upper2=scalars["fraction_below_upper2"],
                gap_region=gap_region,
            )
            all_rows.append(row)
            if gap_region == "below_upper2":
                below_rows.append(row)
            else:
                above_rows.append(row)
        except FileNotFoundError as exc:
            if str(exc).startswith("Missing mode file:"):
                row = build_output_row(
                    input_row,
                    signed_delta=None,
                    fraction_below_upper2=None,
                    gap_region="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                all_rows.append(row)
                error_rows.append(row)
                continue

            warn_once_per_dir(
                mode_path,
                f"   \n"
                f"========================================================================\n"
                f"[NOVA-SPLIT] Continuum file not found in directory:\n"
                f"  {Path(mode_path).resolve().parent}\n"
                f"Rows from this directory will be written with gap_region=error and\n"
                f"excluded from the below/above output CSVs.\n"
                f"========================================================================"
            )
            row = build_output_row(
                input_row,
                signed_delta=None,
                fraction_below_upper2=None,
                gap_region="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            all_rows.append(row)
            error_rows.append(row)
        except Exception as exc:
            warn_once_per_dir(
                mode_path,
                f"[NOVA-SPLIT] Failed to compute upper-gap scalars in directory:\n"
                f"  {Path(mode_path).resolve().parent}\n"
                f"Rows from this directory will be written with gap_region=error and\n"
                f"excluded from the below/above output CSVs.\n"
                f"Reason: {type(exc).__name__}: {exc}"
            )
            row = build_output_row(
                input_row,
                signed_delta=None,
                fraction_below_upper2=None,
                gap_region="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            all_rows.append(row)
            error_rows.append(row)

    write_rows_csv(out_below_csv, output_header, below_rows)
    write_rows_csv(out_above_csv, output_header, above_rows)
    write_rows_csv(out_all_csv, output_header, all_rows)

    print(f"Processed rows: {len(input_rows)}")
    print(f"Below upper2 (TAE-like): {len(below_rows)}")
    print(f"Above upper2 (EAE-like): {len(above_rows)}")
    print(f"Skipped/errors: {len(error_rows)}")
    print(f"Wrote below-upper2 CSV: {out_below_csv}")
    print(f"Wrote above-upper2 CSV: {out_above_csv}")
    print(f"Wrote full CSV:         {out_all_csv}")

    print_label_column_check(
        below_rows,
        above_rows,
        header,
        out_below_csv.name,
        out_above_csv.name,
    )
    print_group_summary("Below upper2", below_rows, header)
    print_group_summary("Above upper2", above_rows, header)
    print_group_summary("Errors", error_rows, header)


if __name__ == "__main__":
    main()
