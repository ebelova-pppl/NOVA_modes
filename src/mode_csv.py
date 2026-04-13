from __future__ import annotations

import csv
from collections.abc import Collection

from path_utils import resolve_mode_csv_path

PATH_HEADER_NAMES = frozenset({"path", "filepath", "mode_path"})
LABEL_HEADER_NAMES = frozenset(
    {"label", "class", "target", "manual_label", "rf_label"}
)
LABEL_VALUE_NAMES = frozenset({"good", "bad", "g", "b", "skip", "s"})


def _read_nonempty_rows(csv_path: str) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(csv_path, "r", newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            first = row[0].strip()
            if first.startswith("#"):
                continue
            rows.append(row)
    return rows


def _infer_label_idx(data_rows: list[list[str]], path_idx: int, sample_size: int = 20) -> int | None:
    sample = data_rows[:sample_size]
    max_width = max((len(row) for row in sample), default=0)

    for col_idx in range(max_width):
        if col_idx == path_idx:
            continue

        values = [
            row[col_idx].strip().lower()
            for row in sample
            if col_idx < len(row) and row[col_idx].strip()
        ]
        if values and all(value in LABEL_VALUE_NAMES for value in values):
            return col_idx

    return None


def read_mode_csv_entries(
    csv_path: str,
    *,
    resolve_paths: bool = True,
    path_header_names: Collection[str] = PATH_HEADER_NAMES,
    label_header_names: Collection[str] = LABEL_HEADER_NAMES,
) -> list[tuple[str, str | None]]:
    """
    Read a NOVA mode-list CSV with an optional header row.

    Supported path headers: ``path``, ``filepath``, ``mode_path``.
    Supported label headers: ``label``, ``class``, ``target``,
    ``manual_label``, ``rf_label``.

    If no header is present, the first column is treated as the path and the
    second column, when present, is treated as the label.
    """
    raw_rows = _read_nonempty_rows(csv_path)
    if not raw_rows:
        return []

    first_row = [cell.strip().lower() for cell in raw_rows[0]]
    path_idx = next(
        (idx for idx, cell in enumerate(first_row) if cell in path_header_names),
        None,
    )
    label_idx = next(
        (idx for idx, cell in enumerate(first_row) if cell in label_header_names),
        None,
    )

    data_rows = raw_rows[1:] if path_idx is not None else raw_rows
    if path_idx is None:
        path_idx = 0
        label_idx = 1 if len(raw_rows[0]) >= 2 else None
    elif label_idx is None:
        label_idx = _infer_label_idx(data_rows, path_idx)

    entries: list[tuple[str, str | None]] = []
    for row in data_rows:
        if path_idx >= len(row):
            continue

        raw_path = row[path_idx].strip()
        if not raw_path:
            continue

        path = resolve_mode_csv_path(raw_path) if resolve_paths else raw_path
        label = row[label_idx].strip() if label_idx is not None and label_idx < len(row) else None
        entries.append((path, label))

    return entries


def read_mode_paths_csv(csv_path: str) -> list[str]:
    return [path for path, _ in read_mode_csv_entries(csv_path)]
