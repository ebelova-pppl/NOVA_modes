#!/usr/bin/env python3
"""Shared deterministic I/O helpers for rule-based NOVA shot sorting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


RULE_OUTPUT_FIELDS = [
    "path",
    "mode_key",
    "shot",
    "n",
    "ntor",
    "nr",
    "nhar",
    "omega",
    "gamma_d",
    "rad_loc",
    "rad_width",
    "input_fingerprint",
    "processing_status",
    "gap_region",
    "signed_delta",
    "fraction_below_upper2",
    "preprocessing_primary_reason",
    "rule_decision",
    "rule_primary_reason",
    "rule_triggered_rules",
    "rule_version",
    "rule_features",
    "rule_survivor_policy",
    "rule_survivor_accepted",
    "manual_decision",
    "manual_reason",
    "reviewer",
    "adjudication_timestamp",
    "override_status",
    "override_message",
    "final_decision",
    "decision_source",
    "duplicate_rank_score",
    "duplicate_rank_source",
    "selected_final",
    "diagnostic_message",
]

MANUAL_OVERRIDE_FIELDS = [
    "mode_key",
    "path",
    "input_fingerprint",
    "ntor",
    "frequency",
    "original_rule_decision",
    "manual_decision",
    "manual_reason",
    "reviewer",
    "adjudication_timestamp",
]

ALLOWED_FINAL_DECISIONS = {"GOOD", "BAD", "REVIEW"}


def empty_rule_row() -> dict[str, Any]:
    """Return one blank row with the canonical deterministic output schema."""
    return {field: "" for field in RULE_OUTPUT_FIELDS}


def portable_mode_key(raw_path: str | Path) -> str:
    """Return the portable ``shot/N#/filename`` suffix for a mode path."""
    normalized = str(raw_path).strip().replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("", "/")]
    if len(parts) < 3 or not re.fullmatch(r"N\d+", parts[-2], flags=re.IGNORECASE):
        raise ValueError(f"mode path must end in shot/N#/filename: {raw_path!r}")
    return "/".join(parts[-3:])


def datcon_path_for_mode(mode_path: str | Path, ntor: int | None = None) -> Path:
    """Return the expected continuum path beside a NOVA mode file."""
    mode = Path(mode_path).expanduser()
    if ntor is None:
        match = re.fullmatch(r"N(\d+)", mode.parent.name, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"cannot infer ntor from mode path: {mode}")
        ntor = int(match.group(1))
    return mode.parent / f"datcon{int(ntor)}"


def sha256_file(path: str | Path) -> str:
    """Calculate a file SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(mode_path: str | Path, datcon_path: str | Path) -> str:
    """Fingerprint the exact mode and corresponding continuum file contents."""
    digest = hashlib.sha256()
    digest.update(b"NOVA_MODE_INPUT_V1\0")
    for label, raw_path in ((b"mode\0", mode_path), (b"datcon\0", datcon_path)):
        path = Path(raw_path).expanduser()
        size = path.stat().st_size
        digest.update(label)
        digest.update(int(size).to_bytes(16, byteorder="big", signed=False))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def stable_json(value: Any) -> str:
    """Serialize arrays/objects with stable keys and standards-compliant nulls."""
    return json.dumps(
        _json_compatible(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def read_dict_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a required-header CSV with strict row widths."""
    source = Path(path).expanduser()
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"CSV is empty: {source}")
    fields = [field.strip() for field in rows[0]]
    if not all(fields) or len(fields) != len(set(fields)):
        raise ValueError(f"CSV has blank or duplicate headers: {source}")
    parsed: list[dict[str, str]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(fields):
            raise ValueError(
                f"{source}:{line_number}: expected {len(fields)} columns, "
                f"found {len(row)}"
            )
        parsed.append({field: value.strip() for field, value in zip(fields, row)})
    return fields, parsed


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return stable_json(value)
    if isinstance(value, Real) and not isinstance(value, Integral):
        number = float(value)
        return number if math.isfinite(number) else ""
    return value


def write_dict_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write a header and deterministic line endings."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_value(row.get(field, "")) for field in fieldnames}
                )
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_vertical_summary(
    path: str | Path, fieldnames: Sequence[str], row: Mapping[str, Any]
) -> None:
    """Atomically write the sorter-compatible two-column vertical summary."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for field in fieldnames:
                writer.writerow([field, _csv_value(row.get(field, ""))])
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_text(path: str | Path, content: str) -> None:
    """Atomically write deterministic UTF-8 text."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def rule_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sort by shot, ntor, frequency, then portable filename."""
    ntor = row.get("ntor") or row.get("n")
    omega = row.get("omega")
    try:
        ntor_key = int(ntor)
    except (TypeError, ValueError):
        ntor_key = math.inf
    try:
        omega_key = float(omega)
        if not math.isfinite(omega_key):
            omega_key = math.inf
    except (TypeError, ValueError):
        omega_key = math.inf
    path = str(row.get("path", ""))
    return (
        str(row.get("shot", "")),
        ntor_key,
        omega_key,
        Path(path).name,
        str(row.get("mode_key", path)),
    )
