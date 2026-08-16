#!/usr/bin/env python3
"""Shared deterministic I/O helpers for blind TAE-like review scripts."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


def stable_mode_key(raw_path: str) -> str:
    """Return the stable shot/N/file suffix used across NOVA data roots."""
    normalized = str(raw_path).strip().replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("", "/")]
    if len(parts) < 3:
        raise ValueError(f"mode path must contain shot/N/file: {raw_path!r}")
    return "/".join(parts[-3:])


def read_dict_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a required-header CSV and normalize header whitespace/case."""
    source = Path(path).expanduser()
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        raw_rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not raw_rows:
        raise ValueError(f"CSV is empty: {source}")

    fields = [cell.strip().lower() for cell in raw_rows[0]]
    if not all(fields) or len(fields) != len(set(fields)):
        raise ValueError(f"CSV has blank or duplicate headers: {source}")

    rows: list[dict[str, str]] = []
    for line_number, raw in enumerate(raw_rows[1:], start=2):
        if len(raw) != len(fields):
            raise ValueError(
                f"{source}:{line_number}: expected {len(fields)} columns, "
                f"found {len(raw)}"
            )
        rows.append({field: value.strip() for field, value in zip(fields, raw)})
    return fields, rows


def parse_bool(value: str, *, field: str = "value") -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("", "false", "f", "0", "no", "n"):
        return False
    if normalized in ("true", "t", "1", "yes", "y"):
        return True
    raise ValueError(f"{field} must be true or false, got {value!r}")


def normalize_validity(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {"g": "good", "b": "bad", "s": "skip"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"good", "bad", "skip"}:
        raise ValueError(f"validity must be good, bad, or skip, got {value!r}")
    return normalized


def ensure_new_targets(paths: Sequence[str | Path], force: bool) -> None:
    existing = [str(Path(path).expanduser()) for path in paths if Path(path).expanduser().exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def atomic_write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            handle.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
