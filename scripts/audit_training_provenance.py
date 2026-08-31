#!/usr/bin/env python3
"""Audit NOVA training-shot files against an authoritative reference tree.

The audit is read-only with respect to both input trees. It compares the
training-relevant payload (``egn*``, ``datconN``, optional ``datcon_gf.txt``,
and preserved ``datconN_old`` backups) and writes versioned, machine-readable
artifacts to an explicit output directory.

Example:
    python scripts/audit_training_provenance.py \
      --training-root "$NOVA_DATA" \
      --reference-root "$NOVA_DITW_ROOT" \
      --train-csv "$NOVA_TRAIN_CSV" \
      --out-dir audits/training_provenance/2026-08-20_flux_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from _repo_bootstrap import ensure_repo_src_on_path


REPO_ROOT = ensure_repo_src_on_path()

from nova_mode_loader import load_mode_from_nova  # noqa: E402


SCHEMA_VERSION = "nova-training-provenance-v1"
OUTPUT_FILENAMES = (
    "file_manifest.csv",
    "differences.csv",
    "shot_summary.csv",
    "report.md",
    "run_metadata.json",
    "SHA256SUMS",
)
PATH_HEADERS = ("path", "filepath", "mode_path")
LABEL_HEADERS = ("validity", "label", "class", "target", "manual_label", "rf_label")
N_DIR_RE = re.compile(r"^N([1-9][0-9]*)$")

MANIFEST_FIELDS = (
    "shot",
    "ntor",
    "file_kind",
    "filename",
    "training_relative_path",
    "reference_relative_path",
    "in_training_csv",
    "training_label",
    "status",
    "training_exists",
    "reference_exists",
    "training_size",
    "reference_size",
    "training_mtime_utc",
    "reference_mtime_utc",
    "training_sha256",
    "reference_sha256",
    "mode_shape_training",
    "mode_shape_reference",
    "mode_structure_equal",
    "mode_max_abs_delta",
    "omega_training",
    "omega_reference",
    "omega_relative_change_signed",
    "omega_relative_change_abs",
    "gamma_training",
    "gamma_reference",
    "ntor_training",
    "ntor_reference",
    "comparison_error",
)

SUMMARY_FIELDS = (
    "shot",
    "canonical_rows",
    "canonical_identical",
    "canonical_different",
    "canonical_missing_training",
    "canonical_missing_reference",
    "canonical_missing_both",
    "mode_training_count",
    "mode_reference_count",
    "mode_identical",
    "mode_different",
    "mode_training_only",
    "mode_reference_only",
    "mode_missing_both",
    "datcon_identical",
    "datcon_different",
    "datcon_training_only",
    "datcon_reference_only",
    "datcon_backup_count",
    "datcon_backup_identical_to_reference",
    "datcon_backup_different_from_reference",
    "datcon_gf_identical",
    "datcon_gf_different",
    "datcon_gf_training_only",
    "datcon_gf_reference_only",
    "overall_status",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _normalise_relative_path(raw_path: str) -> str:
    value = raw_path.strip().replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"training CSV contains unsafe relative path: {raw_path!r}")
    if len(path.parts) < 3:
        raise ValueError(
            "training CSV paths must have shot/N#/filename form, got "
            f"{raw_path!r}"
        )
    if not N_DIR_RE.match(path.parts[1]):
        raise ValueError(f"training CSV path has invalid N directory: {raw_path!r}")
    return path.as_posix()


def read_training_rows(train_csv: Path) -> dict[str, str]:
    """Return a unique relative-mode-path to label mapping."""
    with train_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers = tuple(reader.fieldnames or ())
        path_header = next((name for name in PATH_HEADERS if name in headers), None)
        label_header = next((name for name in LABEL_HEADERS if name in headers), None)
        if path_header is None:
            raise ValueError(
                f"{train_csv} has no recognized path column; expected one of {PATH_HEADERS}"
            )
        rows: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            raw_path = row.get(path_header, "")
            if not raw_path or raw_path.lstrip().startswith("#"):
                continue
            relative_path = _normalise_relative_path(raw_path)
            if relative_path in rows:
                raise ValueError(
                    f"{train_csv}:{line_number}: duplicate mode path {relative_path}"
                )
            rows[relative_path] = (row.get(label_header, "") if label_header else "").strip()
    if not rows:
        raise ValueError(f"{train_csv} contains no training rows")
    return rows


class FileInfoCache:
    """Cache file metadata so hashes are computed once per audit run."""

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, Any]] = {}

    def get(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        resolved = path.resolve()
        if resolved not in self._cache:
            stat = path.stat()
            self._cache[resolved] = {
                "size": stat.st_size,
                "mtime_utc": _mtime_utc(path),
                "sha256": sha256_file(path),
            }
        return self._cache[resolved]


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(value) for value in shape)


def _mode_comparison(training_path: Path, reference_path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {
        "mode_shape_training": "",
        "mode_shape_reference": "",
        "mode_structure_equal": "",
        "mode_max_abs_delta": "",
        "omega_training": "",
        "omega_reference": "",
        "omega_relative_change_signed": "",
        "omega_relative_change_abs": "",
        "gamma_training": "",
        "gamma_reference": "",
        "ntor_training": "",
        "ntor_reference": "",
        "comparison_error": "",
    }
    try:
        old_mode, old_omega, old_gamma, old_ntor = load_mode_from_nova(training_path)
        new_mode, new_omega, new_gamma, new_ntor = load_mode_from_nova(reference_path)
        values.update(
            {
                "mode_shape_training": _shape_text(old_mode.shape),
                "mode_shape_reference": _shape_text(new_mode.shape),
                "omega_training": old_omega,
                "omega_reference": new_omega,
                "gamma_training": old_gamma,
                "gamma_reference": new_gamma,
                "ntor_training": old_ntor,
                "ntor_reference": new_ntor,
            }
        )
        if old_omega != 0.0:
            signed = (new_omega - old_omega) / abs(old_omega)
            values["omega_relative_change_signed"] = signed
            values["omega_relative_change_abs"] = abs(signed)
        if old_mode.shape == new_mode.shape:
            equal = bool(np.array_equal(old_mode, new_mode))
            values["mode_structure_equal"] = str(equal).lower()
            values["mode_max_abs_delta"] = float(
                np.max(np.abs(old_mode - new_mode), initial=0.0)
            )
        else:
            values["mode_structure_equal"] = "false"
    except Exception as exc:  # retain a complete audit even for malformed modes
        values["comparison_error"] = f"{type(exc).__name__}: {exc}"
    return values


def _empty_mode_comparison() -> dict[str, Any]:
    return {
        "mode_shape_training": "",
        "mode_shape_reference": "",
        "mode_structure_equal": "",
        "mode_max_abs_delta": "",
        "omega_training": "",
        "omega_reference": "",
        "omega_relative_change_signed": "",
        "omega_relative_change_abs": "",
        "gamma_training": "",
        "gamma_reference": "",
        "ntor_training": "",
        "ntor_reference": "",
        "comparison_error": "",
    }


def _paired_file_row(
    *,
    shot: str,
    ntor: int,
    file_kind: str,
    filename: str,
    training_relative_path: str,
    reference_relative_path: str,
    training_root: Path,
    reference_root: Path,
    training_rows: Mapping[str, str],
    cache: FileInfoCache,
) -> dict[str, Any]:
    training_path = training_root / training_relative_path
    reference_path = reference_root / reference_relative_path
    training_info = cache.get(training_path)
    reference_info = cache.get(reference_path)

    if training_info is not None and reference_info is not None:
        status = (
            "identical"
            if training_info["sha256"] == reference_info["sha256"]
            else "different"
        )
    elif training_info is not None:
        status = "training_only"
    elif reference_info is not None:
        status = "reference_only"
    else:
        status = "missing_both"

    canonical_label = training_rows.get(training_relative_path, "")
    row: dict[str, Any] = {
        "shot": shot,
        "ntor": ntor,
        "file_kind": file_kind,
        "filename": filename,
        "training_relative_path": training_relative_path,
        "reference_relative_path": reference_relative_path,
        "in_training_csv": str(training_relative_path in training_rows).lower(),
        "training_label": canonical_label,
        "status": status,
        "training_exists": str(training_info is not None).lower(),
        "reference_exists": str(reference_info is not None).lower(),
        "training_size": training_info["size"] if training_info else "",
        "reference_size": reference_info["size"] if reference_info else "",
        "training_mtime_utc": training_info["mtime_utc"] if training_info else "",
        "reference_mtime_utc": reference_info["mtime_utc"] if reference_info else "",
        "training_sha256": training_info["sha256"] if training_info else "",
        "reference_sha256": reference_info["sha256"] if reference_info else "",
    }
    row.update(_empty_mode_comparison())
    if file_kind == "mode" and status == "different":
        row.update(_mode_comparison(training_path, reference_path))
    return row


def _n_directories(shot_dir: Path) -> set[int]:
    if not shot_dir.is_dir():
        return set()
    result = set()
    for path in shot_dir.iterdir():
        match = N_DIR_RE.match(path.name)
        if path.is_dir() and match:
            result.add(int(match.group(1)))
    return result


def _mode_names(n_dir: Path) -> set[str]:
    if not n_dir.is_dir():
        return set()
    return {path.name for path in n_dir.glob("egn*") if path.is_file()}


def build_manifest(
    training_root: Path,
    reference_root: Path,
    training_rows: Mapping[str, str],
    *,
    include_datcon_gf: bool = True,
) -> list[dict[str, Any]]:
    """Build a deterministic per-file comparison manifest."""
    paths_by_shot_n: dict[tuple[str, int], set[str]] = defaultdict(set)
    for relative_path in training_rows:
        parts = PurePosixPath(relative_path).parts
        shot = parts[0]
        ntor = int(parts[1][1:])
        paths_by_shot_n[(shot, ntor)].add(parts[-1])

    shots = sorted({shot for shot, _ in paths_by_shot_n})
    cache = FileInfoCache()
    rows: list[dict[str, Any]] = []
    for shot in shots:
        training_shot = training_root / shot
        reference_shot = reference_root / shot
        n_values = _n_directories(training_shot) | _n_directories(reference_shot)
        n_values |= {ntor for row_shot, ntor in paths_by_shot_n if row_shot == shot}
        for ntor in sorted(n_values):
            n_name = f"N{ntor}"
            training_n = training_shot / n_name
            reference_n = reference_shot / n_name
            mode_names = _mode_names(training_n) | _mode_names(reference_n)
            mode_names |= paths_by_shot_n.get((shot, ntor), set())
            for filename in sorted(mode_names):
                relative_path = f"{shot}/{n_name}/{filename}"
                rows.append(
                    _paired_file_row(
                        shot=shot,
                        ntor=ntor,
                        file_kind="mode",
                        filename=filename,
                        training_relative_path=relative_path,
                        reference_relative_path=relative_path,
                        training_root=training_root,
                        reference_root=reference_root,
                        training_rows=training_rows,
                        cache=cache,
                    )
                )

            datcon_name = f"datcon{ntor}"
            datcon_relative = f"{shot}/{n_name}/{datcon_name}"
            if (training_root / datcon_relative).is_file() or (
                reference_root / datcon_relative
            ).is_file():
                rows.append(
                    _paired_file_row(
                        shot=shot,
                        ntor=ntor,
                        file_kind="datcon",
                        filename=datcon_name,
                        training_relative_path=datcon_relative,
                        reference_relative_path=datcon_relative,
                        training_root=training_root,
                        reference_root=reference_root,
                        training_rows=training_rows,
                        cache=cache,
                    )
                )

            backup_name = f"{datcon_name}_old"
            backup_relative = f"{shot}/{n_name}/{backup_name}"
            if (training_root / backup_relative).is_file():
                rows.append(
                    _paired_file_row(
                        shot=shot,
                        ntor=ntor,
                        file_kind="datcon_backup",
                        filename=backup_name,
                        training_relative_path=backup_relative,
                        reference_relative_path=datcon_relative,
                        training_root=training_root,
                        reference_root=reference_root,
                        training_rows=training_rows,
                        cache=cache,
                    )
                )

            if include_datcon_gf:
                gf_relative = f"{shot}/{n_name}/datcon_gf.txt"
                if (training_root / gf_relative).is_file() or (
                    reference_root / gf_relative
                ).is_file():
                    rows.append(
                        _paired_file_row(
                            shot=shot,
                            ntor=ntor,
                            file_kind="datcon_gf",
                            filename="datcon_gf.txt",
                            training_relative_path=gf_relative,
                            reference_relative_path=gf_relative,
                            training_root=training_root,
                            reference_root=reference_root,
                            training_rows=training_rows,
                            cache=cache,
                        )
                    )
    return rows


def build_shot_summary(
    manifest: Sequence[Mapping[str, Any]],
    training_rows: Mapping[str, str],
) -> list[dict[str, Any]]:
    shots = sorted({PurePosixPath(path).parts[0] for path in training_rows})
    rows_by_shot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest:
        rows_by_shot[str(row["shot"])].append(row)

    summaries = []
    for shot in shots:
        shot_rows = rows_by_shot[shot]
        summary: dict[str, Any] = {field: 0 for field in SUMMARY_FIELDS}
        summary["shot"] = shot
        canonical_modes = [
            row
            for row in shot_rows
            if row["file_kind"] == "mode" and row["in_training_csv"] == "true"
        ]
        summary["canonical_rows"] = len(canonical_modes)
        for row in canonical_modes:
            status = str(row["status"])
            if status == "identical":
                summary["canonical_identical"] += 1
            elif status == "different":
                summary["canonical_different"] += 1
            elif status == "reference_only":
                summary["canonical_missing_training"] += 1
            elif status == "training_only":
                summary["canonical_missing_reference"] += 1
            elif status == "missing_both":
                summary["canonical_missing_both"] += 1

        mode_rows = [row for row in shot_rows if row["file_kind"] == "mode"]
        summary["mode_training_count"] = sum(
            row["training_exists"] == "true" for row in mode_rows
        )
        summary["mode_reference_count"] = sum(
            row["reference_exists"] == "true" for row in mode_rows
        )
        for row in mode_rows:
            summary[f"mode_{row['status']}"] += 1

        for kind, prefix in (
            ("datcon", "datcon"),
            ("datcon_gf", "datcon_gf"),
        ):
            for row in shot_rows:
                if row["file_kind"] == kind:
                    summary[f"{prefix}_{row['status']}"] += 1

        backup_rows = [row for row in shot_rows if row["file_kind"] == "datcon_backup"]
        summary["datcon_backup_count"] = len(backup_rows)
        summary["datcon_backup_identical_to_reference"] = sum(
            row["status"] == "identical" for row in backup_rows
        )
        summary["datcon_backup_different_from_reference"] = sum(
            row["status"] == "different" for row in backup_rows
        )

        has_mode_mismatch = any(
            summary[name]
            for name in (
                "mode_different",
                "mode_training_only",
                "mode_reference_only",
                "mode_missing_both",
            )
        )
        has_datcon_mismatch = any(
            summary[name]
            for name in (
                "datcon_different",
                "datcon_training_only",
                "datcon_reference_only",
            )
        )
        if has_mode_mismatch and has_datcon_mismatch:
            summary["overall_status"] = "mode_and_continuum_mismatch"
        elif has_mode_mismatch:
            summary["overall_status"] = "mode_mismatch"
        elif has_datcon_mismatch:
            summary["overall_status"] = "continuum_mismatch"
        else:
            summary["overall_status"] = "active_payload_aligned"
        summaries.append(summary)
    return summaries


def _markdown_table(summary: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Shot | Canonical rows (same/different/missing reference) | "
        "All modes (same/different/training-only/reference-only) | "
        "Active datcon (same/different/training-only/reference-only) | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        lines.append(
            "| `{shot}` | {canonical_identical}/{canonical_different}/"
            "{canonical_missing_reference} (+{canonical_missing_both} missing both) | "
            "{mode_identical}/{mode_different}/{mode_training_only}/"
            "{mode_reference_only} (+{mode_missing_both} missing both) | {datcon_identical}/"
            "{datcon_different}/{datcon_training_only}/{datcon_reference_only} | "
            "`{overall_status}` |".format(**row)
        )
    return lines


def render_report(
    *,
    audit_id: str,
    generated_at: str,
    training_root: Path,
    reference_root: Path,
    train_csv: Path,
    train_csv_sha256: str,
    audit_script_sha256: str,
    manifest: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    include_datcon_gf: bool,
) -> str:
    changed_canonical = [
        row
        for row in manifest
        if row["file_kind"] == "mode"
        and row["in_training_csv"] == "true"
        and row["status"] == "different"
    ]
    missing_canonical = [
        row
        for row in manifest
        if row["file_kind"] == "mode"
        and row["in_training_csv"] == "true"
        and row["status"] == "training_only"
    ]
    backup_rows = [row for row in manifest if row["file_kind"] == "datcon_backup"]
    changed_by_shot = Counter(str(row["shot"]) for row in changed_canonical)
    missing_by_shot = Counter(str(row["shot"]) for row in missing_canonical)
    changed_datcon_by_shot = {
        str(row["shot"]): int(row["datcon_different"])
        for row in summary
        if int(row["datcon_different"])
    }

    lines = [
        f"# NOVA training provenance audit: {audit_id}",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Generated: `{generated_at}`",
        f"- Training root: `{training_root}`",
        f"- Reference root: `{reference_root}`",
        f"- Training CSV: `{train_csv}`",
        f"- Training CSV SHA-256: `{train_csv_sha256}`",
        f"- Audit script SHA-256: `{audit_script_sha256}`",
        f"- Shots: {len(summary)}",
        f"- Canonical rows: {sum(int(row['canonical_rows']) for row in summary)}",
        "",
        "## Scope",
        "",
        "This is a read-only, byte-level comparison of the training-relevant "
        "payload for the shots named by the training CSV: all `egn*` mode files, "
        "active `datconN` files, preserved `datconN_old` backups, and "
        + ("`datcon_gf.txt`." if include_datcon_gf else "no auxiliary datcon files."),
        "Run executables, plots, logs, and other NOVA working-directory artifacts "
        "are intentionally outside the manifest.",
        "",
        "## Shot summary",
        "",
        *_markdown_table(summary),
        "",
        "## Most important current differences",
        "",
    ]
    if changed_by_shot:
        lines.append(
            "Canonical same-name mode files differ in: "
            + ", ".join(f"`{shot}` ({count})" for shot, count in sorted(changed_by_shot.items()))
            + "."
        )
    else:
        lines.append("No canonical same-name mode file differs from the reference copy.")
    if missing_by_shot:
        lines.append(
            "Canonical training modes absent from the reference tree: "
            + ", ".join(f"`{shot}` ({count})" for shot, count in sorted(missing_by_shot.items()))
            + "."
        )
    else:
        lines.append("Every canonical training mode is present in the reference tree.")
    if changed_datcon_by_shot:
        lines.append(
            "Active `datconN` mismatches: "
            + ", ".join(
                f"`{shot}` ({count})" for shot, count in sorted(changed_datcon_by_shot.items())
            )
            + "."
        )
    else:
        lines.append("All active `datconN` files match their reference copies.")
    if backup_rows:
        backup_counts = Counter(str(row["shot"]) for row in backup_rows)
        different_backup_counts = Counter(
            str(row["shot"]) for row in backup_rows if row["status"] == "different"
        )
        lines.append(
            "Preserved `datconN_old` files: "
            + ", ".join(
                f"`{shot}` ({backup_counts[shot]} total, "
                f"{different_backup_counts[shot]} different from active reference)"
                for shot in sorted(backup_counts)
            )
            + "."
        )
    if changed_canonical:
        lines.extend(
            [
                "",
                "## Changed canonical mode diagnostics",
                "",
                "| Shot | Changed modes | Structure unequal | Shape changed | "
                "Median `abs(delta omega / omega)` | Maximum | Parse errors |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        modes_by_shot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in changed_canonical:
            modes_by_shot[str(row["shot"])].append(row)
        for shot in sorted(modes_by_shot):
            mode_rows = modes_by_shot[shot]
            relative_changes = np.asarray(
                [
                    float(row["omega_relative_change_abs"])
                    for row in mode_rows
                    if row["omega_relative_change_abs"] != ""
                ],
                dtype=float,
            )
            structure_unequal = sum(
                row["mode_structure_equal"] == "false" for row in mode_rows
            )
            shape_changed = sum(
                row["mode_shape_training"] != row["mode_shape_reference"]
                for row in mode_rows
            )
            parse_errors = sum(bool(row["comparison_error"]) for row in mode_rows)
            median_change = (
                f"{float(np.median(relative_changes)):.6g}"
                if relative_changes.size
                else "n/a"
            )
            maximum_change = (
                f"{float(np.max(relative_changes)):.6g}"
                if relative_changes.size
                else "n/a"
            )
            lines.append(
                f"| `{shot}` | {len(mode_rows)} | {structure_unequal} | "
                f"{shape_changed} | {median_change} | {maximum_change} | "
                f"{parse_errors} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `identical` means equal SHA-256 content, not merely equal filename, size, or timestamp.",
            "- `training_only` means the training snapshot contains the file but the current reference tree does not.",
            "- `reference_only` means the current reference tree contains the file but the training snapshot does not.",
            "- Changed same-name modes include parsed `omega`, damping, array-shape, and classifier-used mode-structure diagnostics in `file_manifest.csv`.",
            "- `datconN_old` is paired with the corresponding reference `datconN` so a refreshed active file does not erase the prior continuum provenance.",
            "",
            "## Artifacts",
            "",
            "- `file_manifest.csv`: complete scoped file inventory and hashes.",
            "- `differences.csv`: non-identical or missing subset of the manifest.",
            "- `shot_summary.csv`: per-shot counts used in the table above.",
            "- `run_metadata.json`: exact roots, options, schema, and artifact hashes.",
            "- `SHA256SUMS`: integrity hashes for the generated audit artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temp_path = Path(stream.name)
    os.replace(temp_path, path)
    path.chmod(0o644)


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(stream.name)
    os.replace(temp_path, path)
    path.chmod(0o644)


def run_audit(
    *,
    training_root: Path,
    reference_root: Path,
    train_csv: Path,
    out_dir: Path,
    audit_id: str,
    generated_at: str,
    include_datcon_gf: bool = True,
    replace: bool = False,
) -> dict[str, Any]:
    """Run the audit and write all versioned artifacts."""
    for name, path in (
        ("training root", training_root),
        ("reference root", reference_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"training CSV does not exist: {train_csv}")
    existing = [out_dir / name for name in OUTPUT_FILENAMES if (out_dir / name).exists()]
    if existing and not replace:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"audit output already exists in {out_dir}: {names}; use a new versioned "
            "directory or pass --replace"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    training_rows = read_training_rows(train_csv)
    manifest = build_manifest(
        training_root,
        reference_root,
        training_rows,
        include_datcon_gf=include_datcon_gf,
    )
    summary = build_shot_summary(manifest, training_rows)
    differences = [row for row in manifest if row["status"] != "identical"]
    train_csv_digest = sha256_file(train_csv)
    audit_script_digest = sha256_file(Path(__file__).resolve())
    report = render_report(
        audit_id=audit_id,
        generated_at=generated_at,
        training_root=training_root,
        reference_root=reference_root,
        train_csv=train_csv,
        train_csv_sha256=train_csv_digest,
        audit_script_sha256=audit_script_digest,
        manifest=manifest,
        summary=summary,
        include_datcon_gf=include_datcon_gf,
    )

    manifest_path = out_dir / "file_manifest.csv"
    differences_path = out_dir / "differences.csv"
    summary_path = out_dir / "shot_summary.csv"
    report_path = out_dir / "report.md"
    _atomic_write_csv(manifest_path, MANIFEST_FIELDS, manifest)
    _atomic_write_csv(differences_path, MANIFEST_FIELDS, differences)
    _atomic_write_csv(summary_path, SUMMARY_FIELDS, summary)
    _atomic_write_text(report_path, report)

    artifact_paths = (manifest_path, differences_path, summary_path, report_path)
    artifact_hashes = {path.name: sha256_file(path) for path in artifact_paths}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": audit_id,
        "generated_at": generated_at,
        "training_root": str(training_root),
        "reference_root": str(reference_root),
        "train_csv": str(train_csv),
        "train_csv_sha256": train_csv_digest,
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": audit_script_digest,
        "include_datcon_gf": include_datcon_gf,
        "shot_count": len(summary),
        "canonical_row_count": len(training_rows),
        "manifest_row_count": len(manifest),
        "difference_row_count": len(differences),
        "artifact_sha256": artifact_hashes,
    }
    metadata_path = out_dir / "run_metadata.json"
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    checksum_paths = (*artifact_paths, metadata_path)
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
    )
    _atomic_write_text(out_dir / "SHA256SUMS", checksum_text)
    return metadata


def _default_generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare NOVA training-shot mode/continuum files with a reference "
            "shot database and write versioned SHA-256 provenance artifacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/audit_training_provenance.py \\\n"
            "    --training-root \"$NOVA_DATA\" \\\n"
            "    --reference-root \"$NOVA_DITW_ROOT\" \\\n"
            "    --train-csv \"$NOVA_TRAIN_CSV\" \\\n"
            "    --out-dir audits/training_provenance/2026-08-20_flux_v1"
        ),
    )
    parser.add_argument(
        "--training-root",
        default=os.environ.get("NOVA_DATA"),
        help="Training-data root (default: $NOVA_DATA).",
    )
    parser.add_argument(
        "--reference-root",
        default=os.environ.get("NOVA_DITW_ROOT"),
        help="Reference shot-database root (default: $NOVA_DITW_ROOT).",
    )
    parser.add_argument(
        "--train-csv",
        default=os.environ.get(
            "NOVA_TRAIN_CSV",
            str(REPO_ROOT / "training_labels" / "tae_like_train.csv"),
        ),
        help="Canonical training CSV used to select shots and labeled rows.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Versioned output directory for manifest and report artifacts.",
    )
    parser.add_argument(
        "--audit-id",
        help="Stable audit identifier (default: output-directory name).",
    )
    parser.add_argument(
        "--generated-at",
        default=_default_generated_at(),
        help="ISO timestamp recorded in metadata (default: current UTC time).",
    )
    parser.add_argument(
        "--skip-datcon-gf",
        action="store_true",
        help="Exclude auxiliary datcon_gf.txt files from the audit.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace known generated files in an existing audit directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.training_root:
        parser.error("--training-root is required when NOVA_DATA is unset")
    if not args.reference_root:
        parser.error("--reference-root is required when NOVA_DITW_ROOT is unset")
    out_dir = Path(args.out_dir).expanduser().resolve()
    metadata = run_audit(
        training_root=Path(args.training_root).expanduser().resolve(),
        reference_root=Path(args.reference_root).expanduser().resolve(),
        train_csv=Path(args.train_csv).expanduser().resolve(),
        out_dir=out_dir,
        audit_id=args.audit_id or out_dir.name,
        generated_at=args.generated_at,
        include_datcon_gf=not args.skip_datcon_gf,
        replace=args.replace,
    )
    print(
        f"Wrote provenance audit {metadata['audit_id']} for "
        f"{metadata['shot_count']} shots and {metadata['canonical_row_count']} "
        f"canonical rows to {out_dir}"
    )
    print(f"Manifest rows: {metadata['manifest_row_count']}")
    print(f"Differences: {metadata['difference_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
