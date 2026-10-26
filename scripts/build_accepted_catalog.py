#!/usr/bin/env python3
"""Build the common 12-column catalog from installed final GOOD lists.

Example: python scripts/build_accepted_catalog.py --sort-root ../sort_outputs \
    --output catalogs/accepted_modes.csv
No sorting, raw-mode loading, or classifier inference is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

from _repo_bootstrap import ensure_repo_src_on_path

REPO = ensure_repo_src_on_path()
from input_validity import REGISTRY_PATH, load_input_validity_registry
from tae_rule_io import portable_mode_key, sha256_file, write_dict_csv, write_text

FIELDS = (
    "path", "shot", "family", "ntor", "omega", "rad_loc", "rad_width",
    "rule_config", "overall_rule_severity", "manual_override",
    "final_decision", "review_status",
)
FAMILIES = {"tae_like": "TAE", "mixed": "mixed", "eae_like": "EAE", "bae_like": "BAE"}
CONFLICT_FIELDS = ("path", "shot", "final_decision", "review_decisions", "review_sources", "input_fingerprint")
REQUIRED = {
    "path", "mode_key", "shot", "ntor", "omega", "rad_loc", "rad_width",
    "gap_region", "input_fingerprint", "rule_version", "manual_decision",
    "override_status", "decision_source", "final_decision", "selected_final",
}


def read_csv(path, required=()):
    """Read strict CSV, including the large JSON cells in original exports."""
    csv.field_size_limit(2**31 - 1)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames or []
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{path}: missing or repeated column names")
        missing = set(required) - set(header)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: malformed CSV row")
            rows.append({key: value.strip() for key, value in row.items()})
    return rows


def read_reviews(config, repo=REPO):
    """Read decisions and the exact input fingerprints to which they apply."""
    reviews = {}
    sources = []
    for spec in read_csv(config, ("path", "decision_column", "shot", "reviewer")):
        path = repo / spec["path"]
        digest = sha256_file(path)
        fingerprints = {}
        fingerprint_path = spec.get("fingerprint_source", "")
        fingerprint_digest = ""
        if fingerprint_path:
            # The canonical training labels omit fingerprints; the saved training
            # audit supplies them. Never infer review from training membership.
            fingerprint_digest = sha256_file(repo / fingerprint_path)
            for row in read_csv(repo / fingerprint_path, ("mode_key", "input_fingerprint")):
                key = portable_mode_key(row["mode_key"])
                if key in fingerprints:
                    raise ValueError(f"{fingerprint_path}: duplicate mode {key}")
                fingerprints[key] = row["input_fingerprint"]
        for row in read_csv(path, (spec["decision_column"],)):
            key = portable_mode_key(row.get("mode_key") or row.get("path", ""))
            if spec["shot"] not in ("*", key.split("/")[0]):
                continue
            decision = row[spec["decision_column"]].upper()
            if decision not in ("GOOD", "BAD", "SKIP", "BAE", "REVIEW"):
                continue
            fingerprint = row.get("input_fingerprint") or fingerprints.get(key, "")
            if len(fingerprint) != 64:
                raise ValueError(f"{path}: missing input fingerprint for {key}")
            reviews.setdefault((key, fingerprint), []).append(dict(source=spec["path"], decision=decision))
        sources.append({**spec, "sha256": digest, "fingerprint_source_sha256": fingerprint_digest})
        if sha256_file(path) != digest:
            raise ValueError(f"Review source changed while reading: {path}")
        if fingerprint_path and sha256_file(repo / fingerprint_path) != fingerprint_digest:
            raise ValueError(f"Fingerprint source changed while reading: {fingerprint_path}")
    return reviews, sources


def number(text, field, key, optional=False):
    if not text and optional:
        return ""
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{key}: invalid {field}={text!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key}: {field} must be finite and nonnegative")
    if field == "rad_loc" and value > 1:
        raise ValueError(f"{key}: rad_loc is outside normalized radius [0, 1]")
    return text if field == "omega" else format(value, ".4g")


def compact_row(row, shot, summary, status, reviews, registry):
    key = portable_mode_key(row["path"])
    if row["mode_key"] != key or row["shot"] != shot or key.split("/")[0] != shot:
        raise ValueError(f"{shot}: inconsistent path, mode_key or shot: {row['path']}")
    ntor = int(row["ntor"])
    if ntor < 1 or key.split("/")[1] != f"N{ntor}":
        raise ValueError(f"{key}: ntor does not match its N directory")
    if registry.diagnostic(shot, ntor):
        raise ValueError(f"{key}: final list contains a currently excluded input; audit the source list")
    if row["final_decision"] != "GOOD" or row["selected_final"].lower() != "true":
        raise ValueError(f"{key}: final list contains a non-GOOD or unselected row")
    if row["gap_region"] not in FAMILIES:
        raise ValueError(f"{key}: unsupported family {row['gap_region']!r}")
    if len(row["input_fingerprint"]) != 64:
        raise ValueError(f"{key}: missing input fingerprint")
    config = summary.get("rule_configuration_name", "")
    if not config:
        raise ValueError(f"{shot}: shot summary does not identify the rule configuration")
    for field in ("rule_configuration_name", "rule_configuration_sha256", "rule_version"):
        if row.get(field) and summary.get(field) and row[field] != summary[field]:
            raise ValueError(f"{key}: {field} differs between row and shot summary")
    manual = row["override_status"] == "APPLIED"
    if manual:
        if row["manual_decision"] != "GOOD" or row["decision_source"] != "manual_override":
            raise ValueError(f"{key}: inconsistent applied manual override")
    elif row["manual_decision"] or row["decision_source"] == "manual_override":
        raise ValueError(f"{key}: manual decision is present without an applied override")
    evidence = reviews.get((key, row["input_fingerprint"]), [])
    decisions = {item["decision"] for item in evidence}
    if manual or decisions == {"GOOD"}:
        review = "reviewed"
    elif decisions or status == "sorted_rules_pending_review":
        review = "pending"  # Includes an unresolved human-label/final-list conflict.
    else:
        review = "unknown"
    result = dict(path=key, shot=shot, family=FAMILIES[row["gap_region"]], ntor=ntor,
                  rule_config=config, manual_override="yes" if manual else "no",
                  final_decision="GOOD", review_status=review)
    for field in ("omega", "rad_loc", "rad_width", "overall_rule_severity"):
        result[field] = number(row.get(field, ""), field, key,
                               optional=field == "overall_rule_severity")
    return result


def build_catalog(sort_root, inventory_path, review_config, registry_path=REGISTRY_PATH):
    sort_root = Path(sort_root)
    inventory_rows = read_csv(inventory_path, ("shot", "status", "active_training_shot", "post_training_checked"))
    inventory = {row["shot"]: row for row in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise ValueError("Inventory contains duplicate shots")
    reviews, review_sources = read_reviews(review_config)
    registry = load_input_validity_registry(registry_path)
    # Exactly one directory level; never enter backups or scan the raw database.
    sources = sorted(p for p in sort_root.iterdir() if p.is_dir() and (p / "good_tae_final.csv").is_file())
    expected = {s for s, r in inventory.items() if r["active_training_shot"] == "yes" or r["post_training_checked"] == "yes"}
    missing = expected - {p.name for p in sources}
    if missing:
        raise ValueError("Missing final lists for inventoried completed shots: " + ", ".join(sorted(missing)))
    if not sources:
        raise ValueError(f"No direct per-shot good_tae_final.csv files found in {sort_root}")
    catalog, receipts, conflicts, seen = [], [], [], set()
    for directory in sources:
        shot = directory.name
        if shot not in inventory:
            raise ValueError(f"{shot}: final list is missing a shot inventory entry")
        selected = directory / "good_tae_final.csv"
        summary_path = directory / "shot_summary_wide.csv"
        hashes = {p.name: sha256_file(p) for p in (selected, summary_path)}
        summaries = read_csv(summary_path, ("shot", "n_final_good", "rule_configuration_name"))
        if len(summaries) != 1 or summaries[0]["shot"] != shot:
            raise ValueError(f"{summary_path}: expected one matching shot summary")
        summary = summaries[0]
        rows = read_csv(selected, REQUIRED)
        if len(rows) != int(summary["n_final_good"]):
            raise ValueError(f"{shot}: final-list row count differs from shot summary")
        shot_rows = []
        for row in rows:
            compact = compact_row(row, shot, summary, inventory[shot]["status"], reviews, registry)
            if compact["path"] in seen:
                raise ValueError(f"Duplicate selected mode: {compact['path']}")
            seen.add(compact["path"])
            shot_rows.append(compact)
            evidence = reviews.get((compact["path"], row["input_fingerprint"]), [])
            if compact["manual_override"] == "no" and any(item["decision"] != "GOOD" for item in evidence):
                conflicts.append(dict(path=compact["path"], shot=shot, final_decision="GOOD",
                                      review_decisions="|".join(sorted({item["decision"] for item in evidence})),
                                      review_sources="|".join(sorted({item["source"] for item in evidence})),
                                      input_fingerprint=row["input_fingerprint"]))
        catalog.extend(shot_rows)
        receipts.append(dict(shot=shot, rows=len(rows), sha256=hashes,
                             rule_config=summary["rule_configuration_name"],
                             rule_configuration_sha256=summary.get("rule_configuration_sha256", ""),
                             rule_versions=sorted({r["rule_version"] for r in rows}),
                             inventory_status=inventory[shot]["status"],
                             active_training_shot=inventory[shot]["active_training_shot"],
                             review_status_counts=dict(Counter(r["review_status"] for r in shot_rows))))
        for name, digest in hashes.items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Source changed while reading: {directory / name}")
    catalog.sort(key=lambda r: (r["shot"], r["ntor"], float(r["omega"]), r["path"]))
    metadata = dict(schema="accepted-mode-catalog-v1", columns=list(FIELDS),
                    sort_root=str(sort_root.resolve()),
                    path_base="NOVA_DITW_ROOT (the live raw-mode database, not sort_root)",
                    omega_units="Unconverted NOVA omega as stored in the source export; not asserted to be Hz or kHz",
                    rounding="rad_loc, rad_width and overall_rule_severity: four significant digits; omega unchanged",
                    review_policy="Matching GOOD-only review evidence or applied manual override => reviewed; conflicting/rejecting review or inventory pending status => pending; otherwise unknown. Training labels count when their saved audit fingerprint matches. Reviewed does not mean blind review or guarantee physical validity.",
                    exporter_sha256=sha256_file(Path(__file__)),
                    inputs={str(Path(p).resolve()): sha256_file(Path(p)) for p in (inventory_path, review_config, registry_path)},
                    review_sources=review_sources, source_shots=receipts, review_conflicts=conflicts, total_modes=len(catalog),
                    missing_severity=sum(not r["overall_rule_severity"] for r in catalog),
                    counts={field: dict(Counter(r[field] for r in catalog)) for field in ("family", "rule_config", "manual_override", "review_status")})
    return catalog, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sort-root", type=Path, default=REPO.parent / "sort_outputs", help="Directory containing direct per-shot final lists (default: ../sort_outputs relative to this repository)")
    parser.add_argument("--output", type=Path, default=REPO / "catalogs/accepted_modes.csv", help="Catalog CSV to write/update; companion .metadata.json records provenance")
    parser.add_argument("--inventory", type=Path, default=REPO / "audits/main_dataset_shots/shot_status.csv", help="Shot inventory supplying expected coverage and pending-review status")
    parser.add_argument("--review-sources", type=Path, default=REPO / "configs/catalog_review_sources.csv", help="Registry of explicit per-mode GOOD review evidence; paths are relative to the repository")
    args = parser.parse_args()
    try:
        # Protect source exports from an accidental output path.
        if args.output.resolve().is_relative_to(args.sort_root.resolve()):
            raise ValueError("Write the combined catalog outside the per-shot sort root")
        rows, metadata = build_catalog(args.sort_root, args.inventory, args.review_sources)
        protected = {Path(p).resolve() for p in metadata["inputs"]}
        for source in metadata["review_sources"]:
            protected.add((REPO / source["path"]).resolve())
            if source.get("fingerprint_source"):
                protected.add((REPO / source["fingerprint_source"]).resolve())
        destinations = (args.output, args.output.with_suffix(".metadata.json"),
                        args.output.with_suffix(".review_conflicts.csv"))
        if any(p.resolve() in protected for p in destinations):
            raise ValueError("Catalog output must not replace an inventory, review or label source")
        write_dict_csv(args.output, FIELDS, rows)
        metadata["catalog_sha256"] = sha256_file(args.output)
        conflict_path = args.output.with_suffix(".review_conflicts.csv")
        write_dict_csv(conflict_path, CONFLICT_FIELDS, metadata["review_conflicts"])
        metadata["review_conflicts_sha256"] = sha256_file(conflict_path)
        write_text(args.output.with_suffix(".metadata.json"), json.dumps(metadata, indent=2) + "\n")
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(rows):,} modes from {len(metadata['source_shots'])} shot lists to {args.output}")
    print("Review status: " + str(metadata["counts"]["review_status"]))
    print(f"Unavailable severity: {metadata['missing_severity']:,}; values left blank")
    print(f"Final-list/review conflicts: {len(metadata['review_conflicts'])}; see {conflict_path}")


if __name__ == "__main__":
    main()
