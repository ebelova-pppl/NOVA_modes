"""Measure continuum-side noise and calibrate hypothetical cuts without sorting.

Example (all paths are supplied by the caller):
  python scripts/audit_continuum_noise.py measure --mode-list labels.csv \
    --data-root /path/to/data --cohort training --out-dir outputs/review_noise
  python scripts/audit_continuum_noise.py sweep \
    --measurements outputs/review_noise/measurements.jsonl \
    --top2-min .005 .01 --local-min .1 .2 --radial-length-min .02 .04 --out-dir outputs/review_noise/sweep
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import product
import json
from pathlib import Path
import sys

from _repo_bootstrap import ensure_repo_src_on_path

REPO = ensure_repo_src_on_path()
from continuum_noise import (  # noqa: E402
    SCHEMA_VERSION, ContinuumNoiseThresholds, assess_continuum_noise,
    measure_continuum_noise,
)
from cont_features import CONTINUUM_PREPROCESSING_VERSION  # noqa: E402
from input_validity import load_input_validity_registry  # noqa: E402
from make_tae_like_list import _inspect_mode_file, _load_gap_data  # noqa: E402
from tae_eae_features import classify_gap_region  # noqa: E402
from tae_rule_io import datcon_path_for_mode, input_fingerprint, sha256_file, stable_json  # noqa: E402


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mode_key(row):
    return row.get("mode_key") or "/".join(Path(row["path"]).parts[-3:])


def measure_one(task):
    supplied, root, cohort, baseline, registry = task
    path = Path(supplied["path"]).expanduser()
    if not path.is_absolute():
        path = root / path
    key = mode_key(supplied)
    label = supplied.get("validity", supplied.get("training_label", "")).lower()
    if label not in ("", "good", "bad"):
        raise ValueError(f"{key}: unsupported truth label {label!r}")
    prior = baseline or supplied
    decision = prior.get("after", prior.get("final_decision", ""))
    expected_fingerprint = prior.get("input_fingerprint", "")
    row = dict(
        cohort=cohort, mode_key=key, path=str(path), shot=path.parents[1].name,
        training_label=label, baseline_decision=decision,
        baseline_reason=prior.get("reason", prior.get("rule_primary_reason", "")),
        input_fingerprint="", baseline_fingerprint_verified=False,
        status="INVALID", reason="", nr=None, gap_region="", features=None,
    )
    if baseline and baseline.get("training_label", label) != label:
        raise ValueError(f"{key}: baseline training label changed")
    try:
        fingerprint = input_fingerprint(path, datcon_path_for_mode(path))
    except OSError as exc:
        if expected_fingerprint:
            raise ValueError(f"{key}: baseline input unavailable: {exc}") from exc
        row["reason"] = f"INPUT_UNAVAILABLE: {exc}"
        return row
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError(f"{key}: baseline fingerprint mismatch")
    row.update(input_fingerprint=fingerprint,
               baseline_fingerprint_verified=bool(expected_fingerprint))
    n = int(path.parent.name.removeprefix("N"))
    exclusion = registry.diagnostic(path.parents[1].name, n)
    if exclusion:
        row["reason"] = "KNOWN_INVALID_INPUT: " + exclusion
        return row
    bundle, reason, message = _inspect_mode_file(path, expected_n=n)
    if bundle is None:
        row["reason"] = reason + ": " + message
        return row
    row["nr"] = bundle["nr"]
    gap, reason, message = _load_gap_data(path, mode=bundle["mode"], omega=bundle["omega"])
    if gap is None:
        row["reason"] = reason + ": " + message
        return row
    region = classify_gap_region(**gap.scalars)
    row["gap_region"] = region
    if region == "eae_like":
        row.update(status="ROUTED_EAE", reason="ROUTED_EAE")
        if baseline and decision != "ROUTED_EAE":
            raise ValueError(f"{key}: baseline gap routing changed")
        return row
    if baseline and decision not in ("GOOD", "BAD", "REVIEW"):
        raise ValueError(f"{key}: baseline routing/status changed")
    features = measure_continuum_noise(bundle["mode"], bundle["omega"], gap.low2, gap.high2)
    row.update(status="MEASURED", features=features)
    return row


def measure(args):
    supplied = read_csv(args.mode_list)
    if not supplied or any(not row.get("path") for row in supplied):
        raise ValueError("mode-list requires a header and a nonempty path on every row")
    keys = [mode_key(row) for row in supplied]
    if len(set(keys)) != len(keys):
        raise ValueError("mode-list has duplicate mode keys")
    baseline = {}
    if args.baseline_csv:
        prior = read_csv(args.baseline_csv)
        baseline = {mode_key(row): row for row in prior}
        if len(baseline) != len(prior) or not set(keys) <= baseline.keys():
            raise ValueError("baseline must have unique keys and cover every requested mode")
    registry = load_input_validity_registry()
    tasks = [(row, args.data_root, args.cohort, baseline.get(mode_key(row)), registry)
             for row in supplied]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = args.out_dir / "measurements.inprogress.jsonl"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with pending.open("w") as stream:
            for i, row in enumerate(pool.map(measure_one, tasks, chunksize=8), 1):
                rows.append(row)
                stream.write(stable_json(row) + "\n")
                if i % 200 == 0:
                    print(f"Measured {i}/{len(tasks)}", flush=True)
    pending.replace(args.out_dir / "measurements.jsonl")
    ledger = [{k: v for k, v in row.items() if k != "features"} for row in rows]
    write_csv(args.out_dir / "mode_summary.csv", ledger, list(ledger[0]))
    region_rows = [
        {**{k: row[k] for k in ("cohort", "mode_key", "training_label", "baseline_decision")},
         **record}
        for row in rows if row["features"] is not None
        for record in row["features"]["records"]
    ]
    write_csv(args.out_dir / "region_metrics.csv", region_rows,
              list(region_rows[0]) if region_rows else ["mode_key", "region_id"])
    counts = Counter(row["status"] for row in rows)
    measured = [row for row in rows if row["status"] == "MEASURED"]
    summary = dict(
        schema_version=SCHEMA_VERSION, cohort=args.cohort, gate_enabled=False,
        n_input_modes=len(rows),
        measurements_sha256=sha256_file(args.out_dir / "measurements.jsonl"),
        mode_list_sha256=sha256_file(args.mode_list),
        baseline_sha256=sha256_file(args.baseline_csv) if args.baseline_csv else None,
        input_validity_sha256=registry.sha256,
        continuum_preprocessing_version=CONTINUUM_PREPROCESSING_VERSION,
        source_sha256={p: sha256_file(REPO / p) for p in (
            "src/continuum_noise.py", "scripts/audit_continuum_noise.py",
            "src/cont_features.py", "src/nova_mode_loader.py", "src/tae_eae_features.py",
            "scripts/make_tae_like_list.py", "src/input_validity.py")},
        counts=dict(counts), n_radial=dict(Counter(row["nr"] for row in measured)),
        measured_label_counts=dict(Counter(row["training_label"] for row in measured)),
        resolution_policy="native-grid-high-pass-with-radial-length",
        baseline_fingerprints_verified=sum(row["baseline_fingerprint_verified"] for row in rows),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def sweep(args):
    rows = []
    for path in args.measurements:
        receipt = json.loads((path.parent / "summary.json").read_text())
        if receipt["measurements_sha256"] != sha256_file(path):
            raise ValueError(f"{path}: measurement hash mismatch; rerun measure")
        with path.open() as stream:
            block = [json.loads(line) for line in stream if line.strip()]
        if len(block) != receipt["n_input_modes"]:
            raise ValueError(f"{path}: incomplete measurement run")
        rows.extend(block)
    keys = [(row["cohort"], row["mode_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate cohort/mode in measurements")
    for row in rows:
        if row["features"] and row["features"]["schema_version"] != SCHEMA_VERSION:
            raise ValueError("incompatible continuum noise schema")
    configs = [ContinuumNoiseThresholds(*values) for values in
               product(args.top2_min, args.local_min, args.radial_length_min)]
    if args.export_flags and len(configs) != 1:
        raise ValueError("--export-flags requires exactly one threshold combination")
    measured = [row for row in rows if row["features"] is not None]
    summaries, flags = [], []
    for config in configs:
        for cohort in sorted({row["cohort"] for row in rows}):
            eligible = [row for row in measured if row["cohort"] == cohort]
            counts = Counter()
            for row in eligible:
                label = row["training_label"] or "unlabeled"
                survivor = row["baseline_decision"] == "GOOD"
                counts[f"{label}_modes"] += 1
                counts[f"{label}_survivors"] += survivor
                decision = assess_continuum_noise(row["features"], config)
                if decision["candidate_found"]:
                    counts["flagged_modes"] += 1
                    counts[f"{label}_flagged"] += 1
                    counts[f"{label}_survivors_flagged"] += survivor
                    if args.export_flags:
                        flags.append({**{k: row[k] for k in (
                            "cohort", "mode_key", "path", "training_label", "baseline_decision",
                            "baseline_reason", "input_fingerprint")}, **decision["witness"]})
            summaries.append(dict(
                cohort=cohort, top2_min=config.top2_min, local_min=config.local_min,
                radial_length_min=config.radial_length_min, eligible_modes=len(eligible),
                **{key: counts[key] for key in (
                    "good_modes", "bad_modes", "unlabeled_modes", "good_survivors", "bad_survivors",
                    "unlabeled_survivors", "flagged_modes", "good_flagged", "bad_flagged", "unlabeled_flagged",
                    "good_survivors_flagged", "bad_survivors_flagged", "unlabeled_survivors_flagged")},
            ))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "threshold_sweep.csv", summaries, list(summaries[0]))
    if args.export_flags:
        write_csv(args.out_dir / "flagged_modes.csv", flags,
                  list(flags[0]) if flags else ["cohort", "mode_key"])
    (args.out_dir / "sweep_inputs.json").write_text(json.dumps(dict(
        schema_version=SCHEMA_VERSION,
        measurements_sha256={str(p): sha256_file(p) for p in args.measurements},
        source_sha256={p: sha256_file(REPO / p) for p in (
            "src/continuum_noise.py", "scripts/audit_continuum_noise.py")},
        thresholds=[dict(top2_min=c.top2_min, local_min=c.local_min, radial_length_min=c.radial_length_min)
                    for c in configs],
    ), indent=2) + "\n")
    print(f"Evaluated {len(configs)} threshold combinations; {len(summaries)} cohort rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("measure", help="Measure inputs; does not classify or modify production outputs")
    p.add_argument("--mode-list", type=Path, required=True, help="CSV with path and optional validity or training_label")
    p.add_argument("--data-root", type=Path, required=True, help="Root for relative mode paths")
    p.add_argument("--cohort", required=True, help="Name to keep this population separate during calibration")
    p.add_argument("--baseline-csv", type=Path, help="Optional fingerprinted rules export or training-comparison CSV")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(run=measure)
    p = subs.add_parser("sweep", help="Test explicit inclusive cuts against saved measurements")
    p.add_argument("--measurements", type=Path, nargs="+", required=True)
    p.add_argument("--top2-min", type=float, nargs="+", required=True, help="Minimum HF energy / full-domain top-two energy")
    p.add_argument("--local-min", type=float, nargs="+", required=True, help="Minimum HF energy / raw energy in the same outside region")
    p.add_argument("--radial-length-min", type=float, nargs="+", required=True, help="Minimum effective radial length in normalized radius in that same region")
    p.add_argument("--export-flags", action="store_true", help="Write witness rows for one threshold combination")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=sweep)
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
