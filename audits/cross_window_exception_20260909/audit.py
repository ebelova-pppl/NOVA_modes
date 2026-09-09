"""Audit a per-crossing low-amplitude, low-K exception without changing rules.

Examples, using the project scientific Python environment:
  python audit.py --kind shots --input-root /path/to/sort_outputs --out-dir OUT
  python audit.py --kind training --input-root /path/to/training/data --out-dir OUT

Counts precede frequency/structure deduplication. No RF or CNN is loaded.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import numpy as np

from cont_features import (
    CONTINUUM_PREPROCESSING_VERSION,
    continuum_crossing_records,
    load_datcon_for_mode,
)
from nova_mode_loader import load_mode_from_nova
from tae_eae_features import classify_gap_region, upper2_scalars
from tae_rule_config import load_rule_run_configuration
from tae_rule_engine import (
    ContinuumCrossingConfig,
    ContinuumCrossingWindowConfig,
    evaluate_mode,
    extract_continuum_crossing_tail_features,
    extract_continuum_crossing_window_features,
)
from tae_rule_io import input_fingerprint, sha256_file, stable_json

WINDOW_REASON = "BAD_CONT_CROSS_WINDOW"
AMPLITUDES = (0.1, 0.15, 0.2, 0.25)
K_LIMITS = (0.05, 0.1, 0.2, 0.4)
SELECTED_A, SELECTED_K = 0.2, 0.1


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def qualifies(row, amplitude, k_limit):
    """All violating crossings must individually meet both strict cuts."""
    return (
        row["nr"] == 201
        and row.get("required_A_max") is not None
        and row.get("required_K_max") is not None
        and row["required_A_max"] < amplitude
        and row["required_K_max"] < k_limit
    )


def evaluate(row, mode, low, high, *, without_window=False):
    kwargs = {"continuum_crossing_config": ContinuumCrossingConfig(w_cross_threshold=None)}
    if without_window:
        kwargs["continuum_crossing_window_config"] = ContinuumCrossingWindowConfig(
            amplitude_min=None, w_min=None
        )
    result = evaluate_mode(row, mode=mode, low2=low, high2=high, **kwargs)
    assert result.decision in {"BAD", "REVIEW"}, result.diagnostic_message
    return result


def measure(task):
    row, kind = task
    row = dict(row)
    path = Path(row["path"])
    mode, omega, gamma, ntor = load_mode_from_nova(str(path))
    fingerprint = input_fingerprint(path, path.with_name(f"datcon{ntor}"))
    if kind == "shots":
        assert fingerprint == row["input_fingerprint"], row["mode_key"]
    base = dict(
        path=str(path), label="", mode_key=row["mode_key"], shot=row["shot"],
        input_fingerprint=fingerprint, training_label=row.get("training_label", ""),
        nr=mode.shape[1], primary_reason="", later_reason="",
    )
    # Preserve the one documented invalid training input; never evaluate it.
    valid_metadata = math.isfinite(omega) and omega > 0 and math.isfinite(gamma)
    if not valid_metadata:
        assert kind == "training", row["mode_key"]
        assert row["mode_key"] == "nstxuG121123K51/N4/egn04w.8769E+01"
        return dict(base, primary_reason="INVALID_METADATA")
    assert mode.shape[1] >= 2 and np.isfinite(mode).all()
    low, high, *_ = load_datcon_for_mode(str(path), mode.shape[1])
    baseline = None
    if kind == "training":
        scalars = upper2_scalars(mode, omega, high)
        region = classify_gap_region(**scalars)
        row.update(omega=omega, gamma_d=gamma, ntor=ntor,
                   input_fingerprint=fingerprint, gap_region=region, **scalars)
        if region == "eae_like":
            return dict(base, primary_reason="ROUTED_EAE")
        baseline = evaluate(row, mode, low, high)
        base["primary_reason"] = baseline.primary_reason
        if baseline.primary_reason != WINDOW_REASON:
            return base
        features = baseline.features
    else:
        base["primary_reason"] = row["rule_primary_reason"]
        assert base["primary_reason"] == WINDOW_REASON
        features = json.loads(row["rule_features"])

    r = np.linspace(0, 1, mode.shape[1])
    crossings = continuum_crossing_records(mode, omega, low, high, r)
    tail = extract_continuum_crossing_tail_features(mode, crossings)
    assert stable_json(crossings) == stable_json(features["crossing_records"])
    assert stable_json(tail) == stable_json(features["crossing_features"]["continuum_crossing_tail"])
    records = []
    for cross in crossings:
        window = extract_continuum_crossing_window_features(mode, [cross])
        if window["cross_window_A_max"] < 0.25 and window["cross_window_W_max"] < 0.05:
            continue
        rc = cross["r_cross"]
        k_record = next(x for x in tail["records"]
                        if x["boundary"] == cross["boundary"] and x["r_cross"] == rc)
        # Interpolate each SIGNED harmonic before taking its absolute value.
        samples = np.array([np.interp(rc, r, profile) for profile in mode])
        records.append(dict(
            boundary=cross["boundary"], r_cross=rc,
            A_cross=float(np.max(np.abs(samples))),
            A_cross_harmonic=int(np.argmax(np.abs(samples))),
            K_cross=k_record["K_cross"], W_cross=cross["W_peak"],
            window_A=window["cross_window_A_max"], window_W=window["cross_window_W_max"],
            inner_energy_fraction=k_record["energy_fraction_inner"],
        ))
    assert records, row["mode_key"]
    k_values = [x["K_cross"] for x in records]
    base.update(
        n_violating_crossings=len(records),
        required_A_max=max(x["A_cross"] for x in records),
        required_K_max=max(k_values) if all(k is not None for k in k_values) else None,
        crossing_records=stable_json(records),
    )
    # Only possible recoveries require full evaluation of all later gates.
    if qualifies(base, max(AMPLITUDES), max(K_LIMITS)):
        if baseline is None:
            baseline = evaluate(row, mode, low, high)
            assert baseline.primary_reason == row["rule_primary_reason"]
            assert stable_json(baseline.features) == stable_json(features)
        later = evaluate(row, mode, low, high, without_window=True)
        base["later_reason"] = later.primary_reason
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("shots", "training"), required=True)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Saved rules root for shots, raw training-data root for training")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.workers > 0
    config = load_rule_run_configuration("tae_rules_production_v6")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = [config.source_path, Path(__file__)]
    sources += [REPO / p for p in ("scripts/tae_rule_engine.py", "src/cont_features.py",
                                  "src/nova_mode_loader.py", "src/tae_eae_features.py")]
    tasks = []
    baseline_counts = Counter()
    if args.kind == "shots":
        membership = REPO / "audits/continuum_monotonic_tail_20260908/regenerated_shot_summary.csv"
        sources.append(membership)
        members = read_csv(membership)
        assert len(members) == 27
        for shot in members:
            source = args.input_root / shot["shot"] / "all_modes_rules.csv"
            sources.append(source)
            rows = read_csv(source)
            assert len(rows) == int(shot["input_modes"])
            for row in rows:
                assert not row["manual_decision"], row["mode_key"]
                reason = row["rule_primary_reason"] or row["processing_status"]
                baseline_counts[reason] += 1
                if reason == WINDOW_REASON:
                    tasks.append((row, args.kind))
    else:
        labels_path = REPO / "training_labels/tae_like_train.csv"
        sources.append(labels_path)
        labels = read_csv(labels_path)
        assert len({x["path"] for x in labels}) == len(labels) == 2390
        assert set(x["validity"] for x in labels) == {"good", "bad"}
        for label in labels:
            key = label["path"]
            tasks.append((dict(path=str(args.input_root / key), mode_key=key,
                               shot=key.split("/")[0], training_label=label["validity"]), args.kind))
    hashes = {str(p): sha256_file(p) for p in sources}
    measured = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(measure, tasks, chunksize=8), 1):
            measured.append(result)
            if i % 100 == 0 or i == len(tasks):
                print(f"{args.kind}: {i}/{len(tasks)}", flush=True)
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items()), "Audit source changed during run"
    if args.kind == "training":
        baseline_counts.update(x["primary_reason"] for x in measured)
    rejected = [x for x in measured if x["primary_reason"] == WINDOW_REASON]
    fields = list(rejected[0])
    write_csv(args.out_dir / "window_rejections.csv", rejected, fields)
    sweep = []
    for a in AMPLITUDES:
        for k in K_LIMITS:
            exempted = [x for x in rejected if qualifies(x, a, k)]
            recovered = [x for x in exempted if x["later_reason"] == "NO_GOOD_TEMPLATE"]
            sweep.append(dict(A_cross_strict_max=a, K_cross_strict_max=k,
                              window_gate_cleared=len(exempted), recovered=len(recovered),
                              still_bad=len(exempted)-len(recovered),
                              training_good=sum(x["training_label"] == "good" for x in recovered),
                              training_bad=sum(x["training_label"] == "bad" for x in recovered)))
    write_csv(args.out_dir / "sweep.csv", sweep, list(sweep[0]))
    selected = [x for x in rejected if qualifies(x, SELECTED_A, SELECTED_K)]
    recovered = [x for x in selected if x["later_reason"] == "NO_GOOD_TEMPLATE"]
    still_bad = [x for x in selected if x["later_reason"] != "NO_GOOD_TEMPLATE"]
    write_csv(args.out_dir / "recovered.csv", recovered, fields)
    write_csv(args.out_dir / "still_bad.csv", still_bad, fields)
    by_shot = [dict(shot=shot, window_rejections=sum(x["shot"] == shot for x in rejected),
                    recovered=sum(x["shot"] == shot for x in recovered))
               for shot in sorted({x["shot"] for x in measured})]
    write_csv(args.out_dir / "by_shot.csv", by_shot, list(by_shot[0]))
    summary = dict(kind=args.kind, status="hypothetical_not_adopted", baseline=dict(baseline_counts),
                   selected_thresholds=dict(A_cross_strict_max=SELECTED_A, K_cross_strict_max=SELECTED_K),
                   recovered=len(recovered), still_bad=len(still_bad),
                   still_bad_reasons=dict(Counter(x["later_reason"] for x in still_bad)),
                   recovered_training_labels=dict(Counter(x["training_label"] for x in recovered)),
                   radial_resolutions=dict(Counter(x["nr"] for x in measured)),
                   preprocessing=CONTINUUM_PREPROCESSING_VERSION, config_sha256=config.sha256,
                   source_sha256=hashes, sweep=sweep)
    if args.kind == "training":
        summary["baseline_label_matrix"] = dict(Counter(
            f'{x["training_label"]}:{x["primary_reason"]}' for x in measured))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_sha256", "sweep"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
