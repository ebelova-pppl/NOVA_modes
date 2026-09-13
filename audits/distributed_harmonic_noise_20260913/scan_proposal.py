"""Audit a hypothetical simultaneous-harmonic noise branch; never sort or relabel.

Example from the repository root:
  python audits/distributed_harmonic_noise_20260913/scan_proposal.py \
    --training-root /path/to/training/data --ditw-root /path/to/DiTw \
    --out-dir outputs/review_distributed_harmonic_noise_20260913

This experimental estimator is deliberately outside the production rule engine.
Use the same native signed high pass and node quadrature as continuum_noise.py.
The global cut uses full-domain top-two-harmonic energy by default. Add
--include-pilots to audit every TAE-side mode in the checked-shot inventory.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from input_validity import load_input_validity_registry
from make_tae_like_list import _inspect_mode_file
from tae_rule_io import datcon_path_for_mode, input_fingerprint, sha256_file

csv.field_size_limit(10**8)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scan_mode(mode, cuts, global_reference="top2"):
    """Keep all witness metrics on the same window and qualifying HF centers."""
    nh_min, global_min, local_min, length_min, window_dr = cuts
    if global_reference not in ("top2", "total"):
        raise ValueError("Global energy reference must be top2 or total")
    mode = np.asarray(mode, dtype=float)
    if mode.ndim != 2 or not np.isfinite(mode).all() or mode.shape[1] < 3:
        raise ValueError("Expected finite native [harmonic, radius] mode array")
    peak = float(np.max(np.abs(mode)))
    if peak <= 0:
        raise ValueError("Mode has no nonzero amplitude")
    mode = mode / peak
    nr = mode.shape[1]
    dr = 1.0 / (nr - 1)
    span = min(nr - 1, int(np.floor(window_dr / dr + 1e-10)))
    if span < 2:
        raise ValueError("Window cannot contain a complete three-point stencil")
    weights = np.full(nr, dr)
    weights[[0, -1]] *= 0.5
    raw = np.sum(mode * mode, axis=0) * weights
    total = float(np.sum(raw))
    harmonic_energy = np.sum(mode * mode * weights, axis=1)
    top2_indices = np.argsort(-harmonic_energy, kind="stable")[:2]
    top2 = float(np.sum(harmonic_energy[top2_indices]))
    reference_energy = top2 if global_reference == "top2" else total
    # Differentiate before selection; no artificial masked-profile boundaries.
    hp = np.zeros_like(mode)
    hp[:, 1:-1] = (mode[:, 2:] - 2 * mode[:, 1:-1] + mode[:, :-2]) / 4
    hp2 = hp * hp
    power = np.sum(hp2, axis=0)
    probabilities = np.divide(hp2, power[None, :], out=np.zeros_like(hp2),
                              where=power[None, :] > 0)
    concentration = np.sum(probabilities * probabilities, axis=0)
    nh = np.divide(1.0, concentration, out=np.zeros_like(power),
                   where=concentration > 0)
    energies = power * weights
    records = []
    for start in range(nr - span):
        end = start + span
        # Both stencil neighbors remain inside this closed radial window.
        centers = np.arange(start + 1, end)
        selected = centers[nh[centers] >= nh_min]
        e = energies[selected]
        hf = float(np.sum(e))
        length = float(dr / np.sum((e / hf) ** 2)) if hf > 0 else 0.0
        raw_window = float(np.sum(raw[start:end + 1]))
        global_ratio = hf / reference_energy
        local_fraction = hf / raw_window if raw_window > 0 else 0.0
        fires = (global_ratio > global_min and local_fraction > local_min
                 and length > length_min)
        records.append(dict(
            r_start=start * dr, r_end=end * dr, n_qualifying=len(selected),
            hf_total_fraction=hf / total, hf_top2_ratio=hf / top2,
            hf_window_fraction=local_fraction, effective_length=length,
            fires=fires,
            minimum_cut_ratio=min(global_ratio / global_min,
                                  local_fraction / local_min, length / length_min),
        ))
    # A violating window always ranks above a nonviolating window. Stable ties
    # retain the earliest window. This ranking is only an audit witness choice.
    best = max(records, key=lambda row: (row["fires"], row["minimum_cut_ratio"]))
    return dict(best, candidate=any(r["fires"] for r in records),
                n_firing_windows=sum(r["fires"] for r in records),
                n_scanned_windows=len(records), native_dr=dr,
                actual_window_dr=span * dr, nr=nr, nhar=mode.shape[0],
                global_reference=global_reference, top2_energy_share=top2 / total,
                top2_harmonic_indices=json.dumps(top2_indices.tolist()),
                candidate_total_reference=any(
                    r["hf_total_fraction"] > global_min
                    and r["hf_window_fraction"] > local_min
                    and r["effective_length"] > length_min for r in records))


def measure(task):
    supplied, cuts, registry, global_reference = task
    row = dict(supplied, status="ERROR", error="")
    try:
        path = Path(row["path"])
        fingerprint = input_fingerprint(path, datcon_path_for_mode(path))
        if fingerprint != row["input_fingerprint"]:
            raise ValueError("Input fingerprint differs from saved baseline")
        n = int(path.parent.name[1:])
        excluded = registry.diagnostic(path.parents[1].name, n)
        if excluded:
            raise ValueError("Active audit input is registry-excluded: " + excluded)
        bundle, reason, message = _inspect_mode_file(path, expected_n=n)
        if bundle is None:
            if row["baseline_decision"] != "INVALID":
                raise ValueError(reason + ": " + message)
            row.update(status="INVALID", error=reason + ": " + message)
        else:
            if row["baseline_decision"] == "INVALID":
                raise ValueError("Previously invalid input now loads; review baseline")
            row.update(scan_mode(bundle["mode"], cuts, global_reference), status="MEASURED")
        if input_fingerprint(path, datcon_path_for_mode(path)) != fingerprint:
            raise ValueError("Input changed during measurement")
    except Exception as exc:
        row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--ditw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, default=REPO / "outputs/review_continuum_noise_v10_20260910/training_comparison.csv")
    parser.add_argument("--rules-root", type=Path, help="Defaults to the sibling sort_outputs directory")
    parser.add_argument("--include-pilots", action="store_true", help="Include all checked-shot TAE-side modes, retaining EAE/INVALID exclusions separately")
    parser.add_argument("--pilot-inventory", type=Path, default=REPO / "audits/main_dataset_shots/shot_status.csv", help="Shot inventory; selects post_training_checked=yes")
    parser.add_argument("--global-reference", choices=("top2", "total"), default="top2", help="Full-domain energy denominator; total reproduces the superseded first audit")
    parser.add_argument("--nhf-min", type=float, default=4, help="Inclusive simultaneous HF harmonic participation cut")
    parser.add_argument("--global-min", type=float, default=.005, help="Strict HF / reference-harmonic energy ratio cut")
    parser.add_argument("--local-min", type=float, default=.05, help="Strict HF / whole-window raw energy cut")
    parser.add_argument("--length-min", type=float, default=.03, help="Strict effective radial-length cut")
    parser.add_argument("--window-dr", type=float, default=.05, help="Maximum closed-window width in normalized radius")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cuts = (args.nhf_min, args.global_min, args.local_min, args.length_min, args.window_dr)
    if not all(np.isfinite(v) and v > 0 for v in cuts) or not args.length_min < args.window_dr <= 1:
        parser.error("Use finite positive cuts with length_min < window_dr <=1")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.out_dir.exists():
        parser.error("Choose a new out-dir to preserve previous audit evidence")
    training = REPO / "training_labels/tae_like_train.csv"
    labels = read_csv(training)
    prior = read_csv(args.baseline_csv)
    baseline = {r["mode_key"]: r for r in prior}
    assert len(baseline) == len(prior)
    tasks = []
    for label in labels:
        key = label["path"]
        previous = baseline[key]
        assert label["validity"] == previous["training_label"]
        tasks.append(dict(cohort="training", mode_key=key,
            path=str(args.training_root / key), training_label=label["validity"],
            baseline_decision=previous["after"], baseline_reason=previous["reason"],
            input_fingerprint=previous["input_fingerprint"]))
    assert len({r["mode_key"] for r in tasks}) == len(tasks)
    rules_root = args.rules_root or REPO.parent / "sort_outputs"
    controls = [("nstxuE205042A01t022", "N10/egn10w.3470E+01"),
                ("nstxuE202947A03t015", "N5/egn05w.2352E+01")]
    pilot_sources = []
    pilot_excluded = []
    pilot_shots = []
    if args.include_pilots:
        pilot_sources.append(args.pilot_inventory)
        pilot_shots = sorted(r["shot"] for r in read_csv(args.pilot_inventory)
                             if r["post_training_checked"] == "yes")
        assert pilot_shots and len(set(pilot_shots)) == len(pilot_shots)
        for shot in pilot_shots:
            source = rules_root / shot / "all_modes_rules.csv"
            pilot_sources.append(source)
            for previous in read_csv(source):
                key = previous["mode_key"]
                assert key.split("/")[0] == shot
                row = dict(cohort="pilot", mode_key=key,
                    path=str(args.ditw_root / key), training_label="",
                    baseline_decision=previous["final_decision"],
                    baseline_reason=previous["rule_primary_reason"] or previous["preprocessing_primary_reason"],
                    input_fingerprint=previous["input_fingerprint"],
                    gap_region=previous["gap_region"], shot=shot,
                    baseline_config=previous["rule_configuration_name"])
                if previous["processing_status"] == "RULE_EVALUATED":
                    assert row["baseline_decision"] in ("BAD", "GOOD", "REVIEW")
                    tasks.append(row)
                else:
                    assert previous["processing_status"] in ("ROUTED_EAE", "INVALID")
                    pilot_excluded.append(dict(row, status="EXCLUDED_" + previous["processing_status"]))
    for shot, suffix in ([] if args.include_pilots else controls):
        key = shot + "/" + suffix
        previous = next(r for r in read_csv(rules_root / shot / "all_modes_rules.csv")
                        if r["mode_key"] == key)
        pilot_sources.append(rules_root / shot / "all_modes_rules.csv")
        tasks.append(dict(cohort="pilot_examples", mode_key=key,
            path=str(args.ditw_root / key), training_label="",
            baseline_decision=previous["final_decision"],
            baseline_reason=previous["rule_primary_reason"],
            input_fingerprint=previous["input_fingerprint"]))
    registry = load_input_validity_registry()
    source_paths = [Path(__file__), training, args.baseline_csv,
                    REPO / "configs/known_invalid_inputs.csv",
                    REPO / "src/nova_mode_loader.py", REPO / "src/continuum_noise.py",
                    REPO / "scripts/make_tae_like_list.py"] + pilot_sources
    hashes = {str(p): sha256_file(p) for p in source_paths}
    args.out_dir.mkdir(parents=True)
    result = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(measure, [(r, cuts, registry, args.global_reference) for r in tasks], chunksize=4), 1):
            result.append(row)
            if i % 200 == 0:
                print(f"Measured {i}/{len(tasks)}", flush=True)
    assert hashes == {str(p): sha256_file(p) for p in source_paths}, "Audit inputs changed"
    errors = [r for r in result if r["status"] == "ERROR"]
    fields = list(dict.fromkeys(k for row in result for k in row))
    write_csv(args.out_dir / "measurements.csv", result, fields)
    tr = [r for r in result if r["cohort"] == "training"]
    flags = [r for r in tr if r.get("candidate")]
    newly = [r for r in flags if r["baseline_decision"] in ("GOOD", "REVIEW")]
    write_csv(args.out_dir / "flagged_training.csv", flags, fields)
    write_csv(args.out_dir / "flagged_good_labels.csv", [r for r in flags if r["training_label"] == "good"], fields)
    write_csv(args.out_dir / "newly_rejected_training.csv", newly, fields)
    control_keys = {shot + "/" + suffix for shot, suffix in controls}
    write_csv(args.out_dir / "pilot_examples.csv", [r for r in result if r["cohort"] != "training" and r["mode_key"] in control_keys], fields)
    pilots = [r for r in result if r["cohort"] == "pilot"]
    pilot_flags = [r for r in pilots if r.get("candidate")]
    pilot_new = [r for r in pilot_flags if r["baseline_decision"] in ("GOOD", "REVIEW")]
    write_csv(args.out_dir / "flagged_pilot.csv", pilot_flags, fields)
    write_csv(args.out_dir / "newly_rejected_pilot.csv", pilot_new, fields)
    write_csv(args.out_dir / "excluded_pilot.csv", pilot_excluded,
              list(dict.fromkeys(k for row in pilot_excluded for k in row)) or fields)
    shot_summary = []
    for shot in pilot_shots:
        measured = [r for r in pilots if r["shot"] == shot]
        excluded = [r for r in pilot_excluded if r["shot"] == shot]
        flagged = [r for r in measured if r.get("candidate")]
        shot_summary.append(dict(shot=shot,tae_side_modes=len(measured),
            prior_good=sum(r["baseline_decision"] == "GOOD" for r in measured),
            flagged_prior_bad=sum(r["baseline_decision"] == "BAD" for r in flagged),
            newly_rejected=sum(r["baseline_decision"] in ("GOOD", "REVIEW") for r in flagged),
            excluded_eae=sum(r["status"] == "EXCLUDED_ROUTED_EAE" for r in excluded),
            excluded_invalid=sum(r["status"] == "EXCLUDED_INVALID" for r in excluded)))
    write_csv(args.out_dir / "pilot_shot_summary.csv", shot_summary,
              list(shot_summary[0]) if shot_summary else ["shot"])
    report = dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(), production_gate_enabled=False,
        threshold_semantics="N_hf >= nhf_min; other three cuts strictly >; all in one window",
        cuts=dict(nhf_min=cuts[0],hf_global_min=cuts[1],global_reference=args.global_reference,hf_window_min=cuts[2],effective_length_min=cuts[3],window_dr=cuts[4]),
        window_policy="native windows; complete stencils inside; raw denominator includes all window nodes with native full-domain trapezoidal node weights",
        training_rows=len(tr), training_labels=dict(Counter(r["training_label"] for r in tr)),
        statuses=dict(Counter(r["status"] for r in result)),
        nr_counts=dict(Counter(r.get("nr") for r in result if r["status"] == "MEASURED")),
        flagged_training_labels=dict(Counter(r["training_label"] for r in flags)),
        flagged_baseline_decisions=dict(Counter(r["baseline_decision"] for r in flags)),
        newly_rejected_training_labels=dict(Counter(r["training_label"] for r in newly)),
        pilot_shots=pilot_shots, pilot_tae_side_rows=len(pilots),
        pilot_baseline_decisions=dict(Counter(r["baseline_decision"] for r in pilots)),
        pilot_excluded_statuses=dict(Counter(r["status"] for r in pilot_excluded)),
        pilot_flagged_baseline_decisions=dict(Counter(r["baseline_decision"] for r in pilot_flags)),
        pilot_newly_rejected=len(pilot_new),
        pilot_newly_rejected_with_total_reference=sum(bool(r.get("candidate_total_reference")) for r in pilot_new),
        sources_sha256=hashes, measurements_sha256=sha256_file(args.out_dir / "measurements.csv"),
        errors=[dict(mode_key=r["mode_key"],error=r["error"]) for r in errors],
    )
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ["training_rows","statuses","nr_counts","flagged_training_labels","flagged_baseline_decisions","newly_rejected_training_labels","pilot_tae_side_rows","pilot_flagged_baseline_decisions","pilot_newly_rejected_with_total_reference","errors"]},indent=2),flush=True)
    if errors:
        raise SystemExit("Audit has input errors; do not treat it as complete calibration")


if __name__ == "__main__":
    main()
