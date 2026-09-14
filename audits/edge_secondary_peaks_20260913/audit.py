"""Project an all-local-energy-peak edge gate without rerunning shot sorting.

python audits/edge_secondary_peaks_20260913/audit.py --rules-root /path/to/sort_outputs \
  --training-root /path/to/data_mixed --out-dir outputs/review_edge_secondary_peaks_20260913
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from nova_mode_loader import load_mode_from_nova
from tae_rule_engine import _signed_local_extrema, _signed_halfmax_component
from tae_rule_io import input_fingerprint, datcon_path_for_mode, sha256_file

csv.field_size_limit(10**8)
SCENARIOS = {
    "all_peaks": {},
    "W_peak_ge_0.05": {"w_min": 0.05},
    "W_peak_ge_0.10": {"w_min": 0.10},
    "W_peak_ge_0.25": {"w_min": 0.25},
    "W_peak_ge_0.50": {"w_min": 0.50},
    "A_ge_0.3_contrast_ge_3": {"a_min": 0.3, "contrast_min": 3},
    "A_ge_0.3_contrast_ge_3_width_le_4": {"a_min": 0.3, "contrast_min": 3, "width_max": 4},
    "A_ge_0.3_contrast_ge_4": {"a_min": 0.3, "contrast_min": 4},
}


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def qualifies(peak, settings):
    return (peak["width_grid"] <= settings.get("width_max", 10) + 1e-12
            and peak["W_peak"] >= settings.get("w_min", 0)
            and peak["amplitude"] >= settings.get("a_min", 0)
            and (not settings.get("contrast_min") or
                 peak["contrast"] is not None and peak["contrast"] >= settings["contrast_min"]))


def measure(task):
    cohort, source, path, latest = task
    path = Path(path)
    fp = input_fingerprint(path, datcon_path_for_mode(path))
    assert fp == source["input_fingerprint"], path
    A, omega, _, n = load_mode_from_nova(str(path))
    assert A.ndim == 2 and np.isfinite(A).all() and np.max(abs(A)) > 0
    assert n == int(path.parent.name[1:])
    r = np.linspace(0, 1, A.shape[1])
    W = np.sum(A*A, axis=0)
    W /= np.max(W)
    B = np.max(abs(A), axis=0) / np.max(abs(A))
    background = float(np.median(B[r >= 0.9 - 1e-12]))
    ip = int(np.argmax(W))
    global_width = _signed_halfmax_component(W, peak_index=ip, radial_grid=r)[3]
    if cohort.startswith("pilot"):
        f = json.loads(source["rule_features"])
        e = f["boundary_features"]["edge_artifact"]
        assert float(source["omega"]) == omega
        assert np.isclose(global_width, e["edge_energy_halfmax_width_grid"], rtol=1e-12)
        assert r[ip] == e["edge_energy_peak_r"]
    peaks = []
    for i, _, _ in _signed_local_extrema(W):
        if r[i] < 0.97 - 1e-12:
            continue
        lo, hi, width, grid, touches = _signed_halfmax_component(W, peak_index=i, radial_grid=r)
        peaks.append(dict(r_peak=float(r[i]), W_peak=float(W[i]), amplitude=float(B[i]),
                          width_grid=grid, inner_edge=lo, outer_edge=hi,
                          touches_boundary=touches,
                          contrast=float(B[i] / background) if background > 0 else None))
    assert input_fingerprint(path, datcon_path_for_mode(path)) == fp
    row = dict(path=str(path), mode_key=source["mode_key"], cohort=cohort,
               training_label=source.get("training_label", ""), latest12=latest,
               input_fingerprint=fp, nr=A.shape[1], global_energy_peak_r=float(r[ip]),
               global_energy_width_grid=global_width, background_amplitude=background,
               edge_local_peak_count=len(peaks))
    for name, settings in SCENARIOS.items():
        row[name] = any(qualifies(p, settings) for p in peaks)
    candidates = [p for p in peaks if qualifies(p, {})]
    witness = max(candidates, key=lambda p: p["W_peak"]) if candidates else None
    for name in ("r_peak", "W_peak", "amplitude", "width_grid", "contrast"):
        row["strongest_candidate_" + name] = witness[name] if witness else None
    return row, dict(mode_key=source["mode_key"], peaks=peaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rules-root", "training-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert not any(args.out_dir.iterdir()), "Choose an empty output directory"
    receipt = REPO / "audits/distributed_harmonic_noise_20260913/adoption/verification.json"
    adopted = json.loads(receipt.read_text())["source_sha256"]
    # Reuse the verified production-v12 training baseline only with unchanged
    # production source, configuration, and training list.
    for p, sha in adopted.items():
        for prefix in ("src", "scripts", "configs", "training_labels"):
            marker = "/" + prefix + "/"
            if marker in p:
                local = REPO / prefix / p.split(marker, 1)[1]
                assert sha256_file(local) == sha, local
                break
    training_csv = REPO / "outputs/review_distributed_noise_v12_20260913/training_comparison.csv"
    training = read(training_csv)
    labels_path = REPO / "training_labels/tae_like_train.csv"
    assert {r["mode_key"]: r["training_label"] for r in training} == {
        r["path"]: r["validity"] for r in read(labels_path)}
    tasks = [("training", row, args.training_root / row["mode_key"], False)
             for row in training if row["after"] == "GOOD"]
    latest = {r["shot"] for r in read(REPO / "audits/pilot12_v11_20260910/selection.csv")}
    pending = {r["mode_key"] for r in read(REPO / "audits/envelope_footprint_20260913/candidate_new_survivors.csv")}
    shots_path = REPO / "audits/main_dataset_shots/shot_status.csv"
    shots = sorted(r["shot"] for r in read(shots_path) if r["post_training_checked"] == "yes")
    saved = {}
    for shot in shots:
        p = args.rules_root / shot / "all_modes_rules.csv"
        saved[str(p)] = sha256_file(p)
        for row in read(p):
            if row["final_decision"] == "GOOD" or row["mode_key"] in pending:
                assert row["rule_configuration_name"] == "tae_rules_production_v12"
                cohort = "pilot" if row["final_decision"] == "GOOD" else "pilot_pending_envelope"
                if cohort == "pilot":
                    assert row["rule_decision"] == "REVIEW", row["mode_key"]
                tasks.append((cohort, row, row["path"], shot in latest))
    outputs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, output in enumerate(pool.map(measure, tasks), 1):
            outputs.append(output)
            if i % 500 == 0:
                print(f"Measured {i}/{len(tasks)} current or pending survivors", flush=True)
    rows = [r for r, _ in outputs]
    write(args.out_dir / "measurements.csv", rows)
    (args.out_dir / "peaks.json").write_text(json.dumps([p for _, p in outputs]) + "\n")
    comparisons = []
    for scenario in SCENARIOS:
        selected = [r for r in rows if r[scenario]]
        write(args.out_dir / (scenario + ".csv"), selected, fields=list(rows[0]))
        comparisons.append(dict(scenario=scenario,
            training_good=sum(r["cohort"] == "training" and r["training_label"] == "good" for r in selected),
            training_bad=sum(r["cohort"] == "training" and r["training_label"] == "bad" for r in selected),
            pilot39=sum(r["cohort"] == "pilot" for r in selected),
            latest12=sum(r["cohort"] == "pilot" and r["latest12"] for r in selected),
            pending_envelope=sum(r["cohort"] == "pilot_pending_envelope" for r in selected)))
    write(args.out_dir / "comparison.csv", comparisons)
    assert all(sha256_file(Path(p)) == sha for p, sha in saved.items())
    summary = dict(status="AUDIT_ONLY", scenarios=SCENARIOS, comparisons=comparisons,
                   measured=len(rows), n_radial=sorted({r["nr"] for r in rows}),
                   cohort_counts=dict(Counter(r["cohort"] for r in rows)),
                   training_survivor_labels=dict(Counter(r["training_label"] for r in rows if r["cohort"] == "training")),
                   latest12_survivors=sum(r["cohort"] == "pilot" and r["latest12"] for r in rows),
                   source_sha256={str(p):sha256_file(p) for p in (Path(__file__), training_csv, labels_path, receipt, shots_path,
                                      REPO / "scripts/tae_rule_engine.py", REPO / "src/nova_mode_loader.py")},
                   saved_output_sha256=saved)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('sha256')}, indent=2))


if __name__ == "__main__":
    main()
