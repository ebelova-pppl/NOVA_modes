"""Audit a broad, locally smooth exception to the narrow-envelope gate.

No production decisions are written. Example:
python audits/envelope_footprint_20260913/audit.py --rules-root /path/to/sort_outputs \
  --training-root /path/to/data_mixed --out-dir outputs/review_envelope_footprint_20260913
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
from envelope_footprint import measure_envelope_footprint
from make_tae_like_list import _inspect_mode_file, _load_gap_data
from tae_eae_features import classify_gap_region
from tae_rule_engine import (
    evaluate_mode, ContinuumCrossingConfig, EdgeArtifactConfig,
    InteriorUnresolvedEnvelopeConfig,
)
from tae_rule_io import input_fingerprint, datcon_path_for_mode, sha256_file, stable_json

REASON = "BAD_INTERIOR_UNRESOLVED_ENVELOPE"
csv.field_size_limit(10**8)


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def morphology(mode, window_dr=0.05):
    """Preserve the original audit columns using the shared production helper."""
    result = measure_envelope_footprint(mode, window_dr)
    if result["status"] != "MEASURED":
        raise ValueError(result["status"])
    return {k: v for k, v in result.items() if k not in {"status", "components", "windows"}}


def measure(task):
    cohort, source, path, compare_features = task
    path = Path(path)
    fp = input_fingerprint(path, datcon_path_for_mode(path))
    assert fp == source["input_fingerprint"], path
    bundle, reason, message = _inspect_mode_file(path, expected_n=int(path.parent.name[1:]))
    assert bundle is not None, (path, reason, message)
    gap, reason, message = _load_gap_data(path, mode=bundle["mode"], omega=bundle["omega"])
    assert gap is not None, (path, reason, message)
    evidence = dict(path=str(path), mode_key=source["mode_key"], shot=path.parents[1].name,
                    ntor=bundle["ntor"], omega=bundle["omega"], gamma_d=bundle["gamma_d"],
                    input_fingerprint=fp, gap_region=classify_gap_region(**gap.scalars))
    # Reproduce the v12 calibration decisions even after production adoption.
    result = evaluate_mode(evidence, mode=bundle["mode"], low2=gap.low2, high2=gap.high2,
                           continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
                           edge_artifact_config=EdgeArtifactConfig(secondary_peak_energy_min=None),
                           interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(footprint_spikes_fraction_max=None))
    if compare_features:
        actual = json.loads(stable_json(result.features))
        prior = json.loads(source["rule_features"])
        for field in ("edge_energy_local_peaks", "edge_body_r_max", "edge_body_amplitude_max"):
            actual["boundary_features"]["edge_artifact"].pop(field)
        actual["resolution_features"]["interior_unresolved_envelope"].pop("footprint_exception")
        for f in (actual, prior):
            f.pop("feature_schema_version")
            f["severity_features"].pop("configuration_sha256")
        assert actual == prior, path
        assert result.primary_reason == source["rule_primary_reason"], path
    else:
        assert result.primary_reason == source["reason"], path
        assert ("GOOD" if result.decision == "REVIEW" else result.decision) == source["after"], path
    fired = [k for k, v in result.features["severity_features"]["gates"].items() if v["fired"]]
    e = result.features["resolution_features"]["interior_unresolved_envelope"]
    row = dict(path=str(path), mode_key=source["mode_key"], cohort=cohort,
               training_label=source.get("training_label", ""), input_fingerprint=fp,
               current_decision="GOOD" if result.decision == "REVIEW" else result.decision,
               current_reason=result.primary_reason, fired_gates=";".join(fired),
               envelope_is_only_rejection=fired == [REASON],
               energy_fwhm_grid=e["energy_halfmax_width_grid"],
               ext_dr=e["ext_dr"], ext_df_gap=e["ext_df_gap"],
               **morphology(bundle["mode"]))
    row["candidate_exception"] = row["spikes_fraction"] < 0.5 and row["local_hf_fraction"] < 0.05
    row["candidate_new_survivor"] = row["envelope_is_only_rejection"] and row["candidate_exception"]
    assert input_fingerprint(path, datcon_path_for_mode(path)) == fp
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rules-root", "training-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert not any(args.out_dir.iterdir()), "Choose a new, empty audit output directory"
    labels_path = REPO / "training_labels/tae_like_train.csv"
    receipt_path = REPO / "audits/distributed_harmonic_noise_20260913/adoption/verification.json"
    hashes = json.loads(receipt_path.read_text())["source_sha256"]
    for name, sha in hashes.items():
        if name.endswith("/training_labels/tae_like_train.csv"):
            assert sha256_file(labels_path) == sha
    baseline = REPO / "outputs/review_distributed_noise_v12_20260913/training_comparison.csv"
    training = {r["mode_key"]: r for r in read(baseline)}
    labels = {r["path"]: r["validity"] for r in read(labels_path)}
    assert {k: r["training_label"] for k, r in training.items()} == labels
    tasks = []
    training_controls = {
        "nstxuG142301Y93/N3/egn03w.8350E+01", "nstxuG142301Y93/N8/egn08w.9431E+01",
    }
    for key, row in training.items():
        if row["reason"] == REASON or key in training_controls:
            tasks.append(("training", row, args.training_root / key, False))
    shots_path = REPO / "audits/main_dataset_shots/shot_status.csv"
    shots = sorted(r["shot"] for r in read(shots_path) if r["post_training_checked"] == "yes")
    pilot_controls = {
        "nstxuE205040A01t016/N6/egn06w.2836E+01", "nstxuE204708F03t017/N8/egn08w.6836E+01",
        "nstxuE204708F03t017/N8/egn08w.8950E+01", "nstxuE204708F03t017/N10/egn10w.7424E+01",
        "nstxuE204645A16t015/N6/egn06w.4765E+01",
    }
    inventory = Counter()
    saved_hashes = {}
    for shot in shots:
        p = args.rules_root / shot / "all_modes_rules.csv"
        saved_hashes[str(p)] = sha256_file(p)
        for row in read(p):
            inventory["all_pilot_rows"] += 1
            f = json.loads(row["rule_features"]) if row["rule_features"] else {}
            gates = f.get("severity_features", {}).get("gates", {})
            if not gates:
                continue
            assert row["rule_configuration_name"] == "tae_rules_production_v12"
            fired = [k for k, v in gates.items() if v["fired"]]
            inventory["tae_side"] += 1
            inventory["envelope_fired"] += REASON in fired
            inventory["only_envelope"] += fired == [REASON]
            if fired == [REASON] or row["mode_key"] in pilot_controls:
                tasks.append(("pilot", row, row["path"], True))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(measure, tasks))
    rows.sort(key=lambda r: (r["cohort"], r["mode_key"]))
    write(args.out_dir / "measurements.csv", rows)
    released = [r for r in rows if r["candidate_new_survivor"]]
    write(args.out_dir / "candidate_new_survivors.csv", released)
    sweep = []
    for fraction in (0.4, 0.5, 0.6):
        for roughness in (0.03, 0.04, 0.05, 0.075, 0.1):
            selected = [r for r in rows if r["envelope_is_only_rejection"] and
                        r["spikes_fraction"] < fraction and r["local_hf_fraction"] < roughness]
            sweep.append(dict(spikes_fraction_max=fraction, local_hf_max=roughness,
                              pilot=sum(r["cohort"] == "pilot" for r in selected),
                              training_good=sum(r["training_label"] == "good" for r in selected),
                              training_bad=sum(r["training_label"] == "bad" for r in selected)))
    write(args.out_dir / "threshold_sweep.csv", sweep)
    assert all(sha256_file(Path(p)) == sha for p, sha in saved_hashes.items())
    summary = dict(status="EXPLORATORY_NOT_ADOPTED", pilot_inventory=dict(inventory),
                   checked_shots=len(shots), training_rows=len(training),
                   training_primary_envelope=sum(r["reason"] == REASON for r in training.values()),
                   freshly_measured=len(rows), new_survivors=len(released),
                   new_survivor_counts=dict(Counter(r["cohort"] + ":" + r["training_label"] for r in released)),
                   settings=dict(spikes_fraction_max=0.5, local_hf_max=0.05, window_dr=0.05),
                   source_hashes={str(p): sha256_file(p) for p in
                                  (Path(__file__), labels_path, baseline, receipt_path, shots_path,
                                   REPO / "src/continuum_noise.py", REPO / "src/envelope_footprint.py",
                                   REPO / "scripts/tae_rule_engine.py",
                                   REPO / "configs/rules/tae_rules_production_v12.yaml")},
                   saved_output_hashes=saved_hashes)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("hashes")}, indent=2))
    for row in released:
        print(row["mode_key"], "F=", row["spikes_fraction"], "Q=", row["local_hf_fraction"])


if __name__ == "__main__":
    main()
