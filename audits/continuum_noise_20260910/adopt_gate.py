"""Stage, verify, and publish the approved v10 extended continuum noise gate.

python audits/continuum_noise_20260910/adopt_gate.py stage \
  --rules-root /path/to/rules --ai-root /path/to/ai \
  --training-root /path/to/training --rf-model models/nova_mode_classifier.joblib \
  --out-root outputs/review_continuum_noise_v10_20260910
Use verify to reuse staged runs; publish installs verified outputs with backups.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from make_tae_like_list import _inspect_mode_file, _load_gap_data
from input_validity import load_input_validity_registry
from tae_eae_features import classify_gap_region
from tae_rule_engine import (
    evaluate_mode,
    ContinuumCrossingConfig,
    BAD_EXTENDED_CONTINUUM_NOISE,
)
from continuum_noise import ContinuumNoiseThresholds
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_io import (
    sha256_file,
    input_fingerprint,
    datcon_path_for_mode,
    stable_json,
)

spec = importlib.util.spec_from_file_location(
    "publisher",
    HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py",
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
TARGET = "nstxuE204186A01t020/N10/egn10w.1271E+02"
TRAINING_TARGET = "nstxuE204669M03t025/N10/egn10w.1295E+02"
BASELINE = HERE.parent / "axis_amplitude_20260909"
TRAINING_BASELINE = REPO / "outputs/review_axis_energy_v9_20260909/training_comparison.csv"
CALIBRATION = REPO / "outputs/review_continuum_noise_20260910/training"
REGISTRY = load_input_validity_registry()


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def keyed(path):
    rows = read(path)
    result = {"/".join(Path(r["path"]).parts[-3:]): r for r in rows}
    assert len(result) == len(rows)
    return result


def shots():
    return [r["shot"] for r in read(BASELINE / "adopted_shot_summary.csv")]


def run_one(task):
    shot, args = task
    original = read(args.rules_root / shot / "all_modes_rules.csv")[0]
    cmd = [
        sys.executable,
        str(REPO / "scripts/sort_shot_mixed.py"),
        "--method",
        "rules",
        "--shot_dir",
        str(Path(original["path"]).parents[1]),
        "--out_dir",
        str(args.out_root / "rules" / shot),
        "--rf_model",
        str(args.rf_model),
    ]
    with (args.out_root / "logs" / (shot + ".log")).open("w") as f:
        subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("Regenerated " + shot, flush=True)


def stage(args):
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(exist_ok=True)
    paths = [
        REPO / p
        for p in (
            "scripts/tae_rule_engine.py",
            "scripts/tae_rule_config.py",
            "scripts/sort_shot_rules.py",
            "scripts/sort_shot_mixed.py",
            "scripts/make_tae_like_list.py",
            "scripts/tae_rule_io.py",
            "src/cont_features.py",
            "src/tae_eae_features.py",
            "src/input_validity.py",
            "src/nova_mode_loader.py",
            "src/mode_features.py",
            "configs/known_invalid_inputs.csv",
            "configs/rules/tae_rules_production_v10.yaml",
            "src/continuum_noise.py",
            "configs/rules/tae_rules_production_v9.yaml",
            "training_labels/tae_like_train.csv",
            "audits/continuum_noise_20260910/adopt_gate.py",
            "audits/axis_amplitude_20260909/current_disagreements.csv",
            "audits/axis_amplitude_20260909/adopted_shot_summary.csv",
            "outputs/review_axis_energy_v9_20260909/training_comparison.csv",
            "outputs/review_continuum_noise_20260910/training/measurements.jsonl",
            "outputs/review_continuum_noise_20260910/training/summary.json",
        )
    ]
    paths.append(args.rf_model.resolve())
    snapshot = dict(
        source_sha256={str(p): sha256_file(p) for p in paths},
        old_trees={s: publisher.tree_digest(args.rules_root / s) for s in shots()},
        ai_trees={s: publisher.tree_digest(args.ai_root / s) for s in shots()},
    )
    p = args.out_root / "run_inputs.json"
    assert not p.exists(), "Use verify to reuse completed staged runs"
    p.write_text(json.dumps(snapshot, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_one, [(s, args) for s in shots()]))
    verify(args)


def training_one(task):
    supplied, root, baseline, measured = task
    path = root / supplied["path"]
    fp = input_fingerprint(path, datcon_path_for_mode(path))
    assert fp == baseline["input_fingerprint"] == measured["input_fingerprint"]
    assert supplied["validity"] == baseline["training_label"]
    row = dict(
        mode_key=supplied["path"],
        input_fingerprint=fp,
        training_label=supplied["validity"],
        gate_candidate=False,
    )
    bundle, reason, message = _inspect_mode_file(
        path, expected_n=int(path.parent.name[1:])
    )
    if bundle is None:
        assert supplied["path"] == "nstxuG121123K51/N4/egn04w.8769E+01", (
            path,
            reason,
            message,
        )
        return dict(row, before="INVALID", after="INVALID", reason=reason)
    assert bundle["nr"] == 201
    assert REGISTRY.diagnostic(path.parents[1].name, bundle["ntor"]) is None
    gap, reason, message = _load_gap_data(
        path, mode=bundle["mode"], omega=bundle["omega"]
    )
    assert gap is not None, (path, reason, message)
    region = classify_gap_region(**gap.scalars)
    if region == "eae_like":
        return dict(row, before="ROUTED_EAE", after="ROUTED_EAE", reason="ROUTED_EAE")
    evidence = dict(
        path=str(path),
        mode_key=supplied["path"],
        shot=path.parents[1].name,
        ntor=bundle["ntor"],
        omega=bundle["omega"],
        gamma_d=bundle["gamma_d"],
        gap_region=region,
        input_fingerprint=fp,
    )
    kwargs = dict(
        mode=bundle["mode"],
        low2=gap.low2,
        high2=gap.high2,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
    )
    before = evaluate_mode(
        evidence,
        **kwargs,
        continuum_noise_config=ContinuumNoiseThresholds(top2_min=None)
    )
    after = evaluate_mode(evidence, **kwargs)
    assert before.decision in {"REVIEW", "BAD"} and after.decision in {"REVIEW", "BAD"}
    prior_f = json.loads(stable_json(before.features))
    current_f = json.loads(stable_json(after.features))
    prior_f["numerical_structure_features"].pop("extended_continuum_noise")
    gate = current_f["numerical_structure_features"].pop("extended_continuum_noise")
    # All v1 measured quantities must survive the length-only conversion exactly.
    legacy = json.loads(stable_json(measured["features"]))
    for key in ("schema_version", "calibrated_n_radial", "resolution_eligible"):
        legacy.pop(key)
    current = {k: gate[k] for k in legacy}
    current["records"] = [{k: v for k, v in r.items() if k != "hf_out_radial_length"}
                          for r in current["records"]]
    assert current == legacy, supplied["path"]
    assert ("GOOD" if before.decision == "REVIEW" else "BAD") == baseline["after"]
    assert prior_f == current_f, supplied["path"]
    if before.decision != after.decision:
        assert (
            before.decision == "REVIEW"
            and after.primary_reason == BAD_EXTENDED_CONTINUUM_NOISE
        )
    else:
        assert before.primary_reason == after.primary_reason
    return dict(
        row,
        before="GOOD" if before.decision == "REVIEW" else "BAD",
        after="GOOD" if after.decision == "REVIEW" else "BAD",
        reason=after.primary_reason,
        gate_candidate=gate["candidate_found"],
    )


def verify_generation_sources(snapshot):
    # Keep the stage-driver hash in the receipt as history. Verification logic
    # can evolve to account for documented missing INVALID inputs; sorter,
    # feature, model, registry, and baseline bytes must still match the stage.
    assert all(sha256_file(Path(p)) == h for p, h in snapshot["source_sha256"].items()
               if Path(p).resolve() != Path(__file__).resolve())


def verify(args):
    snapshot = json.loads((args.out_root / "run_inputs.json").read_text())
    verify_generation_sources(snapshot)
    changes, summaries, disagreements, newly_absent = [], [], [], []
    counts = Counter()
    new_trees = {}
    absent_ai_keys = {
        r["mode_key"] for r in read(HERE.parent / "extremum_floor_20260909/absent_invalid_inputs.csv")
    }
    for shot in shots():
        old_dir, new_dir = args.rules_root / shot, args.out_root / "rules" / shot
        assert publisher.tree_digest(old_dir) == snapshot["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == snapshot["ai_trees"][shot]
        old, new = keyed(old_dir / "all_modes_rules.csv"), keyed(
            new_dir / "all_modes_rules.csv"
        )
        ai = keyed(args.ai_root / shot / "all_modes_scored.csv")
        assert new.keys() <= old.keys() and new.keys() <= ai.keys(), shot
        missing = old.keys() - new.keys()
        for key in sorted(missing):
            prior = old[key]
            assert shot == "nstxuG142301C50" and prior["n"] == "1", key
            assert prior["final_decision"] == "INVALID", key
            assert prior["preprocessing_primary_reason"] == "KNOWN_INVALID_INPUT", key
            assert not Path(prior["path"]).exists(), key
            assert REGISTRY.diagnostic(shot, 1), key
            newly_absent.append(dict(mode_key=key, path=prior["path"],
                input_fingerprint=prior["input_fingerprint"], previous_decision="INVALID",
                previous_reason=prior["preprocessing_primary_reason"],
                current_status="ABSENT_RAW_INPUT"))
        assert ai.keys() - new.keys() == missing | {
            k for k in absent_ai_keys if k.startswith(shot + "/")
        }, shot
        for key, row in new.items():
            prev = old[key]
            counts[row["processing_status"]] += 1
            if row["processing_status"] != "RULE_EVALUATED":
                assert row == prev, (key, "non-evaluated row changed")
                continue
            f, old_f = json.loads(row["rule_features"]), json.loads(
                prev["rule_features"]
            )
            gate = f["numerical_structure_features"].pop("extended_continuum_noise")
            assert gate["gate_enabled"] and gate["thresholds"] == {
                "top2_min": .01, "local_min": .2, "radial_length_min": .04}
            assert gate["n_radial"] == 201
            assert gate["candidate_found"] == any(
                all(r[name] is not None and r[name] >= cut for name, cut in (
                    ("hf_out_top2_ratio", .01), ("hf_out_local_fraction", .2),
                    ("hf_out_radial_length", .04))) for r in gate["records"])
            f["feature_schema_version"] = old_f["feature_schema_version"]
            assert f == old_f, (key, "prior feature changed")
            allowed = {"rule_features", "rule_version"}
            if row["final_decision"] != prev["final_decision"]:
                assert (
                    key == TARGET
                    and prev["final_decision"] == "GOOD"
                    and row["final_decision"] == "BAD"
                )
                assert row["rule_primary_reason"] == BAD_EXTENDED_CONTINUUM_NOISE
                changes.append(
                    dict(
                        mode_key=key,
                        input_fingerprint=row["input_fingerprint"],
                        before="GOOD",
                        after="BAD",
                        reason=row["rule_primary_reason"],
                        hf_out_top2_ratio=gate["witness"]["hf_out_top2_ratio"],
                        hf_out_local_fraction=gate["witness"]["hf_out_local_fraction"],
                        hf_out_radial_length=gate["witness"]["hf_out_radial_length"],
                    )
                )
                allowed |= {
                    "rule_decision",
                    "rule_primary_reason",
                    "rule_triggered_rules",
                    "rule_survivor_accepted",
                    "final_decision",
                    "decision_source",
                    "selected_final",
                }
            assert {k: v for k, v in row.items() if k not in allowed} == {
                k: v for k, v in prev.items() if k not in allowed
            }, key
            if row["final_decision"] != ai[key]["final_label"].upper():
                a = ai[key]
                disagreements.append(
                    dict(
                        shot=shot,
                        mode_key=key,
                        input_fingerprint=row["input_fingerprint"],
                        rules_decision=row["final_decision"],
                        rules_reason=row["rule_primary_reason"],
                        rf_cnn_decision=a["final_label"].upper(),
                        p_rf_good=a["p_rf_good"],
                        p_cnn_good=a["p_cnn_good"],
                        rf_cnn_tier=a["tier"],
                    )
                )
        s = read(new_dir / "shot_summary_wide.csv")[0]
        assert s["rule_configuration_sha256"] == PRODUCTION_RULE_CONFIG_SHA256
        assert (
            "FALLBACK" not in s["duplicate_processing_status"]
            and "NO_RF" not in s["duplicate_processing_status"]
        )
        assert s["extended_continuum_noise_gate_enabled"] == "True"
        summaries.append(
            dict(
                shot=shot,
                inputs=s["n_total_files"],
                invalid=s["n_invalid"],
                good_before_dedup=s["n_final_good_before_clustering"],
                selected_good=s["n_final_good"],
            )
        )
        new_trees[shot] = publisher.tree_digest(new_dir)
    assert len(changes) == 1
    assert len(newly_absent) == 65
    assert counts == {
        "RULE_EVALUATED": 4187,
        "ROUTED_EAE": 14358,
        "INVALID": 707,
    }, counts
    assert sum(int(r["good_before_dedup"]) for r in summaries) == 948
    assert sum(int(r["selected_good"]) for r in summaries) == 942
    print(
        "Verified all 27 exports, unchanged prior features, and the one expected decision change. Checking training.",
        flush=True,
    )
    baseline = {r["mode_key"]: r for r in read(TRAINING_BASELINE)}
    with (CALIBRATION / "measurements.jsonl").open() as stream:
        measured = {r["mode_key"]: r for r in map(json.loads, stream)}
    measurement_receipt = json.loads((CALIBRATION / "summary.json").read_text())
    assert sha256_file(CALIBRATION / "measurements.jsonl") == measurement_receipt["measurements_sha256"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        training = list(pool.map(training_one,
            [(r, args.training_root, baseline[r["path"]], measured[r["path"]])
             for r in read(REPO / "training_labels/tae_like_train.csv")], chunksize=8))
    assert len(training) == len(baseline) == len(measured) == 2390
    assert all(r["before"] == baseline[r["mode_key"]]["after"] for r in training)
    assert sum(r["gate_candidate"] for r in training if r["training_label"] == "good") == 0
    assert sum(r["gate_candidate"] for r in training if r["training_label"] == "bad") == 87
    before_matrix = Counter(r["training_label"] + ":" + r["before"] for r in training)
    after_matrix = Counter(r["training_label"] + ":" + r["after"] for r in training)
    assert before_matrix == {
        "good:GOOD": 542,
        "good:BAD": 33,
        "bad:GOOD": 25,
        "bad:BAD": 1763,
        "bad:ROUTED_EAE": 26,
        "bad:INVALID": 1,
    }
    training_changes = [r for r in training if r["before"] != r["after"]]
    assert [r["mode_key"] for r in training_changes] == [TRAINING_TARGET], training_changes
    assert after_matrix == dict(before_matrix, **{"bad:GOOD": 24, "bad:BAD": 1764})
    previous = {r["mode_key"]: r for r in read(BASELINE / "current_disagreements.csv")}
    current = {r["mode_key"]: r for r in disagreements}
    assert current.keys() == previous.keys() - {TARGET} and len(current) == 226
    assert all(previous[k] == current[k] for k in current)
    write(HERE / "newly_absent_invalid_inputs.csv", newly_absent, list(newly_absent[0]))
    write(HERE / "adopted_changes.csv", changes, list(changes[0]))
    write(HERE / "adopted_shot_summary.csv", summaries, list(summaries[0]))
    write(HERE / "training_changes.csv", training_changes, list(training[0]))
    write(args.out_root / "training_comparison.csv", training, list(training[0]))
    write(
        HERE / "current_disagreements.csv",
        sorted(disagreements, key=lambda r: r["mode_key"]),
        list(disagreements[0]),
    )
    write(
        HERE / "disagreements_removed.csv", [previous[TARGET]], list(previous[TARGET])
    )
    verify_generation_sources(snapshot)
    receipt = dict(
        status="verified",
        verification_driver_sha256=sha256_file(Path(__file__)),
        newly_absent_invalid_inputs=len(newly_absent),
        absent_ai_inputs=len(absent_ai_keys) + len(newly_absent),
        config=PRODUCTION_RULE_CONFIG_NAME,
        config_sha256=PRODUCTION_RULE_CONFIG_SHA256,
        counts=dict(counts),
        good_before_dedup=948,
        selected_good=942,
        shot_decision_changes=1,
        training_rows=2390,
        training_before=dict(before_matrix),
        training_after=dict(after_matrix),
        training_decision_changes=len(training_changes),
        training_good_flagged=0,
        training_bad_flagged=87,
        v1_calibration_values_unchanged=True,
        disagreements=226,
        removed_disagreements=1,
        new_trees=new_trees,
        **snapshot
    )
    (HERE / "adoption_verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("Verified training: " + json.dumps(dict(after_matrix)), flush=True)


def publish(args):
    receipt = json.loads((HERE / "adoption_verification.json").read_text())
    assert receipt["status"] == "verified"
    verify_generation_sources(receipt)
    assert sha256_file(Path(__file__)) == receipt["verification_driver_sha256"]
    backup_name = "before_continuum_noise_v10_20260910"
    plans = []
    for shot in shots():
        source, target = args.out_root / "rules" / shot, args.rules_root / shot
        backup, staging = (
            args.rules_root / backup_name / shot,
            args.rules_root / (".staging_" + backup_name) / shot,
        )
        assert not backup.exists() and not staging.exists()
        assert publisher.tree_digest(source) == receipt["new_trees"][shot]
        assert publisher.tree_digest(target) == receipt["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == receipt["ai_trees"][shot]
        plans.append((shot, source, target, backup, staging))
    for shot, source, target, backup, staging in plans:
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, staging)
        assert publisher.tree_digest(staging) == receipt["new_trees"][shot]
    records = []
    for shot, source, target, backup, staging in plans:
        assert publisher.tree_digest(target) == receipt["old_trees"][shot]
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)
        try:
            staging.rename(target)
        except Exception:
            backup.rename(target)
            raise
        assert publisher.tree_digest(target) == receipt["new_trees"][shot]
        assert publisher.tree_digest(backup) == receipt["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == receipt["ai_trees"][shot]
        records.append(
            dict(
                shot=shot,
                output=str(target),
                backup=str(backup),
                new=receipt["new_trees"][shot],
                old=receipt["old_trees"][shot],
            )
        )
    (args.rules_root / (".staging_" + backup_name)).rmdir()
    (HERE / "publication.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        "Published all 27 v10 rules exports with verified backups; RF-CNN outputs unchanged."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["stage", "verify", "publish"])
    for name in ("rules-root", "ai-root", "training-root", "rf-model", "out-root"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    globals()[args.phase](args)
