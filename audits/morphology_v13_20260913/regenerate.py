"""Regenerate, verify and install the approved v13 morphology changes.

Example: python audits/morphology_v13_20260913/regenerate.py stage
  --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai
  --training-root /path/to/training --ditw-root /path/to/DiTw
  --out-root outputs/review_morphology_v13_20260913
Use publish only after stage has written its complete verification receipt.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
from tae_eae_features import classify_gap_region
from tae_rule_engine import evaluate_mode, ContinuumCrossingConfig
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_io import sha256_file, input_fingerprint, datcon_path_for_mode, portable_mode_key

csv.field_size_limit(10**8)
spec = importlib.util.spec_from_file_location("publisher", HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
TRAINING_BASELINE = REPO / "outputs/review_distributed_noise_v12_20260913/training_comparison.csv"
CHANGED_GATES = {"BAD_EDGE_SPIKE", "BAD_INTERIOR_UNRESOLVED_ENVELOPE"}
EXPECTED_RESCUED = {
    "nstxuE202926A03t025/N3/egn03w.1151E+02",
    "nstxuE202926A03t025/N4/egn04w.3735E+01",
    "nstxuG142301L94/N9/egn09w.2746E+02",
}
EXPECTED_REJECTED = {
    "nstxuE203655F01t025/N2/egn02w.2035E+02",
    "nstxuE203655F01t025/N3/egn03w.1987E+02",
    "nstxuE205057A01t020/N4/egn04w.1444E+02",
}


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def keyed(path):
    rows = read(path)
    result = {r.get("mode_key") or portable_mode_key(r["path"]): r for r in rows}
    assert len(result) == len(rows), path
    return result


def write(path, rows, fields=None):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def shots():
    return sorted(r["shot"] for r in read(REPO / "audits/main_dataset_shots/shot_status.csv") if r["post_training_checked"] == "yes")


def source_hashes():
    files = [p for folder in ("src", "scripts") for p in (REPO / folder).glob("*.py")]
    files += list((REPO / "configs/rules").glob("*.yaml"))
    files += [REPO / p for p in (
        "configs/known_invalid_inputs.csv", "training_labels/tae_like_train.csv",
        "audits/main_dataset_shots/shot_status.csv", "audits/pilot12_v11_20260910/selection.csv")]
    files += [Path(__file__), TRAINING_BASELINE, Path(publisher.__file__)]
    return {str(p): sha256_file(p) for p in sorted(files)}


def run_one(task):
    shot, args = task
    output = args.out_root / "rules" / shot
    assert not output.exists(), output
    command = [sys.executable, str(REPO / "scripts/sort_shot_mixed.py"), "--method", "rules",
               "--shot_dir", str(args.ditw_root / shot), "--out_dir", str(output),
               "--rule_config", "tae_rules_production_v13"]
    override = args.rules_root / shot / "manual_overrides.csv"
    if override.exists() and read(override):
        command += ["--manual_overrides", str(override)]
    with (args.out_root / "logs" / (shot + ".log")).open("w") as f:
        subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("Regenerated " + shot, flush=True)


def training_one(task):
    old, root = task
    key = old["mode_key"]
    path = root / key
    fp = input_fingerprint(path, datcon_path_for_mode(path))
    assert fp == old["input_fingerprint"], key
    bundle, reason, message = _inspect_mode_file(path, expected_n=int(path.parent.name[1:]))
    row = dict(path=key, mode_key=key, training_label=old["training_label"], input_fingerprint=fp,
               before=old["after"], after=old["after"], before_reason=old["reason"], reason=old["reason"],
               footprint_exception_applied=False)
    if bundle is None:
        assert row["before"] == "INVALID", (key, reason, message)
        return row
    gap, reason, message = _load_gap_data(path, mode=bundle["mode"], omega=bundle["omega"])
    assert gap is not None, (key, reason, message)
    region = classify_gap_region(**gap.scalars)
    if region == "eae_like":
        assert row["before"] == "ROUTED_EAE", key
    else:
        evidence = dict(path=str(path), mode_key=key, shot=path.parents[1].name,
                        ntor=bundle["ntor"], omega=bundle["omega"], gamma_d=bundle["gamma_d"],
                        gap_region=region, input_fingerprint=fp)
        result = evaluate_mode(evidence, mode=bundle["mode"], low2=gap.low2, high2=gap.high2,
                               continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None))
        assert result.decision in ("BAD", "REVIEW"), (key, result.diagnostic_message)
        row.update(after="GOOD" if result.decision == "REVIEW" else "BAD", reason=result.primary_reason,
                   footprint_exception_applied=result.features["resolution_features"]["interior_unresolved_envelope"]["footprint_exception"]["applied"])
    assert input_fingerprint(path, datcon_path_for_mode(path)) == fp
    return row


def stage(args):
    assert not args.out_root.exists(), "Use verify for completed staging; otherwise choose a new out-root"
    assert len(shots()) == 39
    args.out_root.mkdir(parents=True)
    (args.out_root / "logs").mkdir()
    snapshot = dict(source_sha256=source_hashes(),
                    old_trees={s: publisher.tree_digest(args.rules_root / s) for s in shots()},
                    ai_trees={s: publisher.tree_digest(args.ai_root / s) for s in shots()})
    (args.out_root / "run_inputs.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print("Captured sources and all 39 rules/AI output trees", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_one, [(s, args) for s in shots()]))
    verify(args)


def compare_features(before, after, key):
    old = json.loads(before["rule_features"])
    new = json.loads(after["rule_features"])
    previous_gates = old["severity_features"]["gates"]
    current_gates = new["severity_features"]["gates"]
    assert previous_gates.keys() == current_gates.keys(), key
    for gate in current_gates.keys() - CHANGED_GATES:
        assert current_gates[gate] == previous_gates[gate], (key, gate)
    for field in ("edge_energy_local_peaks", "edge_body_r_max", "edge_body_amplitude_max"):
        new["boundary_features"]["edge_artifact"].pop(field)
    footprint = new["resolution_features"]["interior_unresolved_envelope"].pop("footprint_exception")
    for f in (old, new):
        f.pop("severity_features")
        f.pop("feature_schema_version")
    assert old == new, key
    assert after["severity_complete"] == "True", key
    return footprint


def verify(args):
    snapshot = json.loads((args.out_root / "run_inputs.json").read_text())
    assert source_hashes() == snapshot["source_sha256"]
    changes, reasons, selections, summaries, disagreements, removed, added, excluded_ai = [], [], [], [], [], [], [], []
    counts = Counter()
    new_trees = {}
    latest12 = {r["shot"] for r in read(REPO / "audits/pilot12_v11_20260910/selection.csv")}
    for shot in shots():
        assert publisher.tree_digest(args.rules_root / shot) == snapshot["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == snapshot["ai_trees"][shot]
        old = keyed(args.rules_root / shot / "all_modes_rules.csv")
        new_dir = args.out_root / "rules" / shot
        new = keyed(new_dir / "all_modes_rules.csv")
        ai = keyed(args.ai_root / shot / "all_modes_scored.csv")
        assert old.keys() == new.keys(), shot
        for key, r in new.items():
            before = old[key]
            assert r["input_fingerprint"] == before["input_fingerprint"], key
            path = args.ditw_root / key
            assert input_fingerprint(path, datcon_path_for_mode(path)) == r["input_fingerprint"], key
            for field in ("processing_status", "gap_region", "signed_delta", "fraction_below_upper2", "nr"):
                assert r[field] == before[field], (key, field)
            counts[r["processing_status"]] += 1
            if r["processing_status"] == "RULE_EVALUATED":
                footprint = compare_features(before, r, key)
                counts["footprint_exception_applied"] += footprint["applied"]
                counts["nr_" + r["nr"]] += 1
            if r["final_decision"] != before["final_decision"]:
                changes.append(dict(path=key, mode_key=key, shot=shot, latest12=shot in latest12,
                    before=before["final_decision"], after=r["final_decision"],
                    before_reason=before["rule_primary_reason"], reason=r["rule_primary_reason"]))
            elif r["rule_primary_reason"] != before["rule_primary_reason"]:
                assert r["rule_primary_reason"] in CHANGED_GATES or before["rule_primary_reason"] in CHANGED_GATES, key
                reasons.append(dict(path=key, decision=r["final_decision"],
                    before_reason=before["rule_primary_reason"], reason=r["rule_primary_reason"]))
            if r["selected_final"] != before["selected_final"]:
                selections.append(dict(path=key, before_selected=before["selected_final"],
                    after_selected=r["selected_final"], decision_changed=r["final_decision"] != before["final_decision"]))
            if r["processing_status"] != "RULE_EVALUATED":
                continue
            a = ai.get(key)
            if a is None or a["final_label"].upper() not in ("GOOD", "BAD"):
                excluded_ai.append(dict(path=key, reason="AI_INPUTS_NOT_CURRENT_OR_NOT_CLASSIFIED"))
                continue
            # Legacy AI rows have no fingerprint. Preserve the entire AI export
            # and require matching frequency, native grid and continuum routing.
            for field in ("omega", "nr", "signed_delta", "fraction_below_upper2"):
                assert float(a[field]) == float(r[field]), (key, field)
            assert a["gap_region"] == r["gap_region"], key
            counts["paired_tae_modes"] += 1
            comparison = dict(path=key, label=r["final_decision"].lower(), mode_key=key, shot=shot,
                input_fingerprint=r["input_fingerprint"], rules_decision=r["final_decision"],
                rules_reason=r["rule_primary_reason"], rf_cnn_decision=a["final_label"].upper(),
                p_rf_good=a["p_rf_good"], p_cnn_good=a["p_cnn_good"])
            was = before["final_decision"] != comparison["rf_cnn_decision"]
            now = r["final_decision"] != comparison["rf_cnn_decision"]
            counts["previous_disagreements"] += was
            counts["current_disagreements"] += now
            if shot in latest12:
                counts["latest12_previous_disagreements"] += was
                counts["latest12_current_disagreements"] += now
            if now:
                disagreements.append(comparison)
            if was and not now:
                removed.append(comparison)
            if now and not was:
                added.append(comparison)
        s = read(new_dir / "shot_summary_wide.csv")[0]
        assert s["rule_configuration_sha256"] == PRODUCTION_RULE_CONFIG_SHA256
        assert s["interior_envelope_footprint_exception_enabled"] == "True"
        assert not read(new_dir / "resolution_warnings.csv")
        assert all(r["cluster_status"] == "PROCESSED_RULE_SEVERITY" for r in read(new_dir / "frequency_clusters.csv"))
        summaries.append(dict(shot=shot, inputs=s["n_total_files"], invalid=s["n_invalid"],
            good_before_dedup=s["n_final_good_before_clustering"], selected_good=s["n_final_good"]))
        new_trees[shot] = publisher.tree_digest(new_dir)
        print("Verified " + shot, flush=True)
    assert {r["mode_key"] for r in changes if r["after"] == "GOOD"} == EXPECTED_RESCUED, changes
    assert {r["mode_key"] for r in changes if r["after"] == "BAD"} == EXPECTED_REJECTED, changes
    assert len(changes) == 6, changes
    with ProcessPoolExecutor(max_workers=4) as pool:
        training = list(pool.map(training_one, [(r, args.training_root) for r in read(TRAINING_BASELINE)], chunksize=4))
    training_changes = [r for r in training if r["before"] != r["after"]]
    assert [r["mode_key"] for r in training_changes] == ["nstxuE204669M03t025/N4/egn04w.1691E+02"], training_changes
    assert source_hashes() == snapshot["source_sha256"]
    write(args.out_root / "training_comparison.csv", training)
    write(HERE / "training_changes.csv", training_changes)
    write(HERE / "label_changes.csv", changes)
    write(HERE / "reason_changes.csv", reasons, ["path", "decision", "before_reason", "reason"])
    write(HERE / "selection_changes.csv", selections, ["path", "before_selected", "after_selected", "decision_changed"])
    write(HERE / "shot_summary.csv", summaries)
    fields = ["path", "label", "mode_key", "shot", "input_fingerprint", "rules_decision", "rules_reason", "rf_cnn_decision", "p_rf_good", "p_cnn_good"]
    for name, rows in (("disagreements", disagreements), ("disagreements_removed", removed), ("disagreements_added", added)):
        write(HERE / ("current_" + name + ".csv" if name == "disagreements" else name + ".csv"), rows, fields)
        write(HERE / ("latest12_" + name + ".csv"), [r for r in rows if r["shot"] in latest12], fields)
    write(HERE / "ai_comparison_excluded.csv", excluded_ai, ["path", "reason"])
    receipt = dict(status="verified", configuration=PRODUCTION_RULE_CONFIG_NAME,
        configuration_sha256=PRODUCTION_RULE_CONFIG_SHA256, **snapshot, counts=dict(counts), new_trees=new_trees,
        pilot_decision_changes=len(changes), primary_reason_only_changes=len(reasons),
        selection_changed_rows=len(selections), training_decision_changes=len(training_changes),
        training_after=dict(Counter(r["training_label"] + ":" + r["after"] for r in training)),
        selected_good=sum(int(r["selected_good"]) for r in summaries),
        good_before_dedup=sum(int(r["good_before_dedup"]) for r in summaries),
        training_comparison_sha256=sha256_file(args.out_root / "training_comparison.csv"))
    (HERE / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified " + json.dumps({k: receipt[k] for k in (
        "counts", "pilot_decision_changes", "primary_reason_only_changes", "selection_changed_rows", "training_decision_changes", "training_after")}), flush=True)


def publish(args):
    receipt = json.loads((HERE / "verification.json").read_text())
    assert receipt["status"] == "verified"
    assert source_hashes() == receipt["source_sha256"]
    backup_root = args.rules_root / "before_morphology_v13_20260913"
    staging_root = args.rules_root / ".staging_morphology_v13_20260913"
    assert not backup_root.exists() and not staging_root.exists()
    for shot in shots():
        assert publisher.tree_digest(args.out_root / "rules" / shot) == receipt["new_trees"][shot]
        assert publisher.tree_digest(args.rules_root / shot) == receipt["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == receipt["ai_trees"][shot]
    for shot in shots():
        shutil.copytree(args.out_root / "rules" / shot, staging_root / shot)
        assert publisher.tree_digest(staging_root / shot) == receipt["new_trees"][shot]
    backup_root.mkdir()
    published = []
    for shot in shots():
        target, backup = args.rules_root / shot, backup_root / shot
        assert publisher.tree_digest(target) == receipt["old_trees"][shot]
        target.rename(backup)
        try:
            (staging_root / shot).rename(target)
        except Exception:
            backup.rename(target)
            raise
        assert publisher.tree_digest(target) == receipt["new_trees"][shot]
        assert publisher.tree_digest(backup) == receipt["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == receipt["ai_trees"][shot]
        published.append(dict(shot=shot, output=str(target), backup=str(backup)))
        print("Installed " + shot, flush=True)
    staging_root.rmdir()
    (HERE / "publication.json").write_text(json.dumps(published, indent=2) + "\n")
    print("Published 39 verified v13 rules exports with v12 backups; AI outputs unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "verify", "publish"))
    for name in ("rules-root", "ai-root", "out-root", "training-root", "ditw-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args)
