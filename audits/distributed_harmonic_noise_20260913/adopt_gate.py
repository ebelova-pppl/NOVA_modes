"""Stage, verify and publish the approved v12 distributed-harmonic noise gate.

Example: python audits/distributed_harmonic_noise_20260913/adopt_gate.py stage
  --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai
  --training-root /path/to/training --ditw-root /path/to/DiTw
  --out-root outputs/review_distributed_noise_v12_20260913
Publish requires the completed verification receipt and preserves all v11 exports.
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

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from distributed_harmonic_noise import extract_distributed_noise_features, BAD_DISTRIBUTED_HARMONIC_NOISE as REASON
from make_tae_like_list import _inspect_mode_file, _load_gap_data
from tae_eae_features import classify_gap_region
from tae_rule_engine import evaluate_mode, ContinuumCrossingConfig
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_io import sha256_file, input_fingerprint, datcon_path_for_mode, portable_mode_key

csv.field_size_limit(10**8)
spec = importlib.util.spec_from_file_location("publisher", HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
EVIDENCE = HERE / "adoption"
PROPOSAL = REPO / "outputs/review_distributed_harmonic_noise_top2_pilot39_20260913"


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def keyed(path):
    rows = read(path)
    result = {r.get("mode_key") or portable_mode_key(r["path"]): r for r in rows}
    assert len(result) == len(rows)
    return result


def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def shots():
    return sorted(r["shot"] for r in read(REPO / "audits/main_dataset_shots/shot_status.csv") if r["post_training_checked"] == "yes")


def source_hashes():
    files = [REPO / p for p in (
        "src/distributed_harmonic_noise.py", "src/continuum_noise.py", "src/rule_severity.py",
        "scripts/tae_rule_engine.py", "scripts/tae_rule_config.py", "scripts/tae_rule_io.py",
        "scripts/sort_shot_rules.py", "scripts/sort_shot_mixed.py", "scripts/make_tae_like_list.py",
        "src/nova_mode_loader.py", "src/mode_features.py", "src/cont_features.py",
        "src/tae_eae_features.py", "src/input_validity.py", "configs/known_invalid_inputs.csv",
        "configs/rules/tae_rules_production_v11.yaml", "configs/rules/tae_rules_production_v12.yaml",
        "training_labels/tae_like_train.csv", "audits/main_dataset_shots/shot_status.csv")]
    files += [Path(__file__), PROPOSAL / "measurements.csv", PROPOSAL / "summary.json"]
    return {str(p): sha256_file(p) for p in files}


def run_one(task):
    shot, args = task
    output = args.out_root / "rules" / shot
    assert not output.exists(), output
    command = [sys.executable, str(REPO / "scripts/sort_shot_mixed.py"), "--method", "rules",
               "--shot_dir", str(args.ditw_root / shot), "--out_dir", str(output),
               "--rule_config", "tae_rules_production_v12"]
    override = args.rules_root / shot / "manual_overrides.csv"
    if override.exists() and read(override):
        command += ["--manual_overrides", str(override)]
    with (args.out_root / "logs" / (shot + ".log")).open("w") as f:
        subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("Regenerated " + shot, flush=True)


def check_measurement(features, old):
    assert features["candidate_found"] == (old["candidate"] == "True"), old["mode_key"]
    w = features["witness"]
    if w is None:
        assert float(old["minimum_cut_ratio"]) == 0
    else:
        for key in ("hf_top2_ratio", "hf_total_fraction", "hf_window_fraction", "effective_length", "minimum_cut_ratio"):
            assert np.isclose(w[key], float(old[key]), rtol=1e-11, atol=1e-15), (old["mode_key"], key, w[key], old[key])


def training_one(task):
    old, root = task
    key = old["mode_key"]; path = root / key
    fp = input_fingerprint(path, datcon_path_for_mode(path))
    assert fp == old["input_fingerprint"]
    bundle, reason, message = _inspect_mode_file(path, expected_n=int(path.parent.name[1:]))
    row = dict(path=key, mode_key=key, training_label=old["training_label"], input_fingerprint=fp,
               before=old["baseline_decision"], after=old["baseline_decision"], reason=old["baseline_reason"], candidate=False)
    if bundle is None:
        assert old["status"] == "INVALID", (key, reason, message)
        return row
    features = extract_distributed_noise_features(bundle["mode"])
    check_measurement(features, old)
    row["candidate"] = features["candidate_found"]
    gap, reason, message = _load_gap_data(path, mode=bundle["mode"], omega=bundle["omega"])
    assert gap is not None, (key, reason, message)
    region = classify_gap_region(**gap.scalars)
    if region == "eae_like":
        assert row["before"] == "ROUTED_EAE"
    else:
        evidence = dict(path=str(path), mode_key=key, shot=path.parents[1].name,
                        ntor=bundle["ntor"], omega=bundle["omega"], gamma_d=bundle["gamma_d"],
                        gap_region=region, input_fingerprint=fp)
        result = evaluate_mode(evidence, mode=bundle["mode"], low2=gap.low2, high2=gap.high2,
                               continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None))
        assert result.decision in ("BAD", "REVIEW"), (key, result.diagnostic_message)
        assert result.features["numerical_structure_features"]["distributed_harmonic_noise"] == features
        row.update(after="GOOD" if result.decision == "REVIEW" else "BAD", reason=result.primary_reason)
        expected = "BAD" if old["baseline_decision"] == "GOOD" and row["candidate"] else old["baseline_decision"]
        assert row["after"] == expected, row
        if row["before"] == row["after"]:
            assert row["reason"] == old["baseline_reason"], row
    assert input_fingerprint(path, datcon_path_for_mode(path)) == fp
    return row


def stage(args):
    assert not args.out_root.exists(), "Use verify for completed staging; otherwise choose a new out-root"
    assert len(shots()) == 39
    EVIDENCE.mkdir(exist_ok=True)
    args.out_root.mkdir(parents=True)
    (args.out_root / "logs").mkdir()
    snapshot = dict(source_sha256=source_hashes(),
                    old_trees={s: publisher.tree_digest(args.rules_root / s) for s in shots()},
                    ai_trees={s: publisher.tree_digest(args.ai_root / s) for s in shots()})
    (args.out_root / "run_inputs.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_one, [(s, args) for s in shots()]))
    verify(args)


def verify(args):
    snapshot = json.loads((args.out_root / "run_inputs.json").read_text())
    assert source_hashes() == snapshot["source_sha256"]
    proposal = [r for r in read(PROPOSAL / "measurements.csv")]
    pilot_proposal = {r["mode_key"]: r for r in proposal if r["cohort"] == "pilot"}
    changes, selections, summaries, disagreements, removed, added, excluded_ai = [], [], [], [], [], [], []
    counts = Counter(); new_trees = {}
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
                expected = pilot_proposal[key]
                prior_f, new_f = json.loads(before["rule_features"]), json.loads(r["rule_features"])
                gate = new_f.pop("numerical_structure_features").copy()
                noise = gate.pop("distributed_harmonic_noise")
                assert gate == prior_f.pop("numerical_structure_features"), key
                check_measurement(noise, expected)
                for f in (prior_f, new_f):
                    f.pop("severity_features"); f.pop("feature_schema_version")
                assert prior_f == new_f, key
                assert r["severity_complete"] == "True", key
                assert (float(r["gate_severity_" + REASON]) > 1) == noise["candidate_found"], key
                expected_decision = "BAD" if before["final_decision"] == "GOOD" and noise["candidate_found"] else before["final_decision"]
                assert r["final_decision"] == expected_decision, key
            if r["final_decision"] != before["final_decision"]:
                assert r["rule_primary_reason"] == REASON
                changes.append(dict(path=key, mode_key=key, before=before["final_decision"], after=r["final_decision"], reason=r["rule_primary_reason"]))
            else:
                assert r["rule_primary_reason"] == before["rule_primary_reason"], key
            if r["selected_final"] != before["selected_final"]:
                selections.append(dict(path=key, before_selected=before["selected_final"], after_selected=r["selected_final"], decision_changed=r["final_decision"] != before["final_decision"]))
            if r["processing_status"] != "RULE_EVALUATED":
                continue
            a = ai.get(key)
            if a is None or a["final_label"].upper() not in ("GOOD", "BAD"):
                excluded_ai.append(dict(path=key, reason="AI_INPUTS_NOT_CURRENT_OR_NOT_CLASSIFIED"))
                continue
            # Legacy AI CSVs lack per-row fingerprints. Preserve their verified
            # output trees and require matching frequency, resolution and routing.
            # Corrected C50/N1 has no usable AI classification and is excluded above.
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
            if now: disagreements.append(comparison)
            if was and not now: removed.append(comparison)
            if now and not was: added.append(comparison)
        s = read(new_dir / "shot_summary_wide.csv")[0]
        assert s["rule_configuration_sha256"] == PRODUCTION_RULE_CONFIG_SHA256
        assert s["distributed_harmonic_noise_gate_enabled"] == "True"
        assert not read(new_dir / "resolution_warnings.csv")
        assert all(r["cluster_status"] == "PROCESSED_RULE_SEVERITY" for r in read(new_dir / "frequency_clusters.csv"))
        summaries.append(dict(shot=shot, inputs=s["n_total_files"], invalid=s["n_invalid"],
                              good_before_dedup=s["n_final_good_before_clustering"], selected_good=s["n_final_good"]))
        new_trees[shot] = publisher.tree_digest(new_dir)
    assert [r["mode_key"] for r in changes] == ["nstxuE205042A01t022/N10/egn10w.3470E+01"], changes
    with ProcessPoolExecutor(max_workers=4) as pool:
        training = list(pool.map(training_one, [(r, args.training_root) for r in proposal if r["cohort"] == "training"], chunksize=4))
    training_changes = [r for r in training if r["before"] != r["after"]]
    assert [r["mode_key"] for r in training_changes] == ["nstxuG121123J38/N8/egn08w.2222E+02"], training_changes
    assert source_hashes() == snapshot["source_sha256"]
    write(args.out_root / "training_comparison.csv", training)
    write(EVIDENCE / "training_changes.csv", training_changes)
    write(EVIDENCE / "adopted_changes.csv", changes)
    write(EVIDENCE / "selection_changes.csv", selections, ["path", "before_selected", "after_selected", "decision_changed"])
    write(EVIDENCE / "shot_summary.csv", summaries)
    fields = ["path", "label", "mode_key", "shot", "input_fingerprint", "rules_decision", "rules_reason", "rf_cnn_decision", "p_rf_good", "p_cnn_good"]
    write(EVIDENCE / "current_disagreements.csv", disagreements, fields)
    write(EVIDENCE / "disagreements_removed.csv", removed, fields)
    write(EVIDENCE / "disagreements_added.csv", added, fields)
    write(EVIDENCE / "ai_comparison_excluded.csv", excluded_ai, ["path", "reason"])
    receipt = dict(status="verified", configuration=PRODUCTION_RULE_CONFIG_NAME,
                    configuration_sha256=PRODUCTION_RULE_CONFIG_SHA256, **snapshot,
                    counts=dict(counts), new_trees=new_trees, pilot_decision_changes=len(changes),
                    selection_changed_rows=len(selections), training_decision_changes=len(training_changes),
                    flagged_training_labels=dict(Counter(r["training_label"] for r in training if r["candidate"])),
                    training_after=dict(Counter(r["training_label"] + ":" + r["after"] for r in training)),
                    selected_good=sum(int(r["selected_good"]) for r in summaries),
                    good_before_dedup=sum(int(r["good_before_dedup"]) for r in summaries),
                    training_comparison_sha256=sha256_file(args.out_root / "training_comparison.csv"))
    (EVIDENCE / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified " + json.dumps({k: receipt[k] for k in ("counts", "pilot_decision_changes", "selection_changed_rows", "training_decision_changes", "flagged_training_labels")}), flush=True)


def publish(args):
    receipt = json.loads((EVIDENCE / "verification.json").read_text())
    assert source_hashes() == receipt["source_sha256"]
    backup_root = args.rules_root / "before_distributed_noise_v12_20260913"
    staging_root = args.rules_root / ".staging_distributed_noise_v12_20260913"
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
    staging_root.rmdir()
    (EVIDENCE / "publication.json").write_text(json.dumps(published, indent=2) + "\n")
    print("Published 39 verified v12 rules exports with v11 backups; AI outputs unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "verify", "publish"))
    for name in ("rules-root", "ai-root", "out-root", "training-root", "ditw-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args)
