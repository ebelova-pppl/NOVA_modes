"""Apply Elena's completed 39-shot review using canonical manual overrides.

Example: python audits/pilot39_manual_review_20260914/apply_review.py stage
  --data-root /path/to/DiTw --rules-root /path/to/sort_outputs
  --ai-root /path/to/sort_outputs_ai --runtime-dir outputs/review_manual39_new
The publish phase requires verified staging and preserves previous shot trees.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from sort_shot_rules import load_manual_overrides
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_io import MANUAL_OVERRIDE_FIELDS, datcon_path_for_mode, input_fingerprint, portable_mode_key, sha256_file

csv.field_size_limit(10**8)
spec = importlib.util.spec_from_file_location("publisher", HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
WORKSHEET = HERE.parent / "pilot39_v13_review_20260913/disagreements_elena.csv"
BASELINE = WORKSHEET.with_name("disagreements.csv")
INVENTORY = HERE.parent / "main_dataset_shots/shot_status.csv"
FIELDS = ["path", "label", "shot", "ntor", "omega", "gap_region", "final_decision", "final_reason",
          "automatic_final_decision", "original_rule_decision", "rule_primary_reason",
          "decision_source", "manual_decision", "manual_reason", "reviewer", "adjudication_timestamp",
          "selected_final", "overall_rule_severity", "input_fingerprint", "mode_key"]
COMPARE_FIELDS = FIELDS + ["rf_cnn_decision", "rf_decision", "cnn_decision", "p_rf_good", "p_cnn_good"]
MUTABLE = {"manual_decision", "manual_reason", "reviewer", "adjudication_timestamp", "override_status",
           "override_message", "final_decision", "decision_source", "rule_survivor_accepted",
           "selected_final", "duplicate_rank_score", "duplicate_rank_source"}


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def keyed(path):
    rows = read(path)
    result = {r.get("mode_key") or portable_mode_key(r["path"]): r for r in rows}
    assert len(result) == len(rows), path
    return result


def write(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def source_hashes():
    files = [p for folder in ("src", "scripts") for p in (REPO / folder).glob("*.py")]
    files += list((REPO / "configs/rules").glob("*.yaml"))
    files += [WORKSHEET, BASELINE, INVENTORY, Path(__file__), Path(publisher.__file__),
              REPO / "configs/known_invalid_inputs.csv", REPO / "training_labels/tae_like_train.csv"]
    return {str(p): sha256_file(p) for p in sorted(files)}


def stage(args):
    assert not args.runtime_dir.exists(), "Use verify for completed staging; otherwise choose an empty destination"
    assert not (HERE / "manual_overrides.csv").exists(), "Preserve the existing adjudication receipt"
    args.runtime_dir.mkdir(parents=True)
    for folder in ("logs", "overrides"):
        (args.runtime_dir / folder).mkdir()
    shots = sorted(r["shot"] for r in read(INVENTORY) if r["post_training_checked"] == "yes")
    rows, original = read(WORKSHEET), keyed(BASELINE)
    assert len(shots) == 39 and len(rows) == len(original) == 365
    assert {r["mode_key"] for r in rows} == original.keys()
    timestamp = datetime.now(timezone.utc).isoformat()
    overrides = []
    for r in rows:
        assert all(r[k] == v for k, v in original[r["mode_key"]].items()), r["mode_key"]
        if not r["manual_decision"].strip():
            assert not r["manual_reason"].strip(), r["mode_key"]
            continue
        decision = r["manual_decision"].strip().upper()
        assert decision in ("GOOD", "BAD") and decision != r["rules_decision"]
        assert decision == r["rf_cnn_decision"] and r["manual_reason"].strip()
        assert r["shot"] in shots
        path = args.data_root / r["mode_key"]
        assert input_fingerprint(path, datcon_path_for_mode(path)) == r["input_fingerprint"]
        overrides.append(dict(mode_key=r["mode_key"], path=str(path), input_fingerprint=r["input_fingerprint"],
            ntor=r["ntor"], frequency=r["omega"], original_rule_decision=r["original_rule_decision"],
            manual_decision=decision, manual_reason=r["manual_reason"], reviewer="Elena",
            adjudication_timestamp=timestamp))
    assert Counter(r["manual_decision"] for r in overrides) == {"GOOD": 9, "BAD": 8}
    changed_shots = sorted({portable_mode_key(r["path"]).split("/")[0] for r in overrides})
    assert len(changed_shots) == 13
    write(HERE / "manual_overrides.csv", overrides, MANUAL_OVERRIDE_FIELDS)
    assert load_manual_overrides(HERE / "manual_overrides.csv") == overrides
    for shot in changed_shots:
        assert not read(args.rules_root / shot / "manual_overrides.csv"), shot
        write(args.runtime_dir / "overrides" / (shot + ".csv"),
              [r for r in overrides if r["mode_key"].startswith(shot + "/")], MANUAL_OVERRIDE_FIELDS)
    snapshot = dict(shots=shots, changed_shots=changed_shots, source_sha256=source_hashes(),
        override_sha256=sha256_file(HERE / "manual_overrides.csv"), imported_utc=timestamp,
        old_trees={s: publisher.tree_digest(args.rules_root / s) for s in shots},
        ai_trees={s: publisher.tree_digest(args.ai_root / s) for s in shots})
    (args.runtime_dir / "run_inputs.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print("Validated 17 overrides; regenerating 13 affected shots", flush=True)

    def run(shot):
        command = [sys.executable, str(REPO / "scripts/sort_shot_mixed.py"), "--method", "rules",
            "--shot_dir", str(args.data_root / shot), "--out_dir", str(args.runtime_dir / "rules" / shot),
            "--rule_config", PRODUCTION_RULE_CONFIG_NAME, "--manual_overrides",
            str(args.runtime_dir / "overrides" / (shot + ".csv"))]
        with (args.runtime_dir / "logs" / (shot + ".log")).open("w") as f:
            subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
        print("Regenerated " + shot, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, changed_shots))
    verify(args)


def verify(args):
    snapshot = json.loads((args.runtime_dir / "run_inputs.json").read_text())
    assert source_hashes() == snapshot["source_sha256"]
    assert sha256_file(HERE / "manual_overrides.csv") == snapshot["override_sha256"]
    overrides = {r["mode_key"]: r for r in load_manual_overrides(HERE / "manual_overrides.csv")}
    comparisons, all_tae, changes, selections, summaries, excluded = [], [], [], [], [], []
    counts = Counter()
    new_trees = {}
    for shot in snapshot["shots"]:
        assert publisher.tree_digest(args.rules_root / shot) == snapshot["old_trees"][shot], shot
        assert publisher.tree_digest(args.ai_root / shot) == snapshot["ai_trees"][shot], shot
        old = keyed(args.rules_root / shot / "all_modes_rules.csv")
        directory = args.runtime_dir / "rules" / shot if shot in snapshot["changed_shots"] else args.rules_root / shot
        new = keyed(directory / "all_modes_rules.csv")
        ai = keyed(args.ai_root / shot / "all_modes_scored.csv")
        assert new.keys() == old.keys(), shot
        local = Counter()
        for key, r in new.items():
            before = old[key]
            assert all(r[k] == v for k, v in before.items() if k not in MUTABLE), key
            if key in overrides:
                override = overrides[key]
                assert r["override_status"] == "APPLIED" and r["decision_source"] == "manual_override", key
                assert all(r[k] == override[k] for k in ("manual_decision", "manual_reason", "reviewer", "adjudication_timestamp"))
                assert r["final_decision"] == override["manual_decision"] != before["final_decision"]
                changes.append(dict(path=key, shot=shot, before=before["final_decision"], after=r["final_decision"],
                    manual_reason=r["manual_reason"], original_rule_decision=r["rule_decision"],
                    rule_primary_reason=r["rule_primary_reason"], input_fingerprint=r["input_fingerprint"]))
            else:
                assert all(r[k] == before[k] for k in MUTABLE - {"selected_final", "duplicate_rank_score", "duplicate_rank_source"}), key
            if r["selected_final"] != before["selected_final"]:
                selections.append(dict(path=key, before_selected=before["selected_final"], after_selected=r["selected_final"],
                    final_label_changed=key in overrides))
            local[r["processing_status"]] += 1
            if r["processing_status"] != "RULE_EVALUATED":
                continue
            path = args.data_root / key
            assert input_fingerprint(path, datcon_path_for_mode(path)) == r["input_fingerprint"], key
            local["good_before_dedup"] += r["final_decision"] == "GOOD"
            local["selected_good"] += r["selected_final"] == "True"
            local["manual_overrides"] += r["override_status"] == "APPLIED"
            row = dict(path=key, label=r["final_decision"].lower(), shot=shot, ntor=r["ntor"], omega=r["omega"],
                gap_region=r["gap_region"], final_decision=r["final_decision"],
                final_reason=r["manual_reason"] or r["rule_primary_reason"], automatic_final_decision=before["final_decision"],
                original_rule_decision=r["rule_decision"], rule_primary_reason=r["rule_primary_reason"],
                decision_source=r["decision_source"], manual_decision=r["manual_decision"], manual_reason=r["manual_reason"],
                reviewer=r["reviewer"], adjudication_timestamp=r["adjudication_timestamp"], selected_final=r["selected_final"],
                overall_rule_severity=r["overall_rule_severity"], input_fingerprint=r["input_fingerprint"], mode_key=key)
            all_tae.append(row)
            a = ai.get(key)
            if a is None or a["final_label"].upper() not in ("GOOD", "BAD"):
                excluded.append(dict(row, reason="NO_CURRENT_AI_CLASSIFICATION"))
                continue
            for field in ("omega", "nr", "signed_delta", "fraction_below_upper2"):
                assert float(a[field]) == float(r[field]), (key, field)
            assert a["gap_region"] == r["gap_region"], key
            for field in ("p_rf_good", "p_cnn_good"):
                assert math.isfinite(float(a[field])) and 0 <= float(a[field]) <= 1
            row = dict(row, rf_cnn_decision=a["final_label"].upper(),
                rf_decision="GOOD" if float(a["p_rf_good"]) >= .5 else "BAD",
                cnn_decision="GOOD" if float(a["p_cnn_good"]) >= .5 else "BAD",
                p_rf_good=a["p_rf_good"], p_cnn_good=a["p_cnn_good"])
            comparisons.append(row)
            local["paired_tae"] += 1
            local["automatic_rules_vs_ai"] += before["final_decision"] != row["rf_cnn_decision"]
            for model in ("rf_cnn", "rf", "cnn"):
                local["final_vs_" + model] += r["final_decision"] != row[model + "_decision"]
            if r["final_decision"] != row["rf_cnn_decision"]:
                local["bad_ai_good" if r["final_decision"] == "BAD" else "good_ai_bad"] += 1
        summary = read(directory / "shot_summary_wide.csv")[0]
        assert summary["rule_configuration_sha256"] == PRODUCTION_RULE_CONFIG_SHA256
        assert int(summary["n_overrides_applied"]) == local["manual_overrides"]
        for field in ("n_stale_overrides", "n_ambiguous_overrides", "n_ineligible_overrides", "n_unmatched_overrides"):
            assert int(summary[field]) == 0, (shot, field)
        assert not read(directory / "resolution_warnings.csv")
        assert all(r["cluster_status"] == "PROCESSED_RULE_SEVERITY" for r in read(directory / "frequency_clusters.csv"))
        assert {r["mode_key"] for r in read(directory / "good_tae_final.csv")} == {
            k for k, r in new.items() if r["final_decision"] == "GOOD" and r["selected_final"] == "True"}
        summaries.append(dict(shot=shot, **{k: local[k] for k in ("RULE_EVALUATED", "ROUTED_EAE", "INVALID",
            "good_before_dedup", "selected_good", "manual_overrides", "paired_tae", "automatic_rules_vs_ai",
            "final_vs_rf_cnn", "bad_ai_good", "good_ai_bad", "final_vs_rf", "final_vs_cnn")}))
        counts.update(local)
        new_trees[shot] = publisher.tree_digest(directory)
        print("Verified " + shot, flush=True)
    assert {r["path"] for r in changes} == overrides.keys() and len(changes) == 17
    assert counts["good_before_dedup"] == 1647 and counts["manual_overrides"] == 17
    assert counts["automatic_rules_vs_ai"] == 365 and counts["final_vs_rf_cnn"] == 348
    assert source_hashes() == snapshot["source_sha256"]
    order = lambda r: (r["shot"], int(r["ntor"]), float(r["omega"]), r["mode_key"])
    all_tae.sort(key=order)
    comparisons.sort(key=order)
    different = [r for r in comparisons if r["final_decision"] != r["rf_cnn_decision"]]
    previous = keyed(BASELINE)
    assert {r["mode_key"] for r in different} == previous.keys() - overrides.keys()
    write(HERE / "label_changes.csv", changes, list(changes[0]))
    write(HERE / "selection_changes.csv", selections, ["path", "before_selected", "after_selected", "final_label_changed"])
    write(HERE / "shot_summary.csv", summaries, list(summaries[0]))
    write(HERE / "disagreements.csv", different, COMPARE_FIELDS)
    write(HERE / "rules_bad_ai_good.csv", [r for r in different if r["final_decision"] == "BAD"], COMPARE_FIELDS)
    write(HERE / "rules_good_ai_bad.csv", [r for r in different if r["final_decision"] == "GOOD"], COMPARE_FIELDS)
    for model in ("rf", "cnn"):
        write(HERE / f"final_vs_{model}.csv", [r for r in comparisons if r["final_decision"] != r[model + "_decision"]], COMPARE_FIELDS)
    write(HERE / "resolved_disagreements.csv", [r for r in comparisons if r["mode_key"] in overrides], COMPARE_FIELDS)
    write(HERE / "ai_comparison_excluded.csv", excluded, FIELDS + ["reason"])
    write(HERE / "accepted_tae_modes.csv", [r for r in all_tae if r["selected_final"] == "True"], FIELDS)
    write(args.runtime_dir / "all_tae_classifications.csv", all_tae, FIELDS)
    write(args.runtime_dir / "all_comparisons.csv", comparisons, COMPARE_FIELDS)
    by_key = {r["mode_key"]: r for r in comparisons}
    dispositions = [dict(path=k, before=r["rules_decision"], after=by_key[k]["final_decision"],
        action="OVERRIDE" if k in overrides else "KEEP_RULE_DECISION",
        manual_reason=by_key[k]["manual_reason"], input_fingerprint=r["input_fingerprint"])
        for k, r in previous.items()]
    write(HERE / "review_dispositions.csv", dispositions, list(dispositions[0]))
    receipt = dict(status="verified", **snapshot, counts=dict(counts), new_trees=new_trees,
        rule_configuration_sha256=PRODUCTION_RULE_CONFIG_SHA256, label_changes=len(changes),
        selected_changed_rows=len(selections), review_scope="All 365 disagreements reviewed; 17 overrides, 348 retained rule decisions.",
        rules_root=str(args.rules_root), ai_root=str(args.ai_root), data_root=str(args.data_root),
        output_sha256={p.name: sha256_file(p) for p in sorted(HERE.glob("*.csv"))})
    (HERE / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified totals " + json.dumps(dict(counts)), flush=True)


def publish(args):
    receipt = json.loads((HERE / "verification.json").read_text())
    assert receipt["status"] == "verified" and source_hashes() == receipt["source_sha256"]
    assert all(sha256_file(HERE / name) == digest for name, digest in receipt["output_sha256"].items())
    backup = args.rules_root / "before_manual_review_20260914"
    staging = args.rules_root / ".staging_manual_review_20260914"
    assert not backup.exists() and not staging.exists()
    for shot in receipt["shots"]:
        assert publisher.tree_digest(args.rules_root / shot) == receipt["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == receipt["ai_trees"][shot]
    for shot in receipt["changed_shots"]:
        source = args.runtime_dir / "rules" / shot
        assert publisher.tree_digest(source) == receipt["new_trees"][shot]
        shutil.copytree(source, staging / shot)
        assert publisher.tree_digest(staging / shot) == receipt["new_trees"][shot]
    backup.mkdir()
    installed = []
    for shot in receipt["changed_shots"]:
        target = args.rules_root / shot
        assert publisher.tree_digest(target) == receipt["old_trees"][shot]
        target.rename(backup / shot)
        try:
            (staging / shot).rename(target)
        except Exception:
            (backup / shot).rename(target)
            raise
        assert publisher.tree_digest(backup / shot) == receipt["old_trees"][shot]
        installed.append(dict(shot=shot, output=str(target), backup=str(backup / shot)))
    for shot in receipt["shots"]:
        assert publisher.tree_digest(args.rules_root / shot) == receipt["new_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == receipt["ai_trees"][shot]
    staging.rmdir()
    (HERE / "publication.json").write_text(json.dumps(dict(status="installed", installed=installed,
        unchanged_rules_shots=26, unchanged_ai_shots=39), indent=2) + "\n")
    print("Installed 13 updated shot exports; 26 rules and all 39 AI exports unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "verify", "publish"))
    for name in ("data-root", "rules-root", "ai-root", "runtime-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args)
