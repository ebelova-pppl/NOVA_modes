"""Regenerate, verify and publish the approved clearance-only v8 policy.

python audits/extremum_floor_20260909/adopt_clearance.py stage \
  --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
  --training-root /path/to/training/data --rf-model models/nova_mode_classifier.joblib \
  --out-root outputs/review_extremum_clearance_v8_20260909
Repeat with verify to reuse staged runs; publish installs verified backups.
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
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]
import audit as floor_audit
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_io import sha256_file

spec = importlib.util.spec_from_file_location(
    "publisher",
    HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py",
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def keyed(path):
    rows = read(path)
    result = {"/".join(Path(r["path"]).parts[-3:]): r for r in rows}
    assert len(result) == len(rows)
    return result


def write(path, rows, fields=None):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def shots():
    return [
        r["shot"]
        for r in read(
            HERE.parent
            / "continuum_monotonic_tail_20260908/regenerated_shot_summary.csv"
        )
    ]


def run_one(task):
    shot, args = task
    original = read(args.rules_root / shot / "all_modes_rules.csv")[0]
    shot_dir = Path(original["path"]).parents[1]
    target = args.out_root / "rules" / shot
    command = [
        sys.executable,
        str(REPO / "scripts/sort_shot_mixed.py"),
        "--method",
        "rules",
        "--shot_dir",
        str(shot_dir),
        "--out_dir",
        str(target),
        "--rf_model",
        str(args.rf_model),
    ]
    with (args.out_root / "logs" / f"{shot}.log").open("w") as log:
        subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    print("Regenerated " + shot, flush=True)


def stage(args):
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(exist_ok=True)
    sources = [
        REPO / p
        for p in (
            "scripts/tae_rule_engine.py",
            "scripts/tae_rule_config.py",
            "scripts/sort_shot_rules.py",
            "scripts/sort_shot_mixed.py",
            "scripts/make_tae_like_list.py",
            "src/cont_features.py",
            "src/tae_eae_features.py",
            "src/input_validity.py",
            "src/nova_mode_loader.py",
            "configs/known_invalid_inputs.csv",
            "configs/rules/tae_rules_production_v7.yaml",
            "configs/rules/tae_rules_production_v8.yaml",
            "training_labels/tae_like_train.csv",
            "audits/extremum_floor_20260909/audit.py",
            "audits/extremum_floor_20260909/adopt_clearance.py",
            "outputs/review_extremum_floor_20260909/all_measurements.csv",
            "audits/r06_input_validity_20260909/current_disagreements.csv",
            "audits/extremum_floor_20260909/newly_rejected_width_1_0.csv",
        )
    ]
    sources.append(args.rf_model.resolve())
    snapshot = dict(
        source_sha256={str(p): sha256_file(p) for p in sources},
        old_trees={s: publisher.tree_digest(args.rules_root / s) for s in shots()},
        ai_trees={s: publisher.tree_digest(args.ai_root / s) for s in shots()},
    )
    snapshot_path = args.out_root / "run_inputs.json"
    assert not snapshot_path.exists(), "Use verify for an already staged run"
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_one, [(s, args) for s in shots()]))
    verify(args)


def verify(args):
    snapshot = json.loads((args.out_root / "run_inputs.json").read_text())
    assert all(sha256_file(Path(p)) == h for p, h in snapshot["source_sha256"].items())
    expected = {r["mode_key"]: r for r in read(HERE / "newly_rejected_width_1_0.csv")}
    changes = []
    summaries = []
    disagreements = []
    absent_invalid = []
    counts = Counter()
    invalid_provenance_refreshes = 0
    new_trees = {}
    for shot in shots():
        old_dir = args.rules_root / shot
        new_dir = args.out_root / "rules" / shot
        assert publisher.tree_digest(old_dir) == snapshot["old_trees"][shot]
        assert publisher.tree_digest(args.ai_root / shot) == snapshot["ai_trees"][shot]
        old = keyed(old_dir / "all_modes_rules.csv")
        new = keyed(new_dir / "all_modes_rules.csv")
        ai = keyed(args.ai_root / shot / "all_modes_scored.csv")
        assert old.keys() == ai.keys()
        assert new.keys() <= old.keys(), (shot, "unexpected new inputs")
        # Eight previously excluded C50 N1 files disappeared from the raw
        # directory after the earlier exports. Never hide missing valid inputs.
        for key in sorted(old.keys() - new.keys()):
            prev = old[key]
            assert key.startswith("nstxuG142301C50/N1/")
            assert prev["processing_status"] == prev["final_decision"] == "INVALID"
            assert ai[key]["final_label"].upper() == "INVALID"
            assert not Path(prev["path"]).exists(), key
            absent_invalid.append(
                dict(
                    mode_key=key,
                    input_fingerprint=prev["input_fingerprint"],
                    prior_status="INVALID",
                    current_status="ABSENT_FROM_RAW_DIRECTORY",
                )
            )
        for key, row in new.items():
            prev = old[key]
            for field in (
                "input_fingerprint",
                "omega",
                "gamma_d",
                "ntor",
                "nr",
                "nhar",
                "rad_loc",
                "rad_width",
                "processing_status",
                "gap_region",
                "signed_delta",
                "fraction_below_upper2",
                "manual_decision",
            ):
                assert row[field] == prev[field], (key, field)
            counts["inputs"] += 1
            if row["processing_status"] != "RULE_EVALUATED":
                comparable = dict(prev)
                if row["diagnostic_message"] != prev["diagnostic_message"]:
                    # C50's last export preceded the v2 registry adding R06.
                    assert key.startswith("nstxuG142301C50/N1/")
                    old_suffix = (
                        "known-invalid-inputs-v1 registry_sha256="
                        "8205ac1066ea6816eb383dbd49b6f2412da6b2996ea2bc9886fab3af9a467ba5"
                    )
                    new_suffix = (
                        "known-invalid-inputs-v2 registry_sha256="
                        + sha256_file(REPO / "configs/known_invalid_inputs.csv")
                    )
                    assert prev["diagnostic_message"].endswith(old_suffix)
                    comparable["diagnostic_message"] = (
                        prev["diagnostic_message"].removesuffix(old_suffix) + new_suffix
                    )
                    invalid_provenance_refreshes += 1
                assert row == comparable, (key, "non-evaluated row changed")
                continue
            counts["features_checked"] += 1
            f = json.loads(row["rule_features"])
            before = json.loads(prev["rule_features"])
            e = f["resolution_features"]["interior_unresolved_envelope"]
            old_e = before["resolution_features"]["interior_unresolved_envelope"]
            assert (
                e["ext_df_gap_min"] == 0.001 and e["ext_df_gap_min_inclusive"] is False
            )
            assert e["extremum_exception_applied"] == bool(
                old_e["extremum_exception_applied"] and old_e["ext_df_gap"] > 0.001
            )
            f["feature_schema_version"] = before["feature_schema_version"]
            e["ext_df_gap_min"] = old_e["ext_df_gap_min"]
            e.pop("ext_df_gap_min_inclusive")
            e["extremum_exception_applied"] = old_e["extremum_exception_applied"]
            assert f == before, (key, "prior measured features changed")
            if row["final_decision"] != prev["final_decision"]:
                assert (
                    key in expected
                    and expected[key]["input_fingerprint"] == row["input_fingerprint"]
                )
                assert (
                    prev["final_decision"] == "GOOD" and row["final_decision"] == "BAD"
                )
                assert row["rule_primary_reason"] == "BAD_INTERIOR_UNRESOLVED_ENVELOPE"
                changes.append(
                    dict(
                        mode_key=key,
                        input_fingerprint=row["input_fingerprint"],
                        before="GOOD",
                        after="BAD",
                        reason=row["rule_primary_reason"],
                    )
                )
            else:
                assert (
                    row["rule_decision"] == prev["rule_decision"]
                    and row["rule_primary_reason"] == prev["rule_primary_reason"]
                )
                assert row["selected_final"] == prev["selected_final"], (
                    key,
                    "representative changed",
                )
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
            s["interior_envelope_ext_df_gap_min"] == "0.001"
            and s["interior_envelope_ext_df_gap_min_inclusive"] == "False"
        )
        assert (
            "FALLBACK" not in s["duplicate_processing_status"]
            and "NO_RF" not in s["duplicate_processing_status"]
        )
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
    assert {r["mode_key"] for r in changes} == {
        k for k, v in expected.items() if v["cohort"] == "shots"
    }
    assert len(absent_invalid) == 8
    assert invalid_provenance_refreshes == 65
    assert counts == {"inputs": 19317, "features_checked": 4187}, counts
    assert sum(int(r["selected_good"]) for r in summaries) == 944
    print(
        "Verified all 27 exports and six label changes; verifying training.", flush=True
    )
    baseline = {
        r["mode_key"]: r
        for r in read(
            REPO / "outputs/review_extremum_floor_20260909/all_measurements.csv"
        )
        if r["cohort"] == "training"
    }
    labels = read(REPO / "training_labels/tae_like_train.csv")
    with ProcessPoolExecutor(max_workers=4) as pool:
        training = list(
            pool.map(
                floor_audit.measure,
                [("training", r, args.training_root) for r in labels],
                chunksize=8,
            )
        )
    training_changes = []
    for row in training:
        before = baseline[row["mode_key"]]
        assert row["input_fingerprint"] == before["input_fingerprint"]
        if row["baseline_decision"] != before["baseline_decision"]:
            assert row["mode_key"] in expected
            assert (
                before["baseline_decision"] == "GOOD"
                and row["baseline_decision"] == "BAD"
            )
            training_changes.append(row)
    assert len(training) == 2390 and len(training_changes) == 1
    assert training_changes[0]["mode_key"] == "nstxuG142301H47/N7/egn07w.2530E+02"
    old_disagreements = read(
        HERE.parent / "r06_input_validity_20260909/current_disagreements.csv"
    )
    previous = {r["mode_key"]: r for r in old_disagreements}
    current = {r["mode_key"]: r for r in disagreements}
    assert all(previous[k] == current[k] for k in previous.keys() & current.keys())
    added = [current[k] for k in sorted(current.keys() - previous.keys())]
    removed = [previous[k] for k in sorted(previous.keys() - current.keys())]
    assert len(added) == 4 and len(removed) == 2 and len(current) == 228
    write(HERE / "clearance_adopted_changes.csv", changes)
    write(HERE / "absent_invalid_inputs.csv", absent_invalid)
    write(HERE / "clearance_training_change.csv", training_changes)
    write(HERE / "clearance_shot_summary.csv", summaries)
    write(
        HERE / "current_disagreements.csv",
        sorted(disagreements, key=lambda r: r["mode_key"]),
    )
    write(HERE / "disagreements_added.csv", added)
    write(HERE / "disagreements_removed.csv", removed)
    assert all(sha256_file(Path(p)) == h for p, h in snapshot["source_sha256"].items())
    receipt = dict(
        status="verified",
        config=PRODUCTION_RULE_CONFIG_NAME,
        config_sha256=PRODUCTION_RULE_CONFIG_SHA256,
        counts=dict(counts),
        absent_previously_invalid_inputs=len(absent_invalid),
        invalid_registry_provenance_refreshes=invalid_provenance_refreshes,
        rules_good_before_dedup=950,
        selected_good=944,
        training_rows=len(training),
        training_matrix=dict(
            Counter(
                r["training_label"] + ":" + r["baseline_decision"] for r in training
            )
        ),
        shot_label_changes=6,
        training_decision_changes=1,
        disagreements=228,
        added_disagreements=4,
        removed_disagreements=2,
        source_sha256=snapshot["source_sha256"],
        old_trees=snapshot["old_trees"],
        new_trees=new_trees,
        ai_trees=snapshot["ai_trees"],
    )
    (HERE / "clearance_verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        "Verified: six shot rejections; one training GOOD conflict; 228 disagreements.",
        flush=True,
    )


def publish(args):
    receipt = json.loads((HERE / "clearance_verification.json").read_text())
    assert receipt["status"] == "verified"
    assert all(sha256_file(Path(p)) == h for p, h in receipt["source_sha256"].items())
    backup_name = "before_extremum_clearance_v8_20260909"
    plans = []
    for shot in shots():
        source = args.out_root / "rules" / shot
        target = args.rules_root / shot
        backup = args.rules_root / backup_name / shot
        staging = args.rules_root / (".staging_" + backup_name) / shot
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
        records.append(
            dict(
                shot=shot,
                output=str(target),
                backup=str(backup),
                new=receipt["new_trees"][shot],
                old=receipt["old_trees"][shot],
            )
        )
    (HERE / "clearance_publication.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(
        "Published all 27 rules exports with verified backups. RF-CNN outputs unchanged."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["stage", "verify", "publish"])
    for name in ("rules-root", "ai-root", "training-root", "rf-model", "out-root"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    globals()[args.phase](args)
