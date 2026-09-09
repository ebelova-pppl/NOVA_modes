"""Verify/publish R06 INVALID exports, preserving previous output directories.

Example (run verify first, then publish):
  python audits/r06_input_validity_20260909/verify_and_publish.py verify \
    --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
    --staged-root outputs/review_r06_invalid_20260909
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHOT = "nstxuG133964R06"
spec = importlib.util.spec_from_file_location(
    "publisher",
    HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py",
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def keyed(path):
    rows = read(path)
    result = {"/".join(Path(r["path"]).parts[-3:]): r for r in rows}
    assert len(result) == len(rows), path
    return result


def verify(args):
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    assert all(digest(REPO / p) == h for p, h in inputs["source_sha256"].items())
    trees, previous = {}, {}
    for method, root, name, filename in [
        ("rules", args.rules_root, "sort_outputs", "all_modes_rules.csv"),
        ("rf-cnn", args.ai_root, "sort_outputs_ai", "all_modes_scored.csv"),
    ]:
        assert publisher.tree_digest(root / SHOT) == inputs["old_trees"][name]
        old = keyed(root / SHOT / filename)
        staged = args.staged_root / method
        new = keyed(staged / filename)
        assert old.keys() == new.keys() and len(new) == 610
        for key, row in new.items():
            assert key.startswith(SHOT + "/")
            for field in (
                "nr",
                "nhar",
                "omega",
                "gamma_d",
                "ntor",
                "rad_loc",
                "rad_width",
            ):
                assert row[field] == old[key][field], (method, key, field)
            assert row["nr"] == "201" and not row["gap_region"]
            if method == "rules":
                assert row["input_fingerprint"] == old[key]["input_fingerprint"]
                assert row["final_decision"] == row["processing_status"] == "INVALID"
                assert row["preprocessing_primary_reason"] == "KNOWN_INVALID_INPUT"
                assert "SUSPECT_EIGENMODE_STRUCTURE" in row["diagnostic_message"]
                assert (
                    not row["rule_version"] and json.loads(row["rule_features"]) == {}
                )
            else:
                assert row["status"] == "rejected" and row["final_label"] == "invalid"
                assert row["rejection_reason"] == "KNOWN_INVALID_INPUT"
                assert "SUSPECT_EIGENMODE_STRUCTURE" in row["error_message"]
                assert not row["p_rf_good"] and not row["p_cnn_good"]
        for filename in (
            "tae_like_all.csv",
            "eae_like.csv",
            "good_tae_final.csv",
            "good_tae_unchecked.csv",
            "bad_tae_like.csv",
        ):
            assert not read(staged / filename), (method, filename)
        assert len(read(staged / "rejected_modes.csv")) == 610
        summary = read(staged / "shot_summary_wide.csv")[0]
        assert summary["n_total_files"] == summary["n_known_invalid_inputs"] == "610"
        for field in (
            "n_tae_like",
            "n_mixed",
            "n_eae_like",
            "n_final_good",
            "n_final_bad",
        ):
            assert summary[field] == "0", (method, field)
        assert (
            summary[
                "n_rule_evaluated" if method == "rules" else "n_sent_to_classifiers"
            ]
            == "0"
        )
        trees[method] = dict(
            old=inputs["old_trees"][name], new=publisher.tree_digest(staged)
        )
        previous[method] = old
    write(
        HERE / "invalidated_modes.csv",
        [
            dict(
                mode_key=k,
                input_fingerprint=r["input_fingerprint"],
                previous_gap_region=r["gap_region"],
                previous_rules_decision=r["final_decision"],
                previous_rf_cnn_decision=previous["rf-cnn"][k]["final_label"],
                input_validity="INVALID",
                reason="SUSPECT_EIGENMODE_STRUCTURE",
            )
            for k, r in previous["rules"].items()
        ],
    )
    prior_path = HERE.parent / "c50_n1_alignment_20260909/current_disagreements.csv"
    prior = read(prior_path)
    retained = [r for r in prior if r["shot"] != SHOT]
    removed = [r for r in prior if r["shot"] == SHOT]
    assert len(prior) == 232 and len(retained) == 226 and len(removed) == 6
    assert all(r["mode_key"] in previous["rules"] for r in removed)
    write(HERE / "current_disagreements.csv", retained)
    write(HERE / "disagreements_removed.csv", removed)
    assert not any(
        SHOT in Path(r["path"]).parts
        for r in read(REPO / "training_labels/tae_like_train.csv")
    )
    receipt = dict(
        status="verified",
        modes_invalidated_per_method=610,
        removed_disagreements=6,
        current_disagreements=226,
        trees=trees,
        source_sha256=inputs["source_sha256"],
        verifier_sha256=digest(Path(__file__)),
        previous_disagreements_sha256=digest(prior_path),
    )
    (HERE / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified 610 INVALID inputs per method; 226 current disagreements.")


def publish(args):
    receipt = json.loads((HERE / "verification.json").read_text())
    assert receipt["status"] == "verified" and receipt["verifier_sha256"] == digest(
        Path(__file__)
    )
    assert all(digest(REPO / p) == h for p, h in receipt["source_sha256"].items())
    backup_name = "before_r06_invalid_20260909"
    plans = []
    for method, root in [("rules", args.rules_root), ("rf-cnn", args.ai_root)]:
        source, target = args.staged_root / method, root / SHOT
        backup, staging = root / backup_name / SHOT, root / (".staging_" + backup_name)
        assert not backup.exists() and not staging.exists()
        assert publisher.tree_digest(source) == receipt["trees"][method]["new"]
        assert publisher.tree_digest(target) == receipt["trees"][method]["old"]
        plans.append((method, source, target, backup, staging))
    for method, source, target, backup, staging in plans:
        shutil.copytree(source, staging)
        assert publisher.tree_digest(staging) == receipt["trees"][method]["new"]
    published = []
    for method, source, target, backup, staging in plans:
        assert publisher.tree_digest(target) == receipt["trees"][method]["old"]
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)
        try:
            staging.rename(target)
        except Exception:
            backup.rename(target)
            raise
        assert publisher.tree_digest(target) == receipt["trees"][method]["new"]
        assert publisher.tree_digest(backup) == receipt["trees"][method]["old"]
        published.append(
            dict(
                method=method,
                output=str(target),
                backup=str(backup),
                **receipt["trees"][method]
            )
        )
    (HERE / "publication.json").write_text(json.dumps(published, indent=2) + "\n")
    print("Published both R06 output sets with verified backups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["verify", "publish"])
    for name in ("rules-root", "ai-root", "staged-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (verify if args.phase == "verify" else publish)(args)
