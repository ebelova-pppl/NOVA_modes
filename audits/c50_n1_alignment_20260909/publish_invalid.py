"""Verify and publish the user-approved C50 N1 input invalidation.

Run each phase with:
  python audits/c50_n1_alignment_20260909/publish_invalid.py verify \
    --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
    --staged-root outputs/review_c50_invalid_n1_20260909
Then replace 'verify' with 'publish' to install the verified outputs with backups.
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
SHOT = "nstxuG142301C50"
PREFIX = SHOT + "/N1/"
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
    return {"/".join(Path(r["path"]).parts[-3:]): r for r in read(path)}


def verify(args):
    inputs = json.loads((HERE / "invalidation_run_inputs.json").read_text())
    assert all(digest(REPO / p) == h for p, h in inputs["source_sha256"].items())
    snapshots = {}
    current = {}
    for method, root, name, filename in [
        ("rules", args.rules_root, "sort_outputs", "all_modes_rules.csv"),
        ("rf-cnn", args.ai_root, "sort_outputs_ai", "all_modes_scored.csv"),
    ]:
        assert publisher.tree_digest(root / SHOT) == inputs["old_trees"][name]
        previous = keyed(root / SHOT / filename)
        new = keyed(args.staged_root / method / filename)
        assert previous.keys() == new.keys() and len(new) == 611
        invalid = [k for k in new if k.startswith(PREFIX)]
        assert len(invalid) == 73
        for key, row in new.items():
            if key not in invalid:
                assert row == previous[key], (method, key)
                continue
            for field in (
                "nr",
                "nhar",
                "omega",
                "gamma_d",
                "ntor",
                "rad_loc",
                "rad_width",
            ):
                assert row[field] == previous[key][field], (method, key, field)
            assert not row["gap_region"]
            if method == "rules":
                assert row["input_fingerprint"] == previous[key]["input_fingerprint"]
                assert row["final_decision"] == row["processing_status"] == "INVALID"
                assert row["preprocessing_primary_reason"] == "KNOWN_INVALID_INPUT"
                assert "CONTINUUM_MODE_MISMATCH" in row["diagnostic_message"]
                assert not row["rule_version"] and json.loads(row["rule_features"]) == {}
            else:
                assert row["status"] == "rejected" and row["final_label"] == "invalid"
                assert row["rejection_reason"] == "KNOWN_INVALID_INPUT"
                assert "CONTINUUM_MODE_MISMATCH" in row["error_message"]
                assert not row["p_rf_good"] and not row["p_cnn_good"]
        for filename in (
            "tae_like_all.csv",
            "eae_like.csv",
            "good_tae_final.csv",
            "good_tae_unchecked.csv",
            "bad_tae_like.csv",
        ):
            assert not any(
                k.startswith(PREFIX)
                for k in keyed(args.staged_root / method / filename)
            )
        assert len(read(args.staged_root / method / "rejected_modes.csv")) == 73
        summary = read(args.staged_root / method / "shot_summary_wide.csv")[0]
        assert summary["n_known_invalid_inputs"] == "73"
        assert summary["n_eae_like"] == "344"
        assert summary["n_final_good"] == ("0" if method == "rules" else "1")
        snapshots[method] = dict(
            old=inputs["old_trees"][name],
            new=publisher.tree_digest(args.staged_root / method),
        )
        current[method] = (previous, new)
    old_rules, new_rules = current["rules"]
    old_ai, new_ai = current["rf-cnn"]
    changed = [
        dict(
            mode_key=k,
            input_fingerprint=row["input_fingerprint"],
            previous_gap_region=old_rules[k]["gap_region"],
            previous_rules_decision=old_rules[k]["final_decision"],
            previous_rf_cnn_decision=old_ai[k]["final_label"],
            input_validity="INVALID",
            reason="CONTINUUM_MODE_MISMATCH",
        )
        for k, row in new_rules.items()
        if k.startswith(PREFIX)
    ]
    write(HERE / "invalidated_modes.csv", changed)
    prior_path = (
        HERE.parent / "cross_window_exception_20260909/regenerated_disagreements.csv"
    )
    prior = read(prior_path)
    retained = [r for r in prior if not r["mode_key"].startswith(PREFIX)]
    removed = [r for r in prior if r["mode_key"].startswith(PREFIX)]
    assert len(prior) == 234 and len(retained) == 232 and len(removed) == 2
    # Recompute the remaining C50 disagreements from both new canonical runs.
    actual = {
        k
        for k, r in new_rules.items()
        if r["processing_status"] == "RULE_EVALUATED"
        and r["final_decision"].lower() != new_ai[k]["final_label"]
    }
    assert actual == {r["mode_key"] for r in retained if r["shot"] == SHOT}
    write(HERE / "current_disagreements.csv", retained)
    write(HERE / "disagreements_removed.csv", removed)
    training = read(REPO / "training_labels/tae_like_train.csv")
    assert not any(r["path"].startswith(PREFIX) for r in training)
    receipt = dict(
        status="verified",
        modes_invalidated=73,
        unchanged_modes_per_method=538,
        removed_disagreements=2,
        current_disagreements=232,
        trees=snapshots,
        source_sha256=inputs["source_sha256"],
        verifier_sha256=digest(Path(__file__)),
        previous_disagreements_sha256=digest(prior_path),
    )
    (HERE / "invalidation_verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        "Verified: 73 INVALID per method; all 538 other rows exactly unchanged; 232 current disagreements."
    )


def publish(args):
    receipt = json.loads((HERE / "invalidation_verification.json").read_text())
    assert receipt["status"] == "verified" and receipt["verifier_sha256"] == digest(
        Path(__file__)
    )
    assert all(digest(REPO / p) == h for p, h in receipt["source_sha256"].items())
    backup_name = "before_c50_n1_invalid_20260909"
    plans = []
    for method, root in [("rules", args.rules_root), ("rf-cnn", args.ai_root)]:
        source, target = args.staged_root / method, root / SHOT
        backup, staging = root / backup_name / SHOT, root / (".staging_" + backup_name)
        assert not backup.exists() and not staging.exists()
        hashes = receipt["trees"][method]
        assert publisher.tree_digest(source) == hashes["new"]
        assert publisher.tree_digest(target) == hashes["old"]
        plans.append((method, source, target, backup, staging))
    for method, source, target, backup, staging in plans:
        shutil.copytree(source, staging)
        assert publisher.tree_digest(staging) == receipt["trees"][method]["new"]
    result = []
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
        result.append(
            dict(
                method=method,
                output=str(target),
                backup=str(backup),
                **receipt["trees"][method]
            )
        )
    (HERE / "invalidation_publication.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("Published both C50 output sets with verified backups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["verify", "publish"])
    for name in ("rules-root", "ai-root", "staged-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (verify if args.phase == "verify" else publish)(args)
