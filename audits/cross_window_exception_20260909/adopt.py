"""Stage/verify v7 rules outputs and training decisions, then publish backups.

Example with the scientific Python environment:
  python adopt.py stage --data-root /path/to/DiTw --training-root /path/to/data \
    --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
    --rf-model models/nova_mode_classifier.joblib --out-root outputs/review_window_v7
  python adopt.py publish --rules-root /path/to/sort_outputs \
    --out-root outputs/review_window_v7
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, load_rule_run_configuration
from tae_rule_io import sha256_file, stable_json, portable_mode_key

# Reuse the already exercised bounded export-tree fingerprinting helper.
spec = importlib.util.spec_from_file_location(
    "old_publisher",
    HERE.parent / "continuum_monotonic_tail_20260908/publish_regenerated.py",
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def old_feature_schema(features):
    features = json.loads(stable_json(features))
    features["feature_schema_version"] = "tae-rule-features-grouped-v18"
    features["crossing_features"].pop("continuum_crossing_window_exception", None)
    return features


def training_case(task):
    from nova_mode_loader import load_mode_from_nova
    from cont_features import load_datcon_for_mode
    from tae_eae_features import classify_gap_region, upper2_scalars
    from tae_rule_engine import (
        ContinuumCrossingConfig,
        ContinuumCrossingWindowConfig,
        evaluate_mode,
    )
    from tae_rule_io import input_fingerprint
    import numpy as np

    label, root = task
    path = root / label["path"]
    mode, omega, gamma, n = load_mode_from_nova(str(path))
    key = label["path"]
    fingerprint = input_fingerprint(path, path.with_name(f"datcon{n}"))
    result = dict(mode_key=key, input_fingerprint=fingerprint, label=label["validity"])
    if not np.isfinite([omega, gamma]).all() or omega <= 0:
        assert key == "nstxuG121123K51/N4/egn04w.8769E+01"
        return dict(result, before="INVALID_METADATA", after="INVALID_METADATA")
    low, high, *_ = load_datcon_for_mode(str(path), mode.shape[1])
    scalars = upper2_scalars(mode, omega, high)
    region = classify_gap_region(**scalars)
    if region == "eae_like":
        return dict(result, before="ROUTED_EAE", after="ROUTED_EAE")
    row = dict(
        path=str(path),
        mode_key=key,
        shot=key.split("/")[0],
        omega=omega,
        gamma_d=gamma,
        ntor=n,
        gap_region=region,
        input_fingerprint=fingerprint,
        **scalars,
    )
    kwargs = dict(
        mode=mode,
        low2=low,
        high2=high,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
    )
    before = evaluate_mode(
        row,
        **kwargs,
        continuum_crossing_window_config=ContinuumCrossingWindowConfig(
            exception_amplitude_max=None
        ),
    )
    after = evaluate_mode(row, **kwargs)
    assert before.decision in {"REVIEW", "BAD"} and after.decision in {
        "REVIEW",
        "BAD",
    }, key
    assert old_feature_schema(before.features) == old_feature_schema(
        after.features
    ), key
    return dict(result, before=before.primary_reason, after=after.primary_reason)


def verify_shot_case(row):
    """Recompute the complete staged record with the current code and raw input."""
    from nova_mode_loader import load_mode_from_nova
    from cont_features import load_datcon_for_mode
    from tae_rule_engine import ContinuumCrossingConfig, evaluate_mode
    from tae_rule_io import input_fingerprint

    path = Path(row["path"])
    mode, omega, gamma, n = load_mode_from_nova(str(path))
    assert (
        input_fingerprint(path, path.with_name(f"datcon{n}"))
        == row["input_fingerprint"]
    )
    assert (omega, gamma, n) == (
        float(row["omega"]),
        float(row["gamma_d"]),
        int(row["ntor"]),
    )
    low, high, *_ = load_datcon_for_mode(str(path), mode.shape[1])
    result = evaluate_mode(
        row,
        mode=mode,
        low2=low,
        high2=high,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
    )
    assert result.decision == row["rule_decision"], row["mode_key"]
    assert result.primary_reason == row["rule_primary_reason"], row["mode_key"]
    assert json.loads(stable_json(result.features)) == json.loads(
        row["rule_features"]
    ), row["mode_key"]
    return row["mode_key"]


def stage(args):
    required = [args.data_root, args.training_root, args.ai_root, args.rf_model]
    assert all(x is not None for x in required)
    config = load_rule_run_configuration(PRODUCTION_RULE_CONFIG_NAME)
    membership = (
        HERE.parent / "continuum_monotonic_tail_20260908/regenerated_shot_summary.csv"
    )
    shots = read(membership)
    assert len(shots) == 27
    expected = {
        r["mode_key"]: r for r in read(HERE / "recovered.csv") if r["cohort"] == "shots"
    }
    expected_training = {
        r["mode_key"]: r
        for r in read(HERE / "recovered.csv")
        if r["cohort"] == "training"
    }
    sources = [
        REPO / p
        for p in (
            "scripts/tae_rule_engine.py",
            "scripts/tae_rule_config.py",
            "scripts/sort_shot_rules.py",
            "scripts/sort_shot_mixed.py",
            "src/cont_features.py",
            "src/tae_eae_features.py",
            "training_labels/tae_like_train.csv",
        )
    ]
    sources += [config.source_path, args.rf_model, membership, Path(__file__)]
    hashes = {str(p): sha256_file(p) for p in sources}
    args.out_root.mkdir(parents=True, exist_ok=True)
    old_trees = {
        s["shot"]: publisher.tree_digest(args.rules_root / s["shot"]) for s in shots
    }

    def run(shot):
        directory = args.out_root / "rules" / shot["shot"]
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO / "scripts/sort_shot_mixed.py"),
            "--method",
            "rules",
            "--rule_config",
            config.name,
            "--shot_dir",
            str(args.data_root / shot["shot"]),
            "--out_dir",
            str(directory),
            "--rf_model",
            str(args.rf_model),
        ]
        with (directory / "regeneration.log").open("w") as log:
            subprocess.run(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        return shot["shot"]

    if args.phase == "stage":
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, job in enumerate(
                as_completed([pool.submit(run, s) for s in shots]), 1
            ):
                print(f"Regenerated {i}/27: {job.result()}", flush=True)
    counts = Counter()
    evaluated = []
    changes, disagreements, shot_summaries = [], [], []
    ai_hashes = {}
    for shot in shots:
        name = shot["shot"]
        assert publisher.tree_digest(args.rules_root / name) == old_trees[name]
        old = {
            r["mode_key"]: r
            for r in read(args.rules_root / name / "all_modes_rules.csv")
        }
        new = {
            r["mode_key"]: r
            for r in read(args.out_root / "rules" / name / "all_modes_rules.csv")
        }
        ai_path = args.ai_root / name / "all_modes_scored.csv"
        ai_hashes[str(ai_path)] = sha256_file(ai_path)
        ai = {portable_mode_key(r["path"]): r for r in read(ai_path)}
        assert old.keys() == new.keys() == ai.keys()
        assert len(new) == int(shot["input_modes"])
        for key, row in new.items():
            previous = old[key]
            assert not row["manual_decision"] and not previous["manual_decision"]
            for field in (
                "input_fingerprint",
                "gap_region",
                "processing_status",
                "omega",
                "gamma_d",
                "nr",
                "nhar",
                "signed_delta",
                "fraction_below_upper2",
            ):
                assert row[field] == previous[field], (key, field)
            counts["inputs"] += 1
            if row["processing_status"] != "RULE_EVALUATED":
                assert row["final_decision"] == previous["final_decision"]
                continue
            features = json.loads(row["rule_features"])
            evaluated.append(row)
            assert old_feature_schema(features) == json.loads(
                previous["rule_features"]
            ), key
            counts["unchanged_prior_features"] += 1
            wanted = (
                "NO_GOOD_TEMPLATE"
                if key in expected
                else previous["rule_primary_reason"]
            )
            assert row["rule_primary_reason"] == wanted, key
            if key in expected:
                assert row["input_fingerprint"] == expected[key]["input_fingerprint"]
                assert previous["rule_primary_reason"] == "BAD_CONT_CROSS_WINDOW"
                assert row["final_decision"] == "GOOD"
                evidence = features["crossing_features"][
                    "continuum_crossing_window_exception"
                ]
                assert evidence["all_violations_exempted"]
                changes.append(
                    dict(
                        shot=name,
                        mode_key=key,
                        input_fingerprint=row["input_fingerprint"],
                        before=previous["final_decision"],
                        after=row["final_decision"],
                    )
                )
            else:
                assert row["final_decision"] == previous["final_decision"]
            counts[row["final_decision"]] += 1
            if row["final_decision"].lower() != ai[key]["final_label"]:
                disagreements.append(
                    dict(
                        shot=name,
                        mode_key=key,
                        input_fingerprint=row["input_fingerprint"],
                        rules_decision=row["final_decision"],
                        rules_reason=row["rule_primary_reason"],
                        rf_cnn_decision=ai[key]["final_label"].upper(),
                        p_rf_good=ai[key]["p_rf_good"],
                        p_cnn_good=ai[key]["p_cnn_good"],
                        rf_cnn_tier=ai[key]["tier"],
                    )
                )
        summary = read(args.out_root / "rules" / name / "shot_summary_wide.csv")[0]
        assert summary["rule_configuration_sha256"] == config.sha256
        assert summary["continuum_crossing_window_exception_enabled"] == "True"
        assert (
            float(summary["continuum_crossing_window_exception_amplitude_max"]) == 0.2
        )
        assert float(summary["continuum_crossing_window_exception_k_max"]) == 0.1
        assert not read(args.out_root / "rules" / name / "resolution_warnings.csv")
        assert summary["duplicate_processing_status"] in {
            "COMPLETED_RF",
            "NO_CLOSE_FREQUENCY_CLUSTERS",
            "SKIPPED_NO_GOOD_MODES",
        }
        counts["selected_good"] += int(summary["n_final_good"])
        shot_summaries.append(
            dict(
                shot=name,
                input_modes=len(new),
                good_before_clustering=int(summary["n_final_good_before_clustering"]),
                selected_good=int(summary["n_final_good"]),
                disagreements=sum(r["shot"] == name for r in disagreements),
            )
        )
    assert {r["mode_key"] for r in changes} == expected.keys()
    assert counts["GOOD"] == 959 and counts["unchanged_prior_features"] == 4267
    print(
        "All 27 shot outputs match the 19 audited changes; recomputing all 4267 rule records.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify_shot_case, evaluated, chunksize=8))
    assert len(verified) == 4267
    counts["recomputed_current_features"] = len(verified)
    print("All current rule records verified; checking training.", flush=True)
    labels = read(REPO / "training_labels/tae_like_train.csv")
    with ProcessPoolExecutor(max_workers=4) as pool:
        training = list(
            pool.map(
                training_case,
                [(label, args.training_root) for label in labels],
                chunksize=8,
            )
        )
    training_changes = [r for r in training if r["before"] != r["after"]]
    assert {r["mode_key"] for r in training_changes} == expected_training.keys()
    for row in training_changes:
        assert row["label"] == "good" and row["after"] == "NO_GOOD_TEMPLATE"
        assert (
            row["input_fingerprint"]
            == expected_training[row["mode_key"]]["input_fingerprint"]
        )
    old_matrix = dict(Counter(f'{r["label"]}:{r["before"]}' for r in training))
    assert (
        old_matrix
        == json.loads((HERE / "summary.json").read_text())["training"][
            "baseline_label_matrix"
        ]
    )
    new_matrix = dict(Counter(f'{r["label"]}:{r["after"]}' for r in training))
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    write(HERE / "regenerated_disagreements.csv", disagreements, list(disagreements[0]))
    write(
        HERE / "regenerated_shot_summary.csv", shot_summaries, list(shot_summaries[0])
    )
    write(HERE / "adopted_changes.csv", changes, list(changes[0]))
    receipt = dict(
        status="verified_locally",
        config=config.name,
        config_sha256=config.sha256,
        counts=dict(counts),
        training_rows=len(training),
        training_changes=training_changes,
        training_matrix=new_matrix,
        disagreements=len(disagreements),
        source_sha256=hashes,
        ai_source_sha256=ai_hashes,
        previous_trees=old_trees,
        current_trees={
            s["shot"]: publisher.tree_digest(args.out_root / "rules" / s["shot"])
            for s in shots
        },
    )
    (HERE / "adoption_verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                counts=dict(counts),
                training_changes=len(training_changes),
                disagreements=len(disagreements),
            )
        ),
        flush=True,
    )


def publish(args):
    receipt = json.loads((HERE / "adoption_verification.json").read_text())
    assert receipt["status"] == "verified_locally"
    assert all(sha256_file(Path(p)) == h for p, h in receipt["source_sha256"].items())
    backup_name = "before_cross_window_exception_20260909"
    plans = []
    for shot, new in receipt["current_trees"].items():
        source, target = args.out_root / "rules" / shot, args.rules_root / shot
        backup = args.rules_root / backup_name / shot
        staging = args.rules_root / (".staging_" + backup_name) / shot
        assert not backup.exists() and not staging.exists()
        assert publisher.tree_digest(source) == new
        assert publisher.tree_digest(target) == receipt["previous_trees"][shot]
        plans.append((shot, source, target, backup, staging))
    for shot, source, target, backup, staging in plans:
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, staging)
        assert publisher.tree_digest(staging) == receipt["current_trees"][shot]
    published = []
    for shot, source, target, backup, staging in plans:
        assert publisher.tree_digest(target) == receipt["previous_trees"][shot]
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)
        try:
            staging.rename(target)
        except Exception:
            backup.rename(target)
            raise
        assert publisher.tree_digest(target) == receipt["current_trees"][shot]
        assert publisher.tree_digest(backup) == receipt["previous_trees"][shot]
        published.append(
            dict(
                shot=shot,
                output=str(target),
                backup=str(backup),
                current=receipt["current_trees"][shot],
                previous=receipt["previous_trees"][shot],
            )
        )
    (args.rules_root / (".staging_" + backup_name)).rmdir()
    assert all(
        sha256_file(Path(p)) == h for p, h in receipt["ai_source_sha256"].items()
    )
    (HERE / "publication.json").write_text(json.dumps(published, indent=2) + "\n")
    print(
        f"Published {len(published)} rules outputs with verified backups; RF-CNN exports preserved."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("stage", "verify", "publish"),
        help="stage runs the sorters; verify reuses staged runs and recomputes every rule record",
    )
    for name in ("rules-root", "out-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("data-root", "training-root", "ai-root", "rf-model"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    (publish if args.phase == "publish" else stage)(args)


if __name__ == "__main__":
    main()
