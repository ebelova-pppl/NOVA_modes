#!/usr/bin/env python3
"""Run a fresh paired pilot while unresolved N1 inputs are being corrected.

See README.md for selection, preflight, stage, and install commands. Existing
shot exports are never overwritten. All scientific preprocessing is shared.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from make_tae_like_list import preprocess_shot
from tae_rule_io import input_fingerprint, portable_mode_key, sha256_file

SEED = 20260910
QUOTAS = [("high", 2), ("low", 4), ("medium", 6)]
CONFIG = REPO / "configs/rules/tae_rules_production_v11.yaml"
ALIGNMENT = REPO / "audits/n1_database_alignment_20260910"


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields or list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def series(shot):
    match = re.match(r"nstxuE(\d+)", shot)
    return match.group(1) if match else ""


def files(shot_dir, ns=range(1, 11)):
    return sorted(p for n in ns for p in (shot_dir / f"N{n}").glob("egn*") if p.is_file())


def keyed(rows):
    result = {r.get("mode_key") or portable_mode_key(r["path"]):r for r in rows}
    assert len(result) == len(rows), "Duplicate mode keys"
    return result


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode()+b"\0")
        digest.update(sha256_file(path).encode()+b"\n")
    return digest.hexdigest()


def source_hashes(args):
    paths = [CONFIG, Path(__file__), args.rf_model, args.cnn_model,
             REPO / "training_labels/tae_like_train.csv", REPO / "configs/known_invalid_inputs.csv"]
    # Fingerprint the local inference implementation, including shared helpers.
    paths += sorted((REPO / "src").glob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    return {str(p):sha256_file(p) for p in paths}


def check_sources(args):
    metadata = json.loads((HERE / "metadata.json").read_text())
    assert source_hashes(args) == metadata["source_sha256"], "Code/config/model/training/registry changed"
    return metadata


def check_alignment_snapshot(args, shot):
    """A previous clear screen cannot authorize changed N1/N2 input bytes."""
    for n in (1, 2):
        snapshot = json.loads((args.alignment_runtime / "groups" / f"{shot}_N{n}.json").read_text())
        assert not snapshot["error"]
        names = [p.name for p in files(args.data_root / shot, [n])]
        assert names == snapshot["mode_names"], f"{shot}/N{n}: alignment mode inventory changed"
        for path, digest in snapshot["sources"].items():
            assert sha256_file(path) == digest, f"Alignment input changed: {path}"


def select(args):
    assert not (HERE / "selection.csv").exists(), "Selection already frozen"
    inventory = read(REPO / "audits/main_dataset_shots/shot_status.csv")
    n1 = {r["shot"]:r for r in read(ALIGNMENT / "database_n1_summary.csv")}
    n2 = {r["shot"]:r for r in read(ALIGNMENT / "database_n2_control_summary.csv")}
    previous = {series(r["shot"]) for r in inventory if r["active_training_shot"] == "yes" or r["post_training_checked"] == "yes"} - {""}
    recalculation_series = {series(r["shot"]) for r in read(ALIGNMENT / "priority_review.csv")} - {""}
    raw = read(args.alignment_runtime / "mode_coverage.csv")
    empty_with_crossings = {r["shot"] for r in raw if r["status"] == "NO_LOGGED_SINGULARITIES" and int(r["n_interior_crossings"] or 0)}
    pool, excluded = [], []
    for row in inventory:
        shot, reason = row["shot"], ""
        if row["active_training_shot"] == "yes" or row["post_training_checked"] == "yes" or row["status"] != "unchecked":
            reason = "training_checked_or_other_status"
        elif series(shot) in previous:
            reason = "E_discharge_already_training_or_checked"
        elif series(shot) in recalculation_series:
            reason = "E_discharge_has_priority_N1_case"
        elif n1.get(shot, {}).get("interpretation") != "WITHIN_REVIEW_TOLERANCE":
            reason = "N1_review_or_insufficient_evidence"
        elif any((root / shot).exists() for root in (args.rules_root, args.ai_root)):
            reason = "existing_output_directory"
        else:
            for screen in (n1[shot], n2[shot]):
                if (not int(screen["n_modes_with_matched_interior_crossings"])
                    or float(screen["fraction_abs_gt_2"] or "nan") != 0
                    or float(screen["raw_fraction_abs_gt_2"] or "nan") != 0
                    or any(int(screen[k]) for k in ("n_raw_no_frequency_match", "n_raw_incomplete_or_conflicting", "n_input_errors"))
                    or shot in empty_with_crossings):
                    reason = "N1_or_N2_raw_log_coverage_or_offset"
                    break
        if reason:
            excluded.append(dict(shot=shot, reason=reason))
            continue
        count = len(files(args.data_root / shot))
        pool.append(dict(shot=shot, shot_family=row["shot_family"], E_series=series(shot),
                         input_modes=count, size_bin="low" if count <= 400 else "medium" if count <= 800 else "high"))
    ordered, chosen, used = [], [], set()
    rng = random.Random(SEED)
    for size, quota in QUOTAS:
        group = sorted((r for r in pool if r["size_bin"] == size), key=lambda r:r["shot"])
        rng.shuffle(group)
        picked = 0
        for rank, row in enumerate(group, 1):
            row["stratum_rank"] = rank
            ordered.append(row)
            if picked < quota and row["E_series"] not in used:
                chosen.append(dict(row, selection_status="pending_preflight", notes=""))
                used.add(row["E_series"])
                picked += 1
        assert picked == quota, (size, picked, quota)
    assert len(chosen) == 12
    write(HERE / "selection.csv", chosen)
    write(HERE / "candidate_pool.csv", ordered)
    write(HERE / "excluded_from_pool.csv", excluded)
    metadata = dict(seed=SEED, quotas=QUOTAS, created_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    selection_method="Shuffled sorted size strata; high then low then medium; unique unseen E discharges; no model scores consulted.",
                    input_data_root=str(args.data_root), rules_output_root=str(args.rules_root), ai_output_root=str(args.ai_root),
                    source_sha256=source_hashes(args), inventory_before_sha256=sha256_file(REPO / "audits/main_dataset_shots/shot_status.csv"),
                    alignment_evidence_sha256={str(p):sha256_file(p) for p in [ALIGNMENT / "database_n1_summary.csv", ALIGNMENT / "database_n2_control_summary.csv", ALIGNMENT / "receipt.json"]},
                    configuration_sha256=sha256_file(CONFIG), continuum_preprocessing_version="datcon-monotonic-tail-v1",
                    cnn_kind="cnn_raw", device="cpu", rf_cnn_policy="unchanged canonical defaults", model_refresh=False)
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Eligible: {len(pool)} shots / {len({r['E_series'] for r in pool})} discharges", flush=True)
    for row in chosen:
        print(row["shot"], row["size_bin"], row["input_modes"], flush=True)


def preflight(args):
    check_sources(args)
    chosen = read(HERE / "selection.csv")
    pool = read(HERE / "candidate_pool.csv")
    accepted, rejected, issues = [], [], []
    queue = [r for r in chosen if r["selection_status"] != "excluded_preflight"]
    while queue:
        row = queue.pop(0)
        shot = row["shot"]
        print(f"Preflight {shot}: {row['input_modes']} files", flush=True)
        try:
            check_alignment_snapshot(args, shot)
            result = preprocess_shot(args.data_root / shot)
            rows = [dict(r) for r in result.rows]
            assert len(rows) == int(row["input_modes"]), "N1-N10 file inventory changed"
            bad = [r for r in rows if r["processing_status"] == "INVALID" or r["nr"] != 201]
            for r in bad:
                issues.append(dict(shot=shot, mode_key=r["mode_key"], reason=r["preprocessing_primary_reason"] or "non201", diagnostic=r["diagnostic_message"]))
            assert not bad, f"{len(bad)} invalid or non201 inputs"
            fields = ["mode_key", "path", "input_fingerprint", "nr", "processing_status", "gap_region", "signed_delta", "fraction_below_upper2"]
            write(args.local_dir / f"{shot}_preflight.csv", [{k:r[k] for k in fields} for r in rows])
            row["selection_status"] = "preflight_passed"
            accepted.append(row)
            del result
        except (AssertionError, OSError, ValueError, SystemExit) as exc:
            issues.append(dict(shot=shot, mode_key="", reason="shot_preflight_failed", diagnostic=str(exc)))
            row["selection_status"], row["notes"] = "excluded_preflight", str(exc)
            rejected.append(row)
            used = {r["shot"] for r in accepted+rejected+queue}
            used_series = {r["E_series"] for r in accepted+queue}
            replacement = next((r for r in pool if r["size_bin"] == row["size_bin"] and r["shot"] not in used and r["E_series"] not in used_series), None)
            if replacement is None:
                write(HERE / "selection.csv", accepted+rejected+queue)
                raise RuntimeError(f"No same-stratum replacement for {shot}") from exc
            queue.insert(0, dict(replacement, selection_status="pending_preflight", notes=f"Replacement for {shot}"))
        write(HERE / "selection.csv", accepted+rejected+queue)
        write(HERE / "preflight_issues.csv", issues, ["shot", "mode_key", "reason", "diagnostic"])
    assert len(accepted) == 12
    print(f"Preflight passed: {sum(int(r['input_modes']) for r in accepted)} native nr=201 files", flush=True)


def verify_shot(args, item):
    shot = item["shot"]
    baseline = keyed(read(args.local_dir / f"{shot}_preflight.csv"))
    rd, ad = args.local_dir / "rules" / shot, args.local_dir / "ai" / shot
    rules, ai = keyed(read(rd / "all_modes_rules.csv")), keyed(read(ad / "all_modes_scored.csv"))
    assert set(rules) == set(ai) == set(baseline) == {portable_mode_key(p) for p in files(args.data_root / shot)}
    with (rd / "shot_summary.csv").open() as f:
        summary = dict(csv.reader(f))
    with (ad / "shot_summary.csv").open() as f:
        ai_summary = dict(csv.reader(f))
    assert summary["rule_configuration_sha256"] == sha256_file(CONFIG)
    assert summary["rule_survivor_policy"] == "accept-as-good-v1"
    assert summary["duplicate_rank_method"] == "rule_severity"
    assert summary["continuum_preprocessing_version"] == ai_summary["continuum_preprocessing_version"] == "datcon-monotonic-tail-v1"
    assert not read(rd / "resolution_warnings.csv")
    assert all(r["cluster_status"] == "PROCESSED_RULE_SEVERITY" for r in read(rd / "frequency_clusters.csv"))
    counts, different, comparisons = Counter(), [], []
    for key, r in rules.items():
        a, old = ai[key], baseline[key]
        source = args.data_root / key
        assert input_fingerprint(source, source.parent / f"datcon{int(r['n'])}") == r["input_fingerprint"] == old["input_fingerprint"]
        assert r["nr"] == a["nr"] == old["nr"] == "201"
        assert r["processing_status"] != "INVALID" and a["status"] != "rejected"
        assert r["gap_region"] == a["gap_region"] == old["gap_region"]
        for name in ("signed_delta", "fraction_below_upper2"):
            assert float(r[name]) == float(a[name]) == float(old[name])
        if r["processing_status"] != "RULE_EVALUATED":
            continue
        assert r["severity_complete"] == "True"
        assert math.isfinite(float(r["overall_rule_severity"]))
        assert all(math.isfinite(float(a[k])) and 0 <= float(a[k]) <= 1 for k in ("p_rf_good", "p_cnn_good"))
        decision = a["final_label"].upper()
        assert decision in ("GOOD", "BAD")
        counts["tae_side"] += 1
        counts["rules_good"] += r["final_decision"] == "GOOD"
        counts["ai_good"] += decision == "GOOD"
        counts["rules_final"] += r["selected_final"] == "True"
        counts["ai_final"] += a["selected_final"] == "True"
        counts["rules_vs_rf"] += (r["final_decision"] == "GOOD") != (float(a["p_rf_good"]) >= .5)
        counts["rules_vs_cnn"] += (r["final_decision"] == "GOOD") != (float(a["p_cnn_good"]) >= .5)
        comparison = dict(path=key, shot=shot, mode_key=key, input_fingerprint=r["input_fingerprint"],
            rules_decision=r["final_decision"], rules_reason=r["rule_primary_reason"],
            rf_cnn_decision=decision, p_rf_good=a["p_rf_good"], p_cnn_good=a["p_cnn_good"], rf_cnn_tier=a["tier"],
            overall_rule_severity=r["overall_rule_severity"], rule_margin=r["rule_margin"], nearest_gate=r["nearest_gate"],
            rules_selected_final=r["selected_final"], rf_cnn_selected_final=a["selected_final"])
        comparisons.append(comparison)
        if r["final_decision"] != decision:
            counts["disagreements"] += 1
            counts["rules_bad_ai_good" if decision == "GOOD" else "rules_good_ai_bad"] += 1
            different.append(comparison)
    check_alignment_snapshot(args, shot)
    result = dict(shot=shot, shot_family=item["shot_family"], size_bin=item["size_bin"], input_modes=len(rules),
                  eae_side=len(rules)-counts["tae_side"], **{k:counts[k] for k in
                  ("tae_side", "rules_good", "ai_good", "rules_final", "ai_final", "disagreements", "rules_bad_ai_good", "rules_good_ai_bad", "rules_vs_rf", "rules_vs_cnn")})
    return result, different, comparisons


def run_shot(args, item):
    shot = item["shot"]
    for method, folder in (("rules", "rules"), ("rf-cnn", "ai")):
        out = args.local_dir / folder / shot
        assert not out.exists(), f"Refusing to overwrite staged output {out}"
        command = [sys.executable, str(REPO / "scripts/sort_shot_mixed.py"), "--method", method,
                   "--shot_dir", str(args.data_root / shot), "--out_dir", str(out)]
        if method == "rules":
            command += ["--rule_config", str(CONFIG)]
        else:
            command += ["--rf_model", str(args.rf_model), "--cnn_model", str(args.cnn_model), "--cnn_model_kind", "cnn_raw", "--device", "cpu"]
        print(f"{shot}: {method}", flush=True)
        with (args.local_dir / f"{shot}_{method}.log").open("w") as f:
            subprocess.run(command, cwd=REPO, check=True, stdout=f, stderr=subprocess.STDOUT)
    result = verify_shot(args, item)
    print(f"{shot}: verified, {result[0]['disagreements']}/{result[0]['tae_side']} disagreements", flush=True)
    return result


def stage(args):
    check_sources(args)
    chosen = [r for r in read(HERE / "selection.csv") if r["selection_status"] == "preflight_passed"]
    assert len(chosen) == 12
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(run_shot, args, item):item for item in chosen}
        for future in as_completed(pending):
            result = future.result()
            results[result[0]["shot"]] = result
    save_verified(args, chosen, results)


def save_verified(args, chosen, results):
    check_sources(args)
    summaries, different, comparisons = [], [], []
    for item in chosen:
        result, diffs, allrows = results[item["shot"]]
        summaries.append(result)
        different.extend(diffs)
        comparisons.extend(allrows)
        item["selection_status"] = "verified_staged"
    fields = list(comparisons[0])
    write(HERE / "shot_summary.csv", summaries)
    write(HERE / "disagreements.csv", different, fields)
    write(args.local_dir / "all_comparisons.csv", comparisons, fields)
    rejected = [r for r in read(HERE / "selection.csv") if r["selection_status"] == "excluded_preflight"]
    write(HERE / "selection.csv", chosen+rejected)
    totals = {k:sum(row[k] for row in summaries) for k,v in summaries[0].items() if isinstance(v, int)}
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), totals=totals,
                   trees={f"{folder}/{r['shot']}":tree_hash(args.local_dir/folder/r["shot"]) for r in chosen for folder in ("rules", "ai")},
                   metadata_sha256=sha256_file(HERE / "metadata.json"), paired_coverage_routing_fingerprints_match=True,
                   all_nr_201=True, all_rule_severities_complete=True, no_resolution_or_ranking_fallback=True,
                   table_sha256={name:sha256_file(HERE/name) for name in ("shot_summary.csv", "disagreements.csv")})
    (HERE / "verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(totals, indent=2), flush=True)


def verify(args):
    chosen = [r for r in read(HERE / "selection.csv") if r["selection_status"] in ("preflight_passed", "verified_staged")]
    assert len(chosen) == 12
    save_verified(args, chosen, {r["shot"]:verify_shot(args, r) for r in chosen})


def install(args):
    check_sources(args)
    receipt = json.loads((HERE / "verification.json").read_text())
    assert sha256_file(HERE / "metadata.json") == receipt["metadata_sha256"]
    for name, digest in receipt["table_sha256"].items():
        assert sha256_file(HERE/name) == digest
    destinations = []
    for key, digest in receipt["trees"].items():
        folder, shot = key.split("/")
        source = args.local_dir / key
        root = args.rules_root if folder == "rules" else args.ai_root
        target = root / shot
        assert not target.exists(), f"Existing output: {target}"
        assert tree_hash(source) == digest
        destinations.append((source, target, digest))
    for source, target, digest in destinations:
        temporary = target.with_name("."+target.name+"_pilot12_v11_20260910")
        assert not temporary.exists(), f"Existing transfer directory: {temporary}"
        shutil.copytree(source, temporary)
        assert tree_hash(temporary) == digest
        assert not target.exists()
        temporary.rename(target)
        assert tree_hash(target) == digest
    publication = dict(created_utc=datetime.now(timezone.utc).isoformat(), verification_sha256=sha256_file(HERE / "verification.json"),
                       outputs=[dict(path=str(target), tree_sha256=digest) for _,target,digest in destinations])
    (HERE / "publication.json").write_text(json.dumps(publication, indent=2)+"\n")
    print(f"Installed and verified {len(destinations)} new shot output directories", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "preflight", "stage", "verify", "install"))
    parser.add_argument("--data-root", type=Path, required=True, help="Live DiTw root")
    parser.add_argument("--rules-root", type=Path, required=True, help="Requested production rules output root")
    parser.add_argument("--ai-root", type=Path, required=True, help="Requested RF-CNN output root")
    parser.add_argument("--local-dir", type=Path, required=True, help="Ignored local preflight and staging directory")
    parser.add_argument("--alignment-runtime", type=Path, required=True, help="Frozen database audit runtime with groups and mode_coverage.csv")
    parser.add_argument("--rf-model", type=Path, default=REPO / "models/nova_mode_classifier.joblib")
    parser.add_argument("--cnn-model", type=Path, default=REPO / "models/nova_cnn_raw.pt")
    args = parser.parse_args()
    for name in ("data_root", "rules_root", "ai_root", "local_dir", "alignment_runtime", "rf_model", "cnn_model"):
        setattr(args, name, getattr(args, name).resolve())
    args.local_dir.mkdir(parents=True, exist_ok=True)
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
