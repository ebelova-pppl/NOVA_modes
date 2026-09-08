#!/usr/bin/env python3
"""Select, preflight, and run a frozen paired pilot; see adjacent README.md."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from make_tae_like_list import preprocess_shot
from tae_rule_io import input_fingerprint, portable_mode_key, sha256_file

SEED = 20260908
QUOTAS = [
    ("NSTX-U E", "low", 2),
    ("NSTX-U E", "medium", 3),
    ("NSTX-U E", "high", 3),
    ("NSTX-U G", "medium", 2),
    ("NSTX-U G", "high", 2),
]


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def series(shot):
    match = re.match(r"nstxuE(\d+)", shot)
    return match.group(1) if match else ""


def select(args):
    inventory = read(args.inventory)
    seen = {
        series(r["shot"])
        for r in inventory
        if r["active_training_shot"] == "yes" or r["post_training_checked"] == "yes"
    } - {""}
    pool, excluded = [], []
    for row in inventory:
        shot, reason = row["shot"], ""
        if (
            row["active_training_shot"] == "yes"
            or row["post_training_checked"] == "yes"
            or row["status"] != "unchecked"
        ):
            reason = row["status"]
        elif series(shot) in seen:
            reason = "E_series_already_training_or_checked"
        elif any((root / shot).exists() for root in (args.rules_root, args.ai_root)):
            reason = "existing_output_directory"
        if reason:
            excluded.append(dict(shot=shot, reason=reason))
            continue
        count = sum(
            len(list((args.data_root / shot / f"N{n}").glob("egn*")))
            for n in range(1, 11)
        )
        if not count:
            excluded.append(dict(shot=shot, reason="empty_no_egn"))
            continue
        pool.append(
            dict(
                shot=shot,
                shot_family=row["shot_family"],
                size_bin=(
                    "low" if count <= 400 else "medium" if count <= 800 else "high"
                ),
                input_modes=count,
                E_series=series(shot),
            )
        )
    rng, ordered, selected, used = random.Random(SEED), [], [], set()
    for family, size, quota in QUOTAS:
        group = sorted(
            [r for r in pool if r["shot_family"] == family and r["size_bin"] == size],
            key=lambda r: r["shot"],
        )
        rng.shuffle(group)
        for rank, row in enumerate(group, 1):
            row["stratum_rank"] = rank
            ordered.append(row)
        chosen = []
        for row in group:
            if row["E_series"] and row["E_series"] in used:
                continue
            chosen.append(
                dict(
                    row,
                    selection_seed=SEED,
                    selection_status="selected_pending_preflight",
                    notes="",
                )
            )
            if row["E_series"]:
                used.add(row["E_series"])
            if len(chosen) == quota:
                break
        assert len(chosen) == quota, (family, size, len(chosen), quota)
        selected.extend(chosen)
    write(args.audit_dir / "candidate_pool.csv", ordered)
    write(args.audit_dir / "excluded_from_pool.csv", excluded)
    write(args.audit_dir / "selection.csv", selected)
    metadata = dict(
        seed=SEED,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        training_labels_sha256=sha256_file(REPO / "training_labels/tae_like_train.csv"),
        inventory_sha256=sha256_file(args.inventory),
        quotas=QUOTAS,
        selection_method="Sorted candidate strata shuffled with random.Random(seed), in quota order; unique E discharge numbers, excluding previously checked/training E series; no eligible small G cases.",
        configuration_sha256=sha256_file(
            REPO / "configs/rules/tae_rules_production_v5.yaml"
        ),
        rf_model_sha256=sha256_file(args.rf_model),
        cnn_model_sha256=sha256_file(args.cnn_model),
    )
    (args.audit_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for row in selected:
        print(row["shot"], row["size_bin"], row["input_modes"], flush=True)


def preflight(args):
    selected = read(args.audit_dir / "selection.csv")
    pool = read(args.audit_dir / "candidate_pool.csv")
    accepted, failures = [], []
    rejected = [r for r in selected if r["selection_status"] == "excluded_preflight"]
    if rejected and (args.audit_dir / "preflight_issues.csv").exists():
        failures = read(args.audit_dir / "preflight_issues.csv")
    queue = [r for r in selected if r["selection_status"] != "excluded_preflight"]
    while queue:
        row = queue.pop(0)
        shot = row["shot"]
        print(f"Preflight {shot} ({row['input_modes']} modes)", flush=True)
        result = preprocess_shot(args.data_root / shot)
        rows = [dict(r) for r in result.rows]
        assert len(rows) == int(row["input_modes"])
        invalid = [r for r in rows if r["processing_status"] == "INVALID"]
        non201 = [
            r for r in rows if r["processing_status"] != "INVALID" and r["nr"] != 201
        ]
        for r in invalid + non201:
            failures.append(
                dict(
                    shot=shot,
                    mode_key=r["mode_key"],
                    input_fingerprint=r["input_fingerprint"],
                    reason=r["preprocessing_primary_reason"] or "unsupported_nr",
                    diagnostic=r["diagnostic_message"] or str(r["nr"]),
                )
            )
        fields = [
            "mode_key",
            "path",
            "input_fingerprint",
            "nr",
            "processing_status",
            "gap_region",
            "signed_delta",
            "fraction_below_upper2",
        ]
        write(
            args.local_dir / f"{shot}_preflight.csv",
            [{key: r[key] for key in fields} for r in rows],
        )
        del result
        if invalid or non201:
            row["selection_status"] = "excluded_preflight"
            row["notes"] = (
                f"{len(invalid)} invalid and {len(non201)} unsupported-resolution inputs; no sorter run"
            )
            rejected.append(row)
            used_shots = {r["shot"] for r in accepted + rejected + queue}
            used_series = {r["E_series"] for r in accepted + queue} - {""}
            replacement = next(
                r
                for r in pool
                if r["shot_family"] == row["shot_family"]
                and r["size_bin"] == row["size_bin"]
                and r["shot"] not in used_shots
                and (not r["E_series"] or r["E_series"] not in used_series)
            )
            queue.insert(
                0,
                dict(
                    replacement,
                    selection_seed=SEED,
                    selection_status="selected_pending_preflight",
                    notes=f"Replacement for {shot}",
                ),
            )
            print(
                f"  Excluded before model/rule inference; replacement {replacement['shot']}",
                flush=True,
            )
        else:
            row["selection_status"] = "preflight_passed"
            accepted.append(row)
            print("  All inputs valid, nr=201, continuum loaded", flush=True)
        write(args.audit_dir / "preflight_progress.csv", accepted + rejected + queue)
        write(
            args.audit_dir / "preflight_issues.csv",
            failures,
            ["shot", "mode_key", "input_fingerprint", "reason", "diagnostic"],
        )
    assert len(accepted) == 12
    write(args.audit_dir / "selection.csv", accepted + rejected)
    (args.audit_dir / "preflight_progress.csv").unlink()


def keyed(rows):
    result = {r.get("mode_key") or portable_mode_key(r["path"]): r for r in rows}
    assert len(result) == len(rows)
    return result


def run(args):
    selection = read(args.audit_dir / "selection.csv")
    chosen = [r for r in selection if r["selection_status"] == "preflight_passed"]
    assert len(chosen) == 12
    metadata = json.loads((args.audit_dir / "metadata.json").read_text())
    assert sha256_file(args.rf_model) == metadata["rf_model_sha256"]
    assert sha256_file(args.cnn_model) == metadata["cnn_model_sha256"]
    config = REPO / "configs/rules/tae_rules_production_v5.yaml"
    assert sha256_file(config) == metadata["configuration_sha256"]
    for r in chosen:
        for root in (args.rules_root, args.ai_root):
            assert not (
                root / r["shot"]
            ).exists(), f"Output already exists: {root / r['shot']}"
    summaries, disagreements, counts_by_reason = [], [], Counter()
    for index, item in enumerate(chosen, 1):
        shot = item["shot"]
        baseline = keyed(read(args.local_dir / f"{shot}_preflight.csv"))
        for method, root in (("rules", args.rules_root), ("rf-cnn", args.ai_root)):
            command = [
                sys.executable,
                str(REPO / "scripts/sort_shot_mixed.py"),
                "--method",
                method,
                "--shot_dir",
                str(args.data_root / shot),
                "--rf_model",
                str(args.rf_model),
                "--out_dir",
                str(root / shot),
            ]
            command += (
                ["--rule_config", str(config)]
                if method == "rules"
                else [
                    "--cnn_model",
                    str(args.cnn_model),
                    "--cnn_model_kind",
                    "cnn_raw",
                    "--device",
                    "cpu",
                ]
            )
            print(f"[{index}/12] {shot}: {method}", flush=True)
            with (args.local_dir / f"{shot}_{method}.log").open("w") as log:
                subprocess.run(
                    command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=REPO
                )
        rule_dir, ai_dir = args.rules_root / shot, args.ai_root / shot
        rules = keyed(read(rule_dir / "all_modes_rules.csv"))
        ai = keyed(read(ai_dir / "all_modes_scored.csv"))
        assert set(rules) == set(ai) == set(baseline)
        with (rule_dir / "shot_summary.csv").open() as stream:
            summary = dict(csv.reader(stream))
        assert summary["rule_configuration_sha256"] == metadata["configuration_sha256"]
        assert summary["rule_survivor_policy"] == "accept-as-good-v1"
        assert not read(rule_dir / "resolution_warnings.csv")
        assert all(
            r["cluster_status"] == "PROCESSED_RF"
            for r in read(rule_dir / "frequency_clusters.csv")
        )
        counts = Counter()
        for key, r in rules.items():
            a, old = ai[key], baseline[key]
            # Check the actual inputs again after BOTH methods, since AI exports
            # do not carry their own mode-plus-continuum fingerprint.
            source = args.data_root / key
            assert (
                input_fingerprint(source, source.parent / f"datcon{int(r['n'])}")
                == r["input_fingerprint"]
                == old["input_fingerprint"]
            )
            assert r["nr"] == a["nr"] == old["nr"] == "201"
            assert r["processing_status"] != "INVALID" and a["status"] != "rejected"
            assert r["gap_region"] == a["gap_region"] == old["gap_region"]
            for field in ("signed_delta", "fraction_below_upper2"):
                assert float(r[field]) == float(a[field]) == float(old[field])
            if r["processing_status"] != "RULE_EVALUATED":
                continue
            counts["tae_side"] += 1
            decision = a["final_label"].upper()
            assert decision in {"GOOD", "BAD"}
            counts["rules_good"] += r["final_decision"] == "GOOD"
            counts["ai_good"] += decision == "GOOD"
            counts["rules_final"] += r["selected_final"] == "True"
            counts["ai_final"] += a["selected_final"] == "True"
            if r["final_decision"] != decision:
                counts["disagreements"] += 1
                direction = (
                    "rules_bad_ai_good" if decision == "GOOD" else "rules_good_ai_bad"
                )
                counts[direction] += 1
                counts_by_reason[f"{direction}:{r['rule_primary_reason']}"] += 1
                disagreements.append(
                    dict(
                        shot=shot,
                        mode_key=key,
                        input_fingerprint=r["input_fingerprint"],
                        rules_decision=r["final_decision"],
                        rules_reason=r["rule_primary_reason"],
                        rf_cnn_decision=decision,
                        p_rf_good=a["p_rf_good"],
                        p_cnn_good=a["p_cnn_good"],
                        rf_cnn_tier=a["tier"],
                    )
                )
        summaries.append(
            dict(
                shot=shot,
                shot_family=item["shot_family"],
                size_bin=item["size_bin"],
                input_modes=len(rules),
                **{
                    k: counts[k]
                    for k in (
                        "tae_side",
                        "rules_good",
                        "ai_good",
                        "rules_final",
                        "ai_final",
                        "disagreements",
                        "rules_bad_ai_good",
                        "rules_good_ai_bad",
                    )
                },
            )
        )
        print(
            f"  {counts['tae_side']} TAE-side, {counts['disagreements']} disagreements; final GOOD rules/AI {counts['rules_final']}/{counts['ai_final']}",
            flush=True,
        )
        item["selection_status"] = "completed"
        write(args.audit_dir / "selection.csv", selection)
        write(args.audit_dir / "shot_summary.csv", summaries)
        write(
            args.audit_dir / "disagreements.csv",
            disagreements,
            [
                "shot",
                "mode_key",
                "input_fingerprint",
                "rules_decision",
                "rules_reason",
                "rf_cnn_decision",
                "p_rf_good",
                "p_cnn_good",
                "rf_cnn_tier",
            ],
        )
    metadata["totals"] = {
        k: sum(r[k] for r in summaries)
        for k in summaries[0]
        if isinstance(summaries[0][k], int)
    }
    metadata["disagreement_reasons"] = dict(counts_by_reason)
    (args.audit_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["totals"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["select", "preflight", "run"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rules-root", type=Path, required=True)
    parser.add_argument("--ai-root", type=Path, required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=REPO / "audits/main_dataset_shots/shot_status.csv",
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--rf-model", type=Path, default=REPO / "models/nova_mode_classifier.joblib"
    )
    parser.add_argument(
        "--cnn-model", type=Path, default=REPO / "models/nova_cnn_raw.pt"
    )
    args = parser.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    {"select": select, "preflight": preflight, "run": run}[args.phase](args)


if __name__ == "__main__":
    main()
