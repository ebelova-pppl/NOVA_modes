#!/usr/bin/env python3
"""Rerun inventory-marked shots and compare with saved production outputs.

Example arguments and limitations are documented in README.md beside this file.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from make_tae_like_list import iter_n_dirs, preflight_n_dirs
from tae_rule_config import load_rule_run_configuration


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(path):
    with path.open(newline="") as stream:
        return dict(csv.reader(stream))


def keyed(rows):
    result = {}
    for row in rows:
        key = row.get("mode_key") or "/".join(Path(row["path"]).parts[-3:])
        assert key not in result, f"Duplicate mode key: {key}"
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=REPO / "audits/main_dataset_shots/shot_status.csv",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--ai-root", type=Path, required=True)
    parser.add_argument("--rf-model", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New local directory for full sorter exports and logs",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="Directory for compact comparison evidence",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    inventory = read_csv(args.inventory)
    checked = [row for row in inventory if row["post_training_checked"] == "yes"]
    assert len(checked) == 15 and len(inventory) == 200
    config = REPO / "configs/rules/tae_rules_production_v5.yaml"
    load_rule_run_configuration(config)
    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "configuration_sha256": sha(config),
        "rf_model_sha256": sha(args.rf_model),
        "inventory_sha256": sha(args.inventory),
        "baseline_files": {},
        "scope": "15 marked shots, N1 through N10, egn*; footer grid census on the same 15 selected shots",
        "rf_cnn_reference": "Saved historical predictions; no CNN rerun or independent input fingerprint in AI tables",
    }
    grids, issues = [], []
    # This is a footer/size census, not another mode loader or validity classifier.
    # Full validation and all scientific preprocessing remain in the shared sorter.
    for row in checked:
        shot = row["shot"]
        shot_dir = args.data_root / shot
        counts = Counter()
        n_files = n_errors = 0
        for n, n_dir in iter_n_dirs(shot_dir):
            for path in sorted(n_dir.glob("egn*")):
                n_files += 1
                try:
                    size = path.stat().st_size
                    with path.open("rb") as stream:
                        stream.seek(-24, 2)
                        nr_raw, _, _ = struct.unpack("=ddd", stream.read(24))
                    nr = int(nr_raw)
                    if (
                        nr <= 0
                        or nr != nr_raw
                        or size % 8
                        or size <= 32
                        or (size // 8 - 4) % (3 * nr)
                    ):
                        raise ValueError("invalid nr or payload dimensions")
                    counts[nr] += 1
                    if nr != 201:
                        issues.append(
                            dict(mode_key=f"{shot}/N{n}/{path.name}", issue=f"nr={nr}")
                        )
                except (OSError, ValueError, OverflowError, struct.error) as exc:
                    n_errors += 1
                    issues.append(
                        dict(mode_key=f"{shot}/N{n}/{path.name}", issue=str(exc))
                    )
        grids.append(
            dict(
                shot=shot,
                inventory_status=row["status"],
                post_training_checked=row["post_training_checked"],
                directory_exists=shot_dir.is_dir(),
                input_modes=n_files,
                nr_counts=json.dumps(dict(sorted(counts.items()))),
                footer_or_size_errors=n_errors,
            )
        )
        if len(grids) % 3 == 0:
            print(f"Grid census: {len(grids)}/15 selected shots checked", flush=True)
    for row in grids:
        if row["post_training_checked"] == "yes":
            assert json.loads(row["nr_counts"]) == {"201": row["input_modes"]}
            assert row["footer_or_size_errors"] == 0
    write_csv(args.audit_dir / "grid_census.csv", grids, list(grids[0]))
    write_csv(args.audit_dir / "grid_exceptions.csv", issues, ["mode_key", "issue"])
    print(
        f"Grid census: {sum(r['input_modes'] for r in grids)} files, {len(issues)} exceptions",
        flush=True,
    )

    shot_rows, changes, disagreements = [], [], []
    for index, item in enumerate(checked, 1):
        shot = item["shot"]
        preflight_n_dirs(args.data_root / shot, n_min=1, n_max=10, pattern="egn*")
        old_dir, ai_dir = args.baseline_root / shot, args.ai_root / shot
        baseline_paths = [
            old_dir / "all_modes_rules.csv",
            old_dir / "shot_summary.csv",
            ai_dir / "all_modes_scored.csv",
            ai_dir / "shot_summary.csv",
        ]
        provenance["baseline_files"][shot] = {
            f"{'rules' if p.parent == old_dir else 'rf_cnn'}/{p.name}": sha(p)
            for p in baseline_paths
        }
        old = keyed(read_csv(baseline_paths[0]))
        ai = keyed(read_csv(baseline_paths[2]))
        old_summary = summary(baseline_paths[1])
        assert old_summary["rule_configuration_name"] == "tae_rules_production_v2"
        assert int(old_summary["n_overrides_applied"]) == 0
        out = args.out_dir / shot
        command = [
            sys.executable,
            str(REPO / "scripts/sort_shot_mixed.py"),
            "--method",
            "rules",
            "--shot_dir",
            str(args.data_root / shot),
            "--rule_config",
            str(config),
            "--rf_model",
            str(args.rf_model),
            "--out_dir",
            str(out),
        ]
        print(
            f"[{index}/15] {shot}: running v5 on {len(old)} baseline inputs", flush=True
        )
        with (args.out_dir / f"{shot}.log").open("w") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=REPO
            )
        new = keyed(read_csv(out / "all_modes_rules.csv"))
        run_summary = summary(out / "shot_summary.csv")
        assert (
            run_summary["rule_configuration_sha256"]
            == provenance["configuration_sha256"]
        )
        assert run_summary["rule_survivor_policy"] == "accept-as-good-v1"
        assert set(old) == set(new) == set(ai), f"Input coverage changed: {shot}"
        assert not read_csv(
            out / "resolution_warnings.csv"
        ), f"Resolution warning: {shot}"
        counts = Counter()
        reasons = Counter()
        for key, row in new.items():
            before, reference = old[key], ai[key]
            assert (
                row["input_fingerprint"] == before["input_fingerprint"] != ""
            ), f"Changed input: {key}"
            assert row["processing_status"] == before["processing_status"]
            if row["processing_status"] == "INVALID":
                assert (
                    row["preprocessing_primary_reason"]
                    == before["preprocessing_primary_reason"]
                )
                assert row["final_decision"] == before["final_decision"] == "INVALID"
                assert reference["status"] == "rejected"
                counts["invalid_unchanged"] += 1
                continue
            assert row["nr"] == "201", f"Unexpected grid: {key}"
            assert row["gap_region"] == before["gap_region"] == reference["gap_region"]
            assert row["signed_delta"] == before["signed_delta"]
            assert row["fraction_below_upper2"] == before["fraction_below_upper2"]
            if row["processing_status"] != "RULE_EVALUATED":
                continue
            counts["tae_side"] += 1
            counts["old_good"] += before["final_decision"] == "GOOD"
            counts["v5_good"] += row["final_decision"] == "GOOD"
            counts["old_final_selected"] += before["selected_final"] == "True"
            counts["v5_final_selected"] += row["selected_final"] == "True"
            ai_decision = reference["final_label"].upper()
            assert ai_decision in {"GOOD", "BAD"}
            counts["old_ai_disagreements"] += before["final_decision"] != ai_decision
            counts["v5_ai_disagreements"] += row["final_decision"] != ai_decision
            fields = dict(
                shot=shot,
                mode_key=key,
                input_fingerprint=row["input_fingerprint"],
                old_decision=before["final_decision"],
                v5_decision=row["final_decision"],
                old_reason=before["rule_primary_reason"],
                v5_reason=row["rule_primary_reason"],
                rf_cnn_decision=ai_decision,
                old_selected_final=before["selected_final"],
                v5_selected_final=row["selected_final"],
            )
            decision_changed = row["final_decision"] != before["final_decision"]
            reason_changed = row["rule_primary_reason"] != before["rule_primary_reason"]
            if decision_changed or reason_changed:
                changes.append(fields)
                counts["decision_changes"] += decision_changed
                counts["reason_only_changes"] += reason_changed and not decision_changed
                if decision_changed:
                    reasons[row["rule_primary_reason"]] += 1
            if row["final_decision"] != ai_decision:
                disagreements.append(fields)
            if before["final_decision"] == "BAD":
                assert row["final_decision"] == "BAD", f"Old rejection released: {key}"
        clusters = read_csv(out / "frequency_clusters.csv")
        assert all(
            r["cluster_status"] in {"PROCESSED_RF", "NOT_APPLICABLE_SINGLETON"}
            for r in clusters
        ), f"RF ranking fallback: {shot}"
        shot_rows.append(
            dict(
                shot=shot,
                input_modes=len(new),
                **{
                    k: counts[k]
                    for k in (
                        "tae_side",
                        "invalid_unchanged",
                        "old_good",
                        "v5_good",
                        "old_final_selected",
                        "v5_final_selected",
                        "decision_changes",
                        "reason_only_changes",
                        "old_ai_disagreements",
                        "v5_ai_disagreements",
                    )
                },
                new_rejections_by_gate=json.dumps(dict(reasons), sort_keys=True),
                cluster_statuses=json.dumps(
                    dict(Counter(r["cluster_status"] for r in clusters))
                ),
            )
        )
        print(
            f"  GOOD {counts['old_good']} -> {counts['v5_good']}; decision changes {counts['decision_changes']}; reason-only {counts['reason_only_changes']}",
            flush=True,
        )
    write_csv(args.audit_dir / "shot_summary.csv", shot_rows, list(shot_rows[0]))
    fieldnames = list(fields)
    write_csv(args.audit_dir / "changes.csv", changes, fieldnames)
    write_csv(
        args.audit_dir / "newly_rejected.csv",
        [
            row
            for row in changes
            if row["old_decision"] == "GOOD" and row["v5_decision"] == "BAD"
        ],
        fieldnames,
    )
    write_csv(args.audit_dir / "rf_cnn_disagreements.csv", disagreements, fieldnames)
    provenance["totals"] = {
        key: sum(row[key] for row in shot_rows)
        for key in shot_rows[0]
        if isinstance(shot_rows[0][key], int)
    }
    provenance["grid_input_files"] = sum(row["input_modes"] for row in grids)
    provenance["grid_exceptions"] = len(issues)
    (args.audit_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
