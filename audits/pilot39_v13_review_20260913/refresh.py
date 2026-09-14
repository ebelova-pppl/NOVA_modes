"""Refresh review lists from saved rules and RF-CNN exports; no classification.

Example: python audits/pilot39_v13_review_20260913/refresh.py
  --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai
  --data-root /path/to/DiTw --out-dir /path/to/new-review
  --runtime-dir outputs/review_pilot39_v13_new
Review worksheets are never overwritten. Use a fresh destination to refresh.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from tae_rule_io import datcon_path_for_mode, input_fingerprint, portable_mode_key, sha256_file
from tae_rule_config import PRODUCTION_RULE_CONFIG_SHA256

csv.field_size_limit(10**8)
FIELDS = ["path", "label", "shot", "ntor", "mode_filename", "omega",
          "rules_decision", "rules_reason", "rf_cnn_decision", "rf_decision", "cnn_decision",
          "p_rf_good", "p_cnn_good", "rf_cnn_tier", "overall_rule_severity", "nearest_gate",
          "rules_selected_final", "rf_cnn_selected_final", "original_rule_decision",
          "decision_source", "input_fingerprint", "mode_key"]
EXCLUDED_FIELDS = ["path", "shot", "rules_decision", "rules_reason", "input_fingerprint", "reason"]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def keyed(path):
    rows = read(path)
    result = {r.get("mode_key") or portable_mode_key(r["path"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate mode keys: {path}")
    return result


def write(path, rows, fields=FIELDS):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rules-root", "ai-root", "data-root", "out-dir", "runtime-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if list(args.out_dir.glob("*.csv")) or (args.out_dir / "verification.json").exists():
        raise ValueError("Choose a fresh review directory; existing review files are preserved")
    args.runtime_dir.mkdir(parents=True, exist_ok=False)
    by_shot = args.runtime_dir / "by_shot"
    by_shot.mkdir()
    inventory = REPO / "audits/main_dataset_shots/shot_status.csv"
    previous = REPO / "audits/distributed_harmonic_noise_20260913/adoption/current_disagreements.csv"
    current = REPO / "audits/morphology_v13_20260913/current_disagreements.csv"
    shots = sorted(r["shot"] for r in read(inventory) if r["post_training_checked"] == "yes")
    assert len(shots) == 39
    sources = [inventory, previous, current, Path(__file__).resolve()]
    for shot in shots:
        sources += [args.rules_root / shot / "all_modes_rules.csv",
                    args.rules_root / shot / "shot_summary_wide.csv",
                    args.ai_root / shot / "all_modes_scored.csv"]
    source_hashes = {str(p): sha256_file(p) for p in sources}
    comparisons, excluded, summaries = [], [], []
    for shot in shots:
        rules = keyed(args.rules_root / shot / "all_modes_rules.csv")
        ai = keyed(args.ai_root / shot / "all_modes_scored.csv")
        summary = read(args.rules_root / shot / "shot_summary_wide.csv")[0]
        assert summary["rule_configuration_sha256"] == PRODUCTION_RULE_CONFIG_SHA256, shot
        counts = Counter()
        shot_rows = []
        for key, r in rules.items():
            counts["inputs"] += 1
            counts[r["processing_status"].lower()] += 1
            if r["processing_status"] != "RULE_EVALUATED":
                continue
            counts["rules_good_before_dedup"] += r["final_decision"] == "GOOD"
            counts["rules_selected_good"] += r["selected_final"] == "True"
            counts["manual_overrides_applied"] += r["override_status"] == "APPLIED"
            source = args.data_root / key
            assert input_fingerprint(source, datcon_path_for_mode(source)) == r["input_fingerprint"], key
            a = ai.get(key)
            if r["final_decision"] not in ("GOOD", "BAD"):
                reason = "RULE_FINAL_NOT_BINARY"
            elif a is None or a["final_label"].upper() not in ("GOOD", "BAD"):
                reason = "AI_INPUTS_NOT_CURRENT_OR_NOT_CLASSIFIED"
            else:
                reason = None
            if reason:
                counts["unpaired_tae"] += 1
                excluded.append(dict(path=key, shot=shot, rules_decision=r["final_decision"],
                    rules_reason=r["rule_primary_reason"], input_fingerprint=r["input_fingerprint"], reason=reason))
                continue
            for field in ("omega", "nr", "signed_delta", "fraction_below_upper2"):
                assert float(a[field]) == float(r[field]), (key, field)
            assert a["gap_region"] == r["gap_region"], key
            rf, cnn = float(a["p_rf_good"]), float(a["p_cnn_good"])
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in (rf, cnn)), key
            row = dict(path=key, label=r["final_decision"].lower(), shot=shot, ntor=r["ntor"],
                mode_filename=source.name, omega=r["omega"], rules_decision=r["final_decision"],
                rules_reason=r["rule_primary_reason"], rf_cnn_decision=a["final_label"].upper(),
                rf_decision="GOOD" if rf >= .5 else "BAD", cnn_decision="GOOD" if cnn >= .5 else "BAD",
                p_rf_good=a["p_rf_good"], p_cnn_good=a["p_cnn_good"], rf_cnn_tier=a["tier"],
                overall_rule_severity=r["overall_rule_severity"], nearest_gate=r["nearest_gate"],
                rules_selected_final=r["selected_final"], rf_cnn_selected_final=a["selected_final"],
                original_rule_decision=r["rule_decision"], decision_source=r["decision_source"],
                input_fingerprint=r["input_fingerprint"], mode_key=key)
            shot_rows.append(row)
            counts["paired_tae"] += 1
            for model in ("rf_cnn", "rf", "cnn"):
                counts["rules_vs_" + model] += row["rules_decision"] != row[model + "_decision"]
            if row["rules_decision"] != row["rf_cnn_decision"]:
                counts["rules_bad_ai_good" if row["rules_decision"] == "BAD" else "rules_good_ai_bad"] += 1
        shot_rows.sort(key=lambda r: (int(r["ntor"]), float(r["omega"]), r["mode_filename"]))
        comparisons.extend(shot_rows)
        write(by_shot / (shot + ".csv"), [r for r in shot_rows if r["rules_decision"] != r["rf_cnn_decision"]])
        summaries.append(dict(shot=shot, **{k: counts[k] for k in (
            "inputs", "rule_evaluated", "routed_eae", "invalid", "paired_tae", "unpaired_tae",
            "rules_good_before_dedup", "rules_selected_good", "manual_overrides_applied",
            "rules_vs_rf_cnn", "rules_bad_ai_good", "rules_good_ai_bad", "rules_vs_rf", "rules_vs_cnn")}))
        print(f"{shot}: {counts['rules_vs_rf_cnn']} disagreements / {counts['paired_tae']} pairs", flush=True)
    different = [r for r in comparisons if r["rules_decision"] != r["rf_cnn_decision"]]
    fresh = {r["mode_key"]: r for r in different}
    saved = keyed(current)
    assert fresh.keys() == saved.keys(), "Unexpected changes since verified v13 installation"
    for key, r in fresh.items():
        for field in ("input_fingerprint", "rules_decision", "rules_reason", "rf_cnn_decision", "p_rf_good", "p_cnn_good"):
            assert r[field] == saved[key][field], (key, field)
    old = keyed(previous)
    full = {r["mode_key"]: r for r in comparisons}
    added = [r for r in different if r["mode_key"] not in old]
    removed = [full[k] for k in old if k not in fresh]
    assert source_hashes == {str(p): sha256_file(p) for p in sources}
    write(args.out_dir / "disagreements.csv", different)
    write(args.out_dir / "rules_bad_ai_good.csv", [r for r in different if r["rules_decision"] == "BAD"])
    write(args.out_dir / "rules_good_ai_bad.csv", [r for r in different if r["rules_decision"] == "GOOD"])
    write(args.out_dir / "disagreements_elena.csv",
          [dict(r, manual_decision="", manual_reason="") for r in different], FIELDS + ["manual_decision", "manual_reason"])
    for model in ("rf", "cnn"):
        write(args.out_dir / f"rules_vs_{model}.csv", [r for r in comparisons if r["rules_decision"] != r[model + "_decision"]])
    write(args.out_dir / "added_since_v12.csv", added)
    write(args.out_dir / "removed_since_v12.csv", removed)
    write(args.out_dir / "shot_summary.csv", summaries, list(summaries[0]))
    write(args.out_dir / "ai_comparison_excluded.csv", excluded, EXCLUDED_FIELDS)
    write(args.runtime_dir / "all_comparisons.csv", comparisons)
    counts = {k: sum(r[k] for r in summaries) for k in summaries[0] if k != "shot"}
    receipt = dict(status="verified", created_utc=datetime.now(timezone.utc).isoformat(),
        shots=shots, counts=counts, rule_configuration_sha256=PRODUCTION_RULE_CONFIG_SHA256,
        source_sha256=source_hashes, unchanged_since_v13=True,
        added_since_v12=len(added), removed_since_v12=len(removed),
        comparison_policy="Final GOOD/BAD before deduplication; saved ensemble decision; standalone RF/CNN >=0.5; no new manual labels.",
        rules_root=str(args.rules_root), ai_root=str(args.ai_root), data_root=str(args.data_root),
        runtime_dir=str(args.runtime_dir),
        output_sha256={p.name: sha256_file(p) for p in sorted(args.out_dir.glob("*.csv"))})
    (args.out_dir / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
