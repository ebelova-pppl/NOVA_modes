"""Check regenerated paired exports against the reviewed candidate audit.

Run after regenerate_batch.py, before publishing any exports. Paths are CLI
arguments so the verifier also works against backups on another host.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from cont_features import CONTINUUM_PREPROCESSING_VERSION
from tae_eae_features import classify_gap_region
from tae_rule_io import portable_mode_key, sha256_file, write_dict_csv


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "out-root",
        "old-pilot-root",
        "old-regression-root",
        "candidate-details",
        "audit-dir",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    expected_candidate = {}
    with args.candidate_details.open() as handle:
        for line in handle:
            row = json.loads(line)
            expected_candidate[row["mode_key"]] = row
    assert len(expected_candidate) == 4083
    memberships = [
        (REPO / "audits/regression15_v5/shot_summary.csv", args.old_regression_root),
        (REPO / "audits/pilot12_v5_20260908/shot_summary.csv", args.old_pilot_root),
    ]
    checks = Counter()
    counts = Counter()
    by_shot, disagreements = [], []
    for membership, old_root in memberships:
        for shot_row in read_csv(membership):
            shot = shot_row["shot"]
            old = {
                r["mode_key"]: r
                for r in read_csv(old_root / shot / "all_modes_rules.csv")
            }
            rules_dir, ai_dir = (args.out_root / m / shot for m in ("rules", "rf-cnn"))
            rules = {
                r["mode_key"]: r for r in read_csv(rules_dir / "all_modes_rules.csv")
            }
            ai = {
                portable_mode_key(r["path"]): r
                for r in read_csv(ai_dir / "all_modes_scored.csv")
            }
            assert set(old) == set(rules) == set(ai)
            assert len(rules) == int(shot_row["input_modes"])
            shot_counts = Counter()
            for key, row in rules.items():
                prior, prediction = old[key], ai[key]
                assert row["input_fingerprint"] == prior["input_fingerprint"], key
                if prior["processing_status"] == "INVALID":
                    assert row["processing_status"] == "INVALID"
                    assert prediction["status"] != "scored"
                    checks["invalid_preserved"] += 1
                    continue
                for field in ("omega", "gamma_d", "nr", "nhar", "ntor"):
                    assert row[field] == prior[field] == prediction[field], (key, field)
                assert row["nr"] == "201"
                for field in ("gap_region", "signed_delta", "fraction_below_upper2"):
                    assert row[field] == prediction[field], (key, field)
                checks["paired_valid_mode_matches"] += 1
                reference = expected_candidate.get(key)
                if reference:
                    expected = reference["last"]
                    assert row["input_fingerprint"] == reference["input_fingerprint"]
                    checks["candidate_mode_matches"] += 1
                else:
                    scalars = {
                        k: float(prior[k])
                        for k in ("signed_delta", "fraction_below_upper2")
                    }
                    route = classify_gap_region(**scalars)
                    expected = dict(gap_region=route, **scalars)
                    expected["decision"] = (
                        "ROUTED_EAE" if route == "eae_like" else prior["rule_decision"]
                    )
                    expected["reason"] = (
                        "frequency_routing"
                        if route == "eae_like"
                        else prior["rule_primary_reason"]
                    )
                assert row["gap_region"] == expected["gap_region"], key
                for field in ("signed_delta", "fraction_below_upper2"):
                    assert float(row[field]) == expected[field], (key, field)
                if expected["decision"] == "ROUTED_EAE":
                    assert row["processing_status"] == "ROUTED_EAE"
                    assert prediction["status"] != "scored"
                    counts["eae_like"] += 1
                    continue
                assert row["rule_decision"] == expected["decision"], key
                assert row["rule_primary_reason"] == expected["reason"], key
                expected_features = (
                    reference["last_features"]
                    if reference
                    else json.loads(prior["rule_features"])
                )
                assert json.loads(row["rule_features"]) == expected_features, key
                checks["exact_rule_feature_matches"] += 1
                expected_final = "GOOD" if expected["decision"] == "REVIEW" else "BAD"
                assert row["final_decision"] == expected_final, key
                counts["rules_" + expected_final] += 1
                shot_counts["rules_" + expected_final] += 1
                assert prediction["status"] == "scored"
                counts["ai_" + prediction["final_label"]] += 1
                shot_counts["ai_" + prediction["final_label"]] += 1
                if expected_final.lower() != prediction["final_label"]:
                    disagreements.append(
                        dict(
                            shot=shot,
                            mode_key=key,
                            input_fingerprint=row["input_fingerprint"],
                            rules_decision=expected_final,
                            rules_reason=row["rule_primary_reason"],
                            rf_cnn_decision=prediction["final_label"].upper(),
                            p_rf_good=prediction["p_rf_good"],
                            p_cnn_good=prediction["p_cnn_good"],
                            rf_cnn_tier=prediction["tier"],
                        )
                    )
                    shot_counts["disagreements"] += 1
            summaries = {}
            for method, directory in [("rules", rules_dir), ("rf-cnn", ai_dir)]:
                summaries[method] = read_csv(directory / "shot_summary_wide.csv")[0]
                for summary in [
                    summaries[method],
                    *read_csv(directory / "shot_summary_by_n.csv"),
                ]:
                    assert (
                        summary["continuum_preprocessing_version"]
                        == CONTINUUM_PREPROCESSING_VERSION
                    )
                counts[method + "_selected_good"] += int(
                    summaries[method]["n_final_good"]
                )
            assert (
                summaries["rules"]["n_interior_harmonic_resolution_ineligible"] == "0"
            )
            assert (
                summaries["rules"]["n_continuum_crossing_tail_resolution_ineligible"]
                == "0"
            )
            clusters = read_csv(rules_dir / "frequency_clusters.csv")
            assert all(r["cluster_status"] == "PROCESSED_RF" for r in clusters), shot
            by_shot.append(
                dict(
                    shot=shot,
                    input_modes=len(rules),
                    rules_good=shot_counts["rules_GOOD"],
                    ai_good=shot_counts["ai_good"],
                    rules_selected=int(summaries["rules"]["n_final_good"]),
                    ai_selected=int(summaries["rf-cnn"]["n_final_good"]),
                    disagreements=shot_counts["disagreements"],
                )
            )
            print(shot, "verified", len(rules), "inputs", flush=True)
    assert checks["paired_valid_mode_matches"] == 19228
    assert checks["candidate_mode_matches"] == 4083
    assert checks["invalid_preserved"] == 97
    assert counts["rules_GOOD"] == 940 and counts["rules_BAD"] == 3327
    assert counts["eae_like"] == 14961
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(
        args.audit_dir / "regenerated_shot_summary.csv", list(by_shot[0]), by_shot
    )
    write_dict_csv(
        args.audit_dir / "regenerated_disagreements.csv",
        list(disagreements[0]),
        disagreements,
    )
    verification = dict(
        status="adopted_and_regenerated_locally",
        continuum_preprocessing_version=CONTINUUM_PREPROCESSING_VERSION,
        shots=27,
        canonical_runs=54,
        checks=dict(checks),
        counts=dict(counts),
        disagreements=len(disagreements),
        candidate_details_sha256=sha256_file(args.candidate_details),
        regeneration_provenance=json.loads(
            (args.out_root / "run_provenance.json").read_text()
        ),
    )
    (args.audit_dir / "adoption_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                checks=dict(checks),
                counts=dict(counts),
                disagreements=len(disagreements),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
