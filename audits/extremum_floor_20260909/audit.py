"""Audit stricter clearance/width requirements for the extremum exception.

python audits/extremum_floor_20260909/audit.py --rules-root /path/to/sort_outputs \
  --training-root /path/to/training/data --out-dir outputs/review_extremum_floor_20260909
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from input_validity import load_input_validity_registry
from make_tae_like_list import _inspect_mode_file, _load_gap_data
from tae_eae_features import classify_gap_region
from tae_rule_config import PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_engine import evaluate_mode, ContinuumCrossingConfig
from tae_rule_io import (
    input_fingerprint,
    datcon_path_for_mode,
    sha256_file,
    stable_json,
)

REGISTRY = load_input_validity_registry()
CLEARANCE_FLOOR = 0.001
WIDTH_FLOORS = (1.0, 1.1)
FIELDS = [
    "path",
    "mode_key",
    "cohort",
    "training_label",
    "input_fingerprint",
    "nr",
    "baseline_decision",
    "baseline_reason",
    "selected_final",
    "energy_peak_r",
    "energy_fwhm_grid",
    "ext_dr",
    "ext_df_gap",
    "extremum_exception",
    "fails_clearance",
    "fails_width_1_0",
    "fails_width_1_1",
    "newly_rejected_width_1_0",
    "newly_rejected_width_1_1",
]


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def measure(task):
    cohort, supplied, root = task
    key = supplied["path"] if cohort == "training" else supplied["mode_key"]
    path = root / key if cohort == "training" else Path(supplied["path"])
    n = int(path.parent.name[1:])
    fingerprint = input_fingerprint(path, datcon_path_for_mode(path, n))
    record = dict(
        path=str(path),
        mode_key=key,
        cohort=cohort,
        training_label=supplied["validity"] if cohort == "training" else "",
        input_fingerprint=fingerprint,
        selected_final=supplied.get("selected_final", ""),
    )
    if cohort == "shots":
        assert fingerprint == supplied["input_fingerprint"], key
    bundle, reason, message = _inspect_mode_file(path, expected_n=n)
    if bundle is None:
        assert cohort == "training" and key == "nstxuG121123K51/N4/egn04w.8769E+01"
        return dict(record, baseline_decision="INVALID", baseline_reason=reason)
    record["nr"] = bundle["nr"]
    assert bundle["nr"] == 201, key
    assert REGISTRY.diagnostic(path.parents[1].name, n) is None, key
    gap, reason, message = _load_gap_data(
        path, mode=bundle["mode"], omega=bundle["omega"]
    )
    assert gap is not None, (key, reason, message)
    region = classify_gap_region(**gap.scalars)
    if region == "eae_like":
        assert cohort == "training"
        return dict(
            record, baseline_decision="ROUTED_EAE", baseline_reason="ROUTED_EAE"
        )
    row = dict(
        path=str(path),
        mode_key=key,
        shot=path.parents[1].name,
        omega=bundle["omega"],
        gamma_d=bundle["gamma_d"],
        ntor=n,
        gap_region=region,
        input_fingerprint=fingerprint,
    )
    result = evaluate_mode(
        row,
        mode=bundle["mode"],
        low2=gap.low2,
        high2=gap.high2,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
    )
    assert result.decision in {"REVIEW", "BAD"}, (key, result.diagnostic_message)
    if cohort == "shots":
        assert result.decision == supplied["rule_decision"] == "REVIEW"
        assert json.loads(stable_json(result.features)) == json.loads(
            supplied["rule_features"]
        ), key
    e = result.features["resolution_features"]["interior_unresolved_envelope"]
    record.update(
        baseline_decision="GOOD" if result.decision == "REVIEW" else "BAD",
        baseline_reason=result.primary_reason,
        energy_peak_r=e["energy_peak_r"],
        energy_fwhm_grid=e["energy_halfmax_width_grid"],
        ext_dr=e["ext_dr"],
        ext_df_gap=e["ext_df_gap"],
        extremum_exception=e["extremum_exception_applied"],
    )
    # Tightening only this exception can change a surviving narrow candidate.
    protected = result.decision == "REVIEW" and e["extremum_exception_applied"]
    fails_clearance = e["ext_df_gap"] is not None and e["ext_df_gap"] <= CLEARANCE_FLOOR
    record["fails_clearance"] = fails_clearance
    for floor in WIDTH_FLOORS:
        suffix = str(floor).replace(".", "_")
        fails_width = (
            e["energy_halfmax_width_grid"] is not None
            and e["energy_halfmax_width_grid"] <= floor
        )
        record["fails_width_" + suffix] = fails_width
        record["newly_rejected_width_" + suffix] = bool(
            protected and (fails_clearance or fails_width)
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rules-root", "training-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        REPO / p
        for p in (
            "scripts/tae_rule_engine.py",
            "scripts/make_tae_like_list.py",
            "src/cont_features.py",
            "src/tae_eae_features.py",
            "src/nova_mode_loader.py",
            "src/input_validity.py",
            "configs/known_invalid_inputs.csv",
            "configs/rules/tae_rules_production_v7.yaml",
            "training_labels/tae_like_train.csv",
            "audits/continuum_monotonic_tail_20260908/regenerated_shot_summary.csv",
        )
    ] + [Path(__file__)]
    sources_hashes = {str(p.relative_to(REPO)): sha256_file(p) for p in sources}
    assert (
        sources_hashes["configs/rules/tae_rules_production_v7.yaml"]
        == PRODUCTION_RULE_CONFIG_SHA256
    )
    shots = read(sources[-2])
    assert len(shots) == 27
    candidates, counts, export_hashes = [], Counter(), {}
    for s in shots:
        path = args.rules_root / s["shot"] / "all_modes_rules.csv"
        export_hashes[str(path)] = sha256_file(path)
        summary = read(path.with_name("shot_summary_wide.csv"))[0]
        assert summary["rule_configuration_sha256"] == PRODUCTION_RULE_CONFIG_SHA256
        for row in read(path):
            counts["all_inputs"] += 1
            counts[row["processing_status"]] += 1
            if row["final_decision"] == "GOOD":
                counts["current_good"] += 1
                counts["current_selected"] += row["selected_final"] == "True"
                assert not row["manual_decision"]
                e = json.loads(row["rule_features"])["resolution_features"][
                    "interior_unresolved_envelope"
                ]
                if e["extremum_exception_applied"]:
                    candidates.append(("shots", row, args.rules_root))
    labels = read(REPO / "training_labels/tae_like_train.csv")
    tasks = candidates + [("training", label, args.training_root) for label in labels]
    print(
        f"Checking {len(candidates)} protected shot survivors and {len(labels)} training inputs.",
        flush=True,
    )
    measurements = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(measure, tasks, chunksize=8):
            measurements.append(result)
            if len(measurements) % 400 == 0:
                print(f"Measured {len(measurements)}/{len(tasks)}", flush=True)
    write(args.out_dir / "all_measurements.csv", measurements)
    training = [r for r in measurements if r["cohort"] == "training"]
    matrix = dict(
        Counter(r["training_label"] + ":" + r["baseline_decision"] for r in training)
    )
    assert matrix == {
        "good:GOOD": 543,
        "good:BAD": 32,
        "bad:GOOD": 25,
        "bad:BAD": 1763,
        "bad:ROUTED_EAE": 26,
        "bad:INVALID": 1,
    }, matrix
    sweep = []
    for floor in WIDTH_FLOORS:
        suffix = str(floor).replace(".", "_")
        affected = [r for r in measurements if r.get("newly_rejected_width_" + suffix)]
        write(args.out_dir / ("newly_rejected_width_" + suffix + ".csv"), affected)
        sweep.append(
            dict(
                clearance_floor=CLEARANCE_FLOOR,
                width_floor_grid=floor,
                strict_lower_cuts=True,
                affected_counts=dict(
                    Counter(
                        r["cohort"]
                        + (
                            ":" + r["training_label"]
                            if r["cohort"] == "training"
                            else ""
                        )
                        for r in affected
                    )
                ),
            )
        )
    assert all(sha256_file(REPO / p) == h for p, h in sources_hashes.items())
    assert all(sha256_file(Path(p)) == h for p, h in export_hashes.items())
    summary = dict(
        status="audit_only",
        shot_counts=dict(counts),
        protected_shot_survivors=len(candidates),
        training_rows=len(labels),
        training_baseline=matrix,
        sweep=sweep,
        source_sha256=sources_hashes,
        export_sha256=export_hashes,
        verified_shot_feature_records=len(candidates),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
