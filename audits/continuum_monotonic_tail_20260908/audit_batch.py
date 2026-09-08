"""Compare the isolated repair on the frozen 27-shot membership.

Example (from the repository root):
  python audits/continuum_monotonic_tail_20260908/audit_batch.py \
    --data-root /path/to/DiTw --pilot-output-root /path/to/sort_outputs \
    --regression-output-root outputs/regression15_v5_20260907 \
    --out-dir outputs/continuum_tail_monotonic_20260908/batch

Only changed continuum directories require raw-mode reevaluation. Unchanged
profiles retain saved decisions after applying the adopted v6 family routing.
This audit does not run RF ranking, CNN inference, or manual adjudication.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import numpy as np

from candidate import read_raw, repair
from cont_features import load_datcon_for_mode
from nova_mode_loader import load_mode_from_nova
from tae_eae_features import classify_gap_region, upper2_scalars
from tae_rule_config import load_rule_run_configuration
from tae_rule_engine import ContinuumCrossingConfig, evaluate_mode
from tae_rule_io import input_fingerprint, sha256_file, stable_json, write_dict_csv


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def saved_current(row):
    if row["processing_status"] == "INVALID":
        return {
            "gap_region": row["gap_region"],
            "decision": "INVALID",
            "reason": row["preprocessing_primary_reason"],
        }
    scalars = {
        key: float(row[key]) for key in ("signed_delta", "fraction_below_upper2")
    }
    route = classify_gap_region(**scalars)
    if route == "eae_like":
        decision, reason = "ROUTED_EAE", "frequency_routing"
    else:
        decision, reason = row["rule_decision"], row["rule_primary_reason"]
        assert decision in {"BAD", "REVIEW"}, row["mode_key"]
    return dict(gap_region=route, decision=decision, reason=reason, **scalars)


def evaluate(row, mode, low, high):
    scalars = upper2_scalars(mode, float(row["omega"]), high)
    route = classify_gap_region(**scalars)
    if route == "eae_like":
        return (
            dict(
                gap_region=route,
                decision="ROUTED_EAE",
                reason="frequency_routing",
                **scalars,
            ),
            None,
        )
    result = evaluate_mode(
        dict(row, gap_region=route),
        mode=mode,
        low2=low,
        high2=high,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
    )
    assert result.decision in {"BAD", "REVIEW"}, result.diagnostic_message
    return (
        dict(
            gap_region=route,
            decision=result.decision,
            reason=result.primary_reason,
            **scalars,
        ),
        result.features,
    )


def full_array(values, nr, first):
    out = np.full(nr, np.nan)
    out[first : first + len(values)] = values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "data-root",
        "pilot-output-root",
        "regression-output-root",
        "out-dir",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_rule_run_configuration("tae_rules_production_v6")
    sources = [
        Path(__file__).relative_to(REPO),
        Path(__file__).with_name("candidate.py").relative_to(REPO),
        Path("src/cont_features.py"),
        Path("src/tae_eae_features.py"),
        Path("src/nova_mode_loader.py"),
        Path("scripts/tae_rule_engine.py"),
        Path("configs/rules/tae_rules_production_v6.yaml"),
    ]
    source_hashes = {str(p): sha256_file(REPO / p) for p in sources}
    grouped = defaultdict(list)
    all_rows, saved_hashes, membership_hashes = [], {}, {}
    for membership, root in [
        (REPO / "audits/regression15_v5/shot_summary.csv", args.regression_output_root),
        (REPO / "audits/pilot12_v5_20260908/shot_summary.csv", args.pilot_output_root),
    ]:
        membership_hashes[str(membership.relative_to(REPO))] = sha256_file(membership)
        for shot in read_csv(membership):
            for n in range(1, 11):
                grouped[(shot["shot"], n)]
            table = root / shot["shot"] / "all_modes_rules.csv"
            rows = read_csv(table)
            saved_hashes[shot["shot"]] = sha256_file(table)
            assert len(rows) == int(shot["input_modes"])
            for row in rows:
                assert not row[
                    "manual_decision"
                ], "Manual overrides require separate comparison"
                all_rows.append(row)
                grouped[(row["shot"], int(row["n"]))].append(row)
    assert len({r["mode_key"] for r in all_rows}) == len(all_rows)
    assert len({r["shot"] for r in all_rows}) == 27
    assert len(grouped) == 270
    baseline = {r["mode_key"]: saved_current(r) for r in all_rows}
    treatments = {fill: dict(baseline) for fill in ("last", "mean")}
    profiles, changed, checked = [], [], Counter()
    started = time.monotonic()
    details = (args.out_dir / "mode_details.jsonl").open("w")
    for (shot, n), rows in sorted(grouped.items()):
        directory = args.data_root / shot / f"N{n}"
        datcon = directory / f"datcon{n}"
        expected = {Path(r["mode_key"]).name for r in rows}
        assert expected == {
            p.name for p in directory.glob("egn*") if p.is_file()
        }, directory
        valid_rows = [r for r in rows if r["processing_status"] != "INVALID"]
        assert not valid_rows or {int(r["nr"]) for r in valid_rows} == {201}, directory
        raw_low, raw_high, radius = read_raw(datcon, nr=201)
        current_low, current_high, i1, i2 = load_datcon_for_mode(
            str(directory / next(iter(expected), f"egn{n:02d}w.audit")), n_r=201
        )
        arrays = {}
        profile = dict(
            shot=shot,
            n=n,
            nr=201,
            input_modes=len(rows),
            datcon_sha256=sha256_file(datcon),
        )
        for fill in treatments:
            low, high, j = repair(raw_low, raw_high, radius, fill=fill)
            low, high = full_array(low, 201, i1 - 1), full_array(high, 201, i1 - 1)
            arrays[fill] = (low, high)
            differs = not (
                np.array_equal(low, current_low, equal_nan=True)
                and np.array_equal(high, current_high, equal_nan=True)
            )
            profile["changed_" + fill] = differs
            profile["onset"] = None if j is None else float(radius[j])
        profiles.append(profile)
        if not any(profile["changed_" + f] for f in treatments):
            continue
        for row in rows:
            path = directory / Path(row["mode_key"]).name
            assert row["processing_status"] != "INVALID", path
            assert input_fingerprint(path, datcon) == row["input_fingerprint"], path
            mode, omega, gamma, ntor = load_mode_from_nova(path)
            assert mode.shape == (int(row["nhar"]), int(row["nr"])), path
            assert (omega, gamma, ntor) == (
                float(row["omega"]),
                float(row["gamma_d"]),
                int(row["ntor"]),
            ), path
            current, features = evaluate(row, mode, current_low, current_high)
            assert current == baseline[row["mode_key"]], (
                path,
                current,
                baseline[row["mode_key"]],
            )
            if features is not None:
                assert json.loads(stable_json(features)) == json.loads(
                    row["rule_features"]
                ), path
                checked["exact_baseline_feature_matches"] += 1
            detail = dict(
                mode_key=row["mode_key"],
                input_fingerprint=row["input_fingerprint"],
                current=current,
                current_features=features,
                onset=profile["onset"],
            )
            for fill, (low, high) in arrays.items():
                candidate, candidate_features = evaluate(row, mode, low, high)
                treatments[fill][row["mode_key"]] = candidate
                detail[fill], detail[fill + "_features"] = candidate, candidate_features
                if candidate != current:
                    checked[fill + "_scalar_or_decision_changes"] += 1
                if any(
                    candidate[k] != current[k]
                    for k in ("gap_region", "decision", "reason")
                ):
                    changed.append(
                        dict(
                            mode_key=row["mode_key"],
                            shot=shot,
                            n=n,
                            scenario=fill,
                            input_fingerprint=row["input_fingerprint"],
                            onset=profile["onset"],
                            **{"before_" + k: v for k, v in current.items()},
                            **{"after_" + k: v for k, v in candidate.items()},
                        )
                    )
            details.write(stable_json(detail) + "\n")
            checked["raw_modes_recomputed"] += 1
        details.flush()
        print(
            f"{shot} N{n}: onset={profile['onset']}; "
            f"{checked['raw_modes_recomputed']} modes checked; "
            f"{time.monotonic() - started:.0f}s",
            flush=True,
        )
    details.close()
    for p in sources:
        assert sha256_file(REPO / p) == source_hashes[str(p)]
    for profile in profiles:
        path = (
            args.data_root
            / profile["shot"]
            / f"N{profile['n']}"
            / f"datcon{profile['n']}"
        )
        assert sha256_file(path) == profile["datcon_sha256"]
    shots = []
    for shot in sorted({r["shot"] for r in all_rows}):
        keys = [r["mode_key"] for r in all_rows if r["shot"] == shot]
        record = {"shot": shot, "input_modes": len(keys)}
        for name, result in [("current_v6", baseline), *treatments.items()]:
            counts = Counter(result[k]["decision"] for k in keys)
            for status in ("REVIEW", "BAD", "ROUTED_EAE", "INVALID"):
                record[name + "_" + status] = counts[status]
        shots.append(record)
    summary = dict(
        status="experimental_not_adopted",
        configuration=config.name,
        configuration_sha256=config.sha256,
        shot_count=27,
        input_modes=len(all_rows),
        profiles_scanned=len(profiles),
        changed_profiles={
            f: sum(p["changed_" + f] for p in profiles) for f in treatments
        },
        baseline_counts=dict(Counter(r["decision"] for r in baseline.values())),
        candidate_counts={
            f: dict(Counter(r["decision"] for r in v.values()))
            for f, v in treatments.items()
        },
        transitions={
            f: dict(
                Counter(
                    r["before_decision"] + " -> " + r["after_decision"]
                    for r in changed
                    if r["scenario"] == f
                )
            )
            for f in treatments
        },
        checked=dict(checked),
        source_sha256=source_hashes,
        membership_sha256=membership_hashes,
        saved_output_sha256=saved_hashes,
        limitations=[
            "Unchanged continuum directories use saved mode results with v6 routing.",
            "97 known invalid inputs retained; no new validity adjudication.",
            "REVIEW survivors become automatic production GOOD before deduplication.",
            "RF ranking, CNN inference, and visual GOOD/BAD adjudication not run.",
        ],
    )
    write_dict_csv(args.out_dir / "profiles.csv", list(profiles[0]), profiles)
    write_dict_csv(args.out_dir / "changed_modes.csv", list(changed[0]), changed)
    write_dict_csv(args.out_dir / "shot_summary.csv", list(shots[0]), shots)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
