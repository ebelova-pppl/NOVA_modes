"""Make TAE-only review lists and separate mixed/potential EAE issue exports.

Example: python audits/continuum_small_review_20261005/make_lists.py
Counts use the saved snapshot. Exported mode/continuum/log sources are
verified against that snapshot before creating lists for the live viewer.
"""
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME = REPO / "outputs/review_n1_recheck_20261005"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from tae_rule_io import input_fingerprint
from mode_csv import read_mode_csv_entries


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    if path.exists() and any(r.get("manual_label") or r.get("manual_reason") for r in read(path)):
        raise RuntimeError(f"Refusing to overwrite manual review annotations: {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    remaining_path = REPO / "audits/cleared_nan_rules_20261005/remaining_unprocessed.csv"
    remaining = [r for r in read(remaining_path) if r["n1_all_beyond_2"]]
    assert len(remaining) == 22
    shots = {r["shot"] for r in remaining}
    metadata = json.loads((RUNTIME / "metadata.json").read_text())
    root = Path(metadata["data_root"])
    crossing_rows = read(RUNTIME / "crossing_offsets.csv")
    coverage_rows = read(RUNTIME / "mode_coverage.csv")
    coverage = {r["path"]: r for r in coverage_rows}
    problems = [r for r in crossing_rows if r["shot"] in shots and r["ntor"] in ("1", "2")
                and r["in_interior"] == "True" and abs(float(r["offset_grid"])) > 2]
    by_mode = defaultdict(list)
    for r in problems:
        by_mode[r["path"]].append(r)
    summaries = []
    gap_statuses = {"NO_FREQUENCY_MATCH", "INCOMPLETE_LOG_BLOCK", "CONFLICTING_LOG_RECORDS", "INPUT_ERROR"}
    for old in remaining:
        shot = old["shot"]
        cs = [r for r in problems if r["shot"] == shot]
        ts = [r for r in cs if r["gap_region"] in ("tae_like", "mixed")]
        ms = [r for r in coverage_rows if r["shot"] == shot]
        row = dict(shot=shot, n1_problem_crossings=sum(r["ntor"] == "1" for r in cs),
                   n2_problem_crossings=sum(r["ntor"] == "2" for r in cs),
                   problem_crossings=len(cs), affected_modes=len({r["path"] for r in cs}),
                   tae_side_problem_crossings=len(ts), tae_side_affected_modes=len({r["path"] for r in ts}),
                   coverage_gap_modes=sum(r["status"] in gap_statuses for r in ms),
                   empty_log_modes_with_core_crossings=sum(r["status"] == "NO_LOGGED_SINGULARITIES"
                       and int(r.get("n_interior_crossings") or 0) > 0 for r in ms))
        assert row["n1_problem_crossings"] == int(old["n1_all_beyond_2"])
        assert row["n2_problem_crossings"] == int(old["n2_all_beyond_2"])
        assert len(ts) == int(old["n1_tae_beyond_2"]) + int(old["n2_tae_beyond_2"])
        summaries.append(row)
    # Keep the original all-frequency and TAE+mixed exports for provenance.
    # The user's revised review scope is strictly TAE-like; EAE is provisional.
    scoped = []
    for old in summaries:
        shot = old["shot"]
        row = {"shot": shot}
        for region in ("tae_like", "mixed", "eae_like"):
            cs = [r for r in problems if r["shot"] == shot and r["gap_region"] == region]
            ms = [r for r in coverage_rows if r["shot"] == shot and r["gap_region"] == region]
            row[f"{region}_flagged_comparisons"] = len(cs)
            row[f"{region}_affected_modes"] = len({r["path"] for r in cs})
            for n in ("1", "2"):
                row[f"{region}_n{n}_flagged_comparisons"] = sum(r["ntor"] == n for r in cs)
            row[f"{region}_coverage_gap_modes"] = sum(r["status"] in gap_statuses for r in ms)
            row[f"{region}_empty_log_modes_with_interior_crossings"] = sum(
                r["status"] == "NO_LOGGED_SINGULARITIES" and int(r["n_interior_crossings"] or 0) > 0
                for r in ms)
        row["tae_review_status"] = ("TAE_CROSSING_REVIEW" if row["tae_like_flagged_comparisons"]
                                    else "NO_FLAGGED_TAE_COMPARISONS")
        row["mixed_review_status"] = ("MIXED_CROSSING_REVIEW" if row["mixed_flagged_comparisons"] else "")
        row["eae_review_status"] = ("POTENTIAL_EAE_CROSSING_ISSUE" if row["eae_like_flagged_comparisons"]
                                    or row["eae_like_empty_log_modes_with_interior_crossings"] else "")
        row["eae_coverage_status"] = "INCOMPLETE_LOG_EVIDENCE" if row["eae_like_coverage_gap_modes"] else ""
        scoped.append(row)
    scoped.sort(key=lambda r: (r["tae_like_flagged_comparisons"], r["shot"]))
    strict_small = [r for r in scoped if r["tae_like_flagged_comparisons"] < 10]
    summaries.sort(key=lambda r: (r["problem_crossings"], r["shot"]))
    small = [r for r in summaries if r["problem_crossings"] < 10]
    tae_small = sorted((r for r in summaries if r["tae_side_problem_crossings"] < 10),
                       key=lambda r: (r["tae_side_problem_crossings"], r["shot"]))
    write(HERE / "all_22_summary.csv", summaries)
    write(HERE / "small_count_shots.csv", small)
    write(HERE / "tae_side_small_count_shots.csv", tae_small)
    write(HERE / "tae_only_all_22_summary.csv", scoped)
    write(HERE / "tae_only_small_count_shots.csv", strict_small)
    verified = {}

    def export(filename, selected, tae_only=False, gaps=False, region=None, include_empty_log=False):
        order = {r["shot"]: i for i, r in enumerate(selected)}
        rows = []
        for key, r in coverage.items():
            if r["shot"] not in order or (tae_only and r.get("gap_region") not in ("tae_like", "mixed")):
                continue
            if region is not None and r.get("gap_region") != region:
                continue
            cs = by_mode[key]
            gap = r["status"] in gap_statuses or (include_empty_log and
                r["status"] == "NO_LOGGED_SINGULARITIES" and int(r["n_interior_crossings"] or 0) > 0)
            if not (gap if gaps else bool(cs)):
                continue
            mode_path = root / key
            dc = mode_path.parent / f"datcon{r['ntor']}"
            for p in (mode_path, dc, mode_path.parent / "out_go", mode_path.parent / "out_go_prev"):
                if str(p) not in metadata["source_sha256"]:
                    assert not p.exists(), f"New source requires recheck: {p}"
                    continue
                if str(p) not in verified:
                    h = sha(p)
                    assert h == metadata["source_sha256"][str(p)], f"Source changed: {p}"
                    verified[str(p)] = h
            rows.append(dict(path=key, manual_label="", manual_reason="", shot=r["shot"], n=r["ntor"],
                omega=r["omega"], nr=r["nr"], gap_region=r["gap_region"], problem_crossings=len(cs),
                max_abs_offset_grid=max((abs(float(c["offset_grid"])) for c in cs), default=""),
                crossing_details=json.dumps([{k: c[k] for k in ("boundary", "crossing_r", "singularity_r", "offset_grid")} for c in cs]),
                audit_status=r["status"], n_interior_crossings=r["n_interior_crossings"],
                input_fingerprint=input_fingerprint(mode_path, dc),
                review_reason=("Incomplete or unavailable log evidence; not a measured offset" if gaps else
                    "Potential EAE crossing issue only: upper EAE boundary is unavailable; nearest-log correspondence is ambiguous"
                    if region == "eae_like" else "Interior crossing-to-log distance >2 grid intervals")))
        rows.sort(key=lambda r: (order[r["shot"]], int(r["n"]), float(r["omega"]), r["path"]))
        write(HERE / filename, rows)
        parsed = read_mode_csv_entries(str(HERE / filename), data_root=root)
        assert len(parsed) == len(rows)
        assert all(Path(p).is_file() and not label for p, label in parsed)
        return rows

    primary = export("small_count_modes.csv", small)
    secondary = export("tae_side_small_count_modes.csv", tae_small, tae_only=True)
    gaps = export("small_count_coverage_gaps.csv", small, gaps=True)
    strict_modes = export("tae_only_small_count_modes.csv", strict_small, region="tae_like")
    mixed_modes = export("mixed_crossing_modes.csv", scoped, region="mixed")
    eae_modes = export("potential_eae_crossing_modes.csv", scoped, region="eae_like")
    eae_gaps = export("eae_log_coverage_gaps.csv", scoped, region="eae_like", gaps=True, include_empty_log=True)
    strict_shots = {r["shot"] for r in strict_small}
    write(HERE / "tae_only_small_count_crossings.csv", [r for r in problems
        if r["shot"] in strict_shots and r["gap_region"] == "tae_like"])
    selected_shots = {r["shot"] for r in small}
    write(HERE / "small_count_crossings.csv", [r for r in problems if r["shot"] in selected_shots])
    assert len(small) == 5 and len(primary) == 13 and sum(r["problem_crossings"] for r in small) == 17
    assert len(tae_small) == 13 and len(secondary) == 36 and sum(r["tae_side_problem_crossings"] for r in tae_small) == 49
    assert len(gaps) == 9 and all(r["gap_region"] == "eae_like" and int(r["n_interior_crossings"]) == 0 for r in gaps)
    assert len(strict_small) == 14 and len(strict_modes) == 39
    assert sum(r["problem_crossings"] for r in strict_modes) == 49
    assert len(mixed_modes) == 47 and len(eae_modes) == 234 and len(eae_gaps) == 71
    source_files = [remaining_path, RUNTIME / "metadata.json", RUNTIME / "crossing_offsets.csv", RUNTIME / "mode_coverage.csv", Path(__file__)]
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), measurement_snapshot=str(RUNTIME.relative_to(REPO)),
                   ntor=[1, 2], radius_min=.03, radius_max_exclusive=.75, offset_grid_strictly_greater_than=2,
                   problem_crossing_count_strictly_less_than=10, primary_shots=14, primary_modes=39, primary_crossings=49,
                   historical_all_frequency_shots=5, historical_all_frequency_modes=13, historical_all_frequency_crossings=17,
                   historical_tae_side_shots=13, historical_tae_side_modes=36, historical_tae_side_crossings=49,
                   historical_all_frequency_coverage_gap_modes=9,
                   current_primary_file="tae_only_small_count_modes.csv", current_scope="tae_like",
                   tae_only_small_shots=len(strict_small), tae_only_small_nonzero_shots=len({r["shot"] for r in strict_modes}),
                   tae_only_small_modes=len(strict_modes), tae_only_small_crossings=49,
                   mixed_modes=len(mixed_modes), potential_eae_modes=len(eae_modes), eae_log_coverage_gap_modes=len(eae_gaps),
                   source_sha256={str(p.relative_to(REPO)): sha(p) for p in source_files},
                   live_export_sources_sha256=verified,
                   files_sha256={p.name: sha(p) for p in HERE.glob("*.csv")},
                   decisions_changed=False, holds_changed=False, production_sorting_run=False)
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"tae_only_small_shots": len(strict_small), "tae_only_small_modes": len(strict_modes),
                      "tae_only_small_crossings": 49, "mixed_modes": len(mixed_modes),
                      "potential_eae_modes": len(eae_modes), "eae_log_coverage_gap_modes": len(eae_gaps)}, indent=2))


if __name__ == "__main__":
    main()
