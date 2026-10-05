"""Compare the October 5 recheck with September 30 without changing labels.

Run after check_database.py has completed the selection.csv inventory:
  python audits/n1_recheck_20261005/summarize.py
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO / "outputs/review_recalculated_input_check_20260930"
NEW = REPO / "outputs/review_n1_recheck_20261005"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shots = [r["shot"] for r in read(HERE / "selection.csv")]
    assert len(shots) == len(set(shots)) == 22
    rows, changes, source_hashes = [], [], {}
    metadata = json.loads((NEW / "metadata.json").read_text())
    assert metadata["shot_count"] == 22 and not metadata["group_errors"]
    for shot in shots:
        for n in (1, 2):
            snapshots = []
            for root in (OLD, NEW):
                path = root / "groups" / f"{shot}_N{n}.json"
                source_hashes[str(path.relative_to(REPO))] = sha(path)
                group = json.loads(path.read_text())
                assert not group["error"], (shot, n, group["error"])
                snapshots.append(group)
            old, new = snapshots
            src_old = {Path(k).name: v for k, v in old["sources"].items()}
            src_new = {Path(k).name: v for k, v in new["sources"].items()}
            old_names, new_names = set(old["mode_names"]), set(new["mode_names"])
            common = old_names & new_names
            row = dict(shot=shot, n=n, previous_modes=len(old_names), current_modes=len(new_names),
                       added=len(new_names-old_names), removed=len(old_names-new_names),
                       changed_modes=sum(src_old[k] != src_new[k] for k in common),
                       identical_modes=sum(src_old[k] == src_new[k] for k in common),
                       datcon_changed=src_old.get(f"datcon{n}") != src_new.get(f"datcon{n}"),
                       logs_changed=any(src_old.get(k) != src_new.get(k) for k in ("out_go", "out_go_prev")),
                       all_sources_identical=old["sources"] == new["sources"])
            for epoch, group in (("previous", old), ("current", new)):
                for cohort in ("tae_side", "all"):
                    modes = [r for r in group["modes"] if cohort == "all" or r.get("gap_region") in ("tae_like", "mixed")]
                    paths = {r["path"] for r in modes}
                    crossings = [r for r in group["crossings"] if r["path"] in paths and r["in_interior"]]
                    prefix = f"{epoch}_{cohort}_"
                    values = sorted(abs(r["offset_grid"]) for r in crossings)
                    row[prefix + "modes"] = len(modes)
                    row[prefix + "informative_modes"] = len({r["path"] for r in crossings})
                    row[prefix + "crossings"] = len(values)
                    row[prefix + "beyond_2"] = sum(v > 2 for v in values)
                    row[prefix + "median_abs_grid"] = (values[(len(values)-1)//2] + values[len(values)//2])/2 if values else ""
                    row[prefix + "max_abs_grid"] = max(values) if values else ""
                    row[prefix + "missing_logs"] = sum(r["status"] == "NO_FREQUENCY_MATCH" for r in modes)
                    row[prefix + "incomplete_conflicting_logs"] = sum(r["status"] in ("INCOMPLETE_LOG_BLOCK", "CONFLICTING_LOG_RECORDS") for r in modes)
                    row[prefix + "empty_logs_with_core_crossings"] = sum(r["status"] == "NO_LOGGED_SINGULARITIES" and r.get("n_interior_crossings", 0) > 0 for r in modes)
                    row[prefix + "input_errors"] = sum(r["status"] == "INPUT_ERROR" for r in modes)
            for name in sorted(src_old.keys() | src_new.keys()):
                changes.append(dict(shot=shot, n=n, name=name, previous_sha256=src_old.get(name, ""),
                                    current_sha256=src_new.get(name, ""), changed=src_old.get(name) != src_new.get(name)))
            rows.append(row)
    n1 = [r for r in rows if r["n"] == 1]
    write(HERE / "n1_before_after.csv", n1, list(n1[0]))
    write(HERE / "n2_control_before_after.csv", [r for r in rows if r["n"] == 2], list(rows[0]))
    write(NEW / "file_comparison.csv", changes, list(changes[0]))
    coverage = read(NEW / "mode_coverage.csv")
    offsets = read(NEW / "crossing_offsets.csv")
    bad = {r["path"] for r in offsets if r["ntor"] == "1" and r["in_interior"] == "True" and abs(float(r["offset_grid"])) > 2}
    review = [r for r in coverage if r["ntor"] == "1" and (r["path"] in bad or
              r["status"] == "NO_LOGGED_SINGULARITIES" and int(r.get("n_interior_crossings") or 0) > 0)]
    write(HERE / "review_modes.csv", review, list(coverage[0]))
    protected = json.loads((HERE / "protected_before.json").read_text())
    assert all(sha(REPO / p) == h for p, h in protected.items())
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), shot_count=22,
                   changed_n1_datcon=sum(r["datcon_changed"] for r in n1),
                   changed_n1_modes=sum(r["changed_modes"] for r in n1),
                   source_snapshots_sha256=source_hashes, protected_sha256=protected,
                   measurement_metadata_sha256=sha(NEW / "metadata.json"),
                   script_sha256=sha(Path(__file__)), review_modes=len(review))
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for r in n1:
        print(r["shot"], "datcon changed:", r["datcon_changed"],
              "TAE:", f'{r["previous_tae_side_beyond_2"]}/{r["previous_tae_side_crossings"]}',
              "->", f'{r["current_tae_side_beyond_2"]}/{r["current_tae_side_crossings"]}',
              "all:", f'{r["current_all_beyond_2"]}/{r["current_all_crossings"]}')


if __name__ == "__main__":
    main()
