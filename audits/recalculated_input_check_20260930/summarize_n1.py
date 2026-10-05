"""Compare fresh alignment snapshots with the September 10 source hashes.

python audits/recalculated_input_check_20260930/summarize_n1.py \
    --runtime-dir outputs/review_recalculated_input_check_20260930
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINE = REPO / "outputs/review_n1_database_alignment_20260910"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    args = parser.parse_args()
    selection = list(csv.DictReader((HERE / "selection.csv").open()))
    selected = [r["shot"] for r in selection if r["check_n1"] == "True"]
    summary = list(csv.DictReader((args.runtime_dir / "shot_summary.csv").open()))
    lookup = {(r["shot"], int(r["ntor"]), r["cohort"]): r for r in summary}
    old_summary = list(csv.DictReader((BASELINE / "shot_summary.csv").open()))
    old_lookup = {(r["shot"], int(r["ntor"]), r["cohort"]): r for r in old_summary}
    rows, evidence, hashes = [], [], {}
    for shot in selected:
        for n in (1, 2):
            oldfile = BASELINE / "groups" / f"{shot}_N{n}.json"
            newfile = args.runtime_dir / "groups" / f"{shot}_N{n}.json"
            old, new = json.loads(oldfile.read_text()), json.loads(newfile.read_text())
            assert not old["error"] and not new["error"], (shot, n)
            hashes[str(oldfile.relative_to(REPO))] = sha(oldfile)
            hashes[str(newfile)] = sha(newfile)
            previous = {Path(k).name: v for k, v in old["sources"].items()}
            current = {Path(k).name: v for k, v in new["sources"].items()}
            oldm = {k: v for k, v in previous.items() if k.startswith("egn")}
            newm = {k: v for k, v in current.items() if k.startswith("egn")}
            common = oldm.keys() & newm.keys()
            for name in sorted(previous.keys() | current.keys()):
                evidence.append(dict(shot=shot, n=n, name=name,
                                     old_sha256=previous.get(name, ""), new_sha256=current.get(name, ""),
                                     status=("added" if name not in previous else "removed" if name not in current
                                             else "identical" if previous[name] == current[name] else "changed")))
            row = dict(shot=shot, n=n, previous_modes=len(oldm), current_modes=len(newm),
                       added=len(newm.keys() - oldm.keys()), removed=len(oldm.keys() - newm.keys()),
                       changed=sum(oldm[k] != newm[k] for k in common),
                       identical=sum(oldm[k] == newm[k] for k in common),
                       datcon_changed=previous.get(f"datcon{n}") != current.get(f"datcon{n}"),
                       all_alignment_sources_identical=old["sources"] == new["sources"])
            for cohort in ("tae_side", "all"):
                for epoch, table in (("previous", old_lookup), ("current", lookup)):
                    s = table[(shot, n, cohort)]
                    for key in ("n_modes_with_matched_interior_crossings", "n_interior_crossings",
                                "fraction_abs_gt_2", "median_abs_grid", "n_no_frequency_match",
                                "n_incomplete_or_conflicting", "n_input_errors"):
                        row[f"{epoch}_{cohort}_{key}"] = s[key]
            rows.append(row)
    for name, content in (("n1_before_after.csv", [r for r in rows if r["n"] == 1]),
                          ("n2_control_before_after.csv", [r for r in rows if r["n"] == 2])):
        with (HERE / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=content[0]); w.writeheader(); w.writerows(content)
    with (args.runtime_dir / "file_changes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=evidence[0]); w.writeheader(); w.writerows(evidence)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   baseline=str(BASELINE.relative_to(REPO)), source_sha256=hashes,
                   script_sha256=sha(Path(__file__)))
    (HERE / "n1_comparison_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for r in rows:
        if r["n"] == 1:
            print(r["shot"], "identical inputs:", r["all_alignment_sources_identical"],
                  "current bad fraction:", r["current_tae_side_fraction_abs_gt_2"])


if __name__ == "__main__":
    main()
