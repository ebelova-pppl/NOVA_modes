"""Screen training-shot continua against frequency-matched NOVA singularity logs.

Read-only diagnosis; no labels or exclusions are changed. Example (tcsh):
  python audits/n1_training_alignment_20260910/check_alignment.py \
    --data-root "$NOVA_DATA" --log-root "$NOVA_DITW_ROOT" \
    --out-dir outputs/review_n1_training_alignment_20260910 --ntor 1 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import numpy as np
from cont_features import CONTINUUM_PREPROCESSING_VERSION, continuum_crossing_records, load_datcon_for_mode
from nova_mode_loader import load_mode_from_nova
from tae_eae_features import classify_gap_region, upper2_scalars


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(path):
    """Pair one complete singularity block with its own hhh,om (omega squared).

    Empty or truncated blocks are retained explicitly. Never reuse radii from
    a preceding frequency. Logs contain NUL padding and Fortran exponents.
    """
    records = []
    block = None
    number = lambda value: float(value.replace("D", "E").replace("d", "e"))
    for line_number, line in enumerate(path.read_text(errors="replace").replace("\x00", "").splitlines(), 1):
        header = re.search(r"Singularities are expected at\s+(\d+)\s+points", line)
        if header:
            block = dict(expected=int(header[1]), radii=[], indices=[], header_line=line_number)
        point = re.search(r"ixmax vs ising\s+\d+\s+(\d+)\s+(\S+)", line)
        if point and block is not None:
            block["indices"].append(int(point[1]))
            block["radii"].append(number(point[2]))
        freq = re.search(r"hhh,om\s+(\S+)\s+(\S+)", line)
        if freq:
            if block is not None:
                records.append(dict(
                    omega2=number(freq[2]), radii=block["radii"], indices=block["indices"],
                    complete=len(block["radii"]) == block["expected"],
                    log=str(path), line=line_number,
                ))
            block = None
    return records


def match_log(records, omega2):
    matches = [record for record in records
               if abs(record["omega2"] - omega2) <= 1e-12 * abs(omega2)]
    if not matches:
        return "NO_FREQUENCY_MATCH", []
    if not all(record["complete"] for record in matches):
        return "INCOMPLETE_LOG_BLOCK", matches
    first = np.sort(matches[0]["radii"])
    if any(len(record["radii"]) != len(first)
           or not np.allclose(np.sort(record["radii"]), first, rtol=0, atol=1e-10)
           for record in matches[1:]):
        return "CONFLICTING_LOG_RECORDS", matches
    if not len(first):
        return "NO_LOGGED_SINGULARITIES", matches
    return "MATCHED", matches


def crossing_offsets(crossings, singularities, nr):
    rows = []
    for crossing in crossings:
        radius = crossing["r_cross"]
        nearest = min(singularities, key=lambda value: abs(value - radius))
        rows.append(dict(boundary=crossing["boundary"], crossing_r=radius,
                         singularity_r=nearest, offset_r=radius-nearest,
                         offset_grid=(radius-nearest)*(nr-1)))
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def measure_shot(task):
    shot, n, training_paths, args = task
    data = args.data_root / shot / f"N{n}"
    logs = args.log_root / shot / f"N{n}"
    sources = {}
    records = []
    for name in ("out_go", "out_go_prev"):
        path = logs / name
        if path.is_file():
            sources[str(path)] = sha(path)
            records.extend(read_log(path))
    datcon = data / f"datcon{n}"
    if datcon.is_file():
        sources[str(datcon)] = sha(datcon)
    modes, crossings, cache = [], [], {}
    for path in sorted(data.glob("egn*")):
        if not path.is_file():
            continue
        key = f"{shot}/N{n}/{path.name}"
        row = dict(shot=shot, ntor=n, path=key, in_training_list=key in training_paths,
                   status="", error="")
        try:
            sources[str(path)] = sha(path)
            reference = logs / path.name
            row["mode_matches_log_directory_copy"] = (
                sources[str(path)] == sha(reference) if reference.is_file() else "unavailable")
            mode, omega, gamma, ntor = load_mode_from_nova(str(path))
            nr = mode.shape[1]
            row.update(omega=omega, nr=nr)
            if ntor != n or not np.isfinite(omega) or omega <= 0 or not np.isfinite(mode).all():
                raise ValueError("Invalid mode header or nonfinite signed profiles")
            if nr not in cache:
                cache[nr] = load_datcon_for_mode(str(path), nr)[:2]
            low, high = cache[nr]
            scalars = upper2_scalars(mode, omega, high)
            row["gap_region"] = classify_gap_region(**scalars)
            all_crossings = continuum_crossing_records(mode, omega, low, high)
            core = [c for c in all_crossings if args.r_min <= c["r_cross"] < args.r_max]
            row.update(n_all_crossings=len(all_crossings), n_interior_crossings=len(core))
            status, matches = match_log(records, omega**2)
            row.update(status=status, log_match_count=len(matches))
            if matches:
                row.update(log=matches[0]["log"], log_line=matches[0]["line"],
                           logged_omega2=matches[0]["omega2"],
                           singularities=json.dumps(matches[0]["radii"]))
            if status == "MATCHED":
                rs = matches[0]["radii"]
                offsets = crossing_offsets(all_crossings, rs, nr)
                for cr in offsets:
                    cr.update(shot=shot, ntor=n, path=key,
                              in_training_list=key in training_paths,
                              gap_region=row["gap_region"], nr=nr,
                              in_interior=args.r_min <= cr["crossing_r"] < args.r_max)
                    crossings.append(cr)
                values = [cr["offset_grid"] for cr in offsets if cr["in_interior"]]
                row["median_interior_offset_grid"] = float(np.median(values)) if values else ""
                row["fraction_interior_abs_gt_2"] = float(np.mean(np.abs(values) > 2)) if values else ""
                row["log_grid_radius_max_error"] = float(max(
                    abs(r-(index-1)/(nr-1)) for r,index in zip(rs,matches[0]["indices"])))
        except (ValueError, OSError, IndexError, OverflowError) as exc:
            row.update(status="INPUT_ERROR", error=str(exc))
        modes.append(row)
    # Detect inputs changing during the scan, including externally recalculated files.
    if any(not Path(path).is_file() or sha(Path(path)) != digest for path,digest in sources.items()):
        raise RuntimeError(f"{shot}/N{n}: input changed during the audit; rerun")
    return modes, crossings, sources


def summaries(shots, ns, modes, crossings):
    rows = []
    for shot in shots:
        for n in ns:
            ms = [r for r in modes if r["shot"] == shot and r["ntor"] == n]
            for cohort in ("all", "tae_side", "training_list"):
                selected = [r for r in ms if cohort == "all" or
                            (cohort == "tae_side" and r.get("gap_region") in ("tae_like", "mixed")) or
                            (cohort == "training_list" and r["in_training_list"])]
                keys = {r["path"] for r in selected}
                cs = [r for r in crossings if r["path"] in keys and r["in_interior"]]
                values = np.array([r["offset_grid"] for r in cs])
                row = dict(shot=shot, ntor=n, cohort=cohort, n_modes=len(selected),
                           n_matched=sum(r["status"] == "MATCHED" for r in selected),
                           n_no_frequency_match=sum(r["status"] == "NO_FREQUENCY_MATCH" for r in selected),
                           n_incomplete_or_conflicting=sum(r["status"] in ("INCOMPLETE_LOG_BLOCK", "CONFLICTING_LOG_RECORDS") for r in selected),
                           n_input_errors=sum(r["status"] == "INPUT_ERROR" for r in selected),
                           n_modes_with_matched_interior_crossings=len({r["path"] for r in cs}),
                           n_interior_crossings=len(cs),
                           median_signed_grid=float(np.median(values)) if len(values) else "",
                           median_abs_grid=float(np.median(np.abs(values))) if len(values) else "",
                           fraction_abs_gt_2=float(np.mean(np.abs(values)>2)) if len(values) else "",
                           fraction_positive=float(np.mean(values>0)) if len(values) else "",
                           p90_abs_grid=float(np.percentile(np.abs(values),90)) if len(values) else "")
                rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Canonical training mode/continuum root")
    p.add_argument("--log-root", type=Path, required=True, help="Original NOVA shot root with out_go/out_go_prev")
    p.add_argument("--training-list", type=Path, default=REPO / "training_labels/tae_like_train.csv", help="Read only paths to select shots and record membership")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ntor", type=int, nargs="+", default=[1], help="Toroidal numbers; include 2 as a within-shot control")
    p.add_argument("--r-min", type=float, default=.03, help="Inclusive interior crossing radius")
    p.add_argument("--r-max", type=float, default=.75, help="Exclusive interior radius, omitting crowded/repaired edge")
    args = p.parse_args()
    if not 0 <= args.r_min < args.r_max <= 1 or any(n<1 for n in args.ntor):
        p.error("Require 0 <= r-min < r-max <= 1 and positive ntor")
    with args.training_list.open() as handle:
        training_paths = {row["path"] for row in csv.DictReader(handle)}
    if any(Path(key).is_absolute() or ".." in Path(key).parts or len(Path(key).parts)!=3 for key in training_paths):
        p.error("Training paths must have relative shot/N/file form")
    shots = sorted({key.split("/")[0] for key in training_paths})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = {str(args.training_list):sha(args.training_list),str(Path(__file__)):sha(Path(__file__))}
    sources.update({str(REPO/path):sha(REPO/path) for path in
                    ["src/cont_features.py","src/nova_mode_loader.py","src/tae_eae_features.py"]})
    modes, crossings = [], []
    tasks = [(shot,n,training_paths,args) for shot in shots for n in args.ntor]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for task,(ms,cs,hashes) in zip(tasks,pool.map(measure_shot,tasks)):
            modes.extend(ms); crossings.extend(cs); sources.update(hashes)
            print(f"{task[0]}/N{task[1]}: {len(ms)} modes, {sum(r['status']=='MATCHED' for r in ms)} log matches",flush=True)
    write_csv(args.out_dir/"mode_coverage.csv",modes)
    write_csv(args.out_dir/"crossing_offsets.csv",crossings)
    write_csv(args.out_dir/"shot_summary.csv",summaries(shots,args.ntor,modes,crossings))
    metadata = dict(schema="continuum-log-alignment-v1",created_utc=datetime.now(timezone.utc).isoformat(),
                    data_root=str(args.data_root),log_root=str(args.log_root),ntor=args.ntor,
                    r_min=args.r_min,r_max=args.r_max,frequency_rtol=1e-12,
                    continuum_preprocessing_version=CONTINUUM_PREPROCESSING_VERSION,
                    source_sha256=sources)
    (args.out_dir/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
