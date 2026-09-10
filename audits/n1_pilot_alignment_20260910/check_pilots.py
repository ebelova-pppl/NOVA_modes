"""Apply the training audit's log-alignment diagnostic to the 27 pilot shots.

Example (tcsh):
  python audits/n1_pilot_alignment_20260910/check_pilots.py \
    --data-root "$NOVA_DITW_ROOT" \
    --out-dir outputs/review_n1_pilot_alignment_20260910
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location(
    "training_alignment", HERE.parent / "n1_training_alignment_20260910/check_alignment.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def scan_group(task):
    """A live recalculation must not discard the other completed shot scans."""
    try:
        modes, crossings, sources = audit.measure_shot(task)
        result = dict(modes=modes, crossings=crossings, sources=sources, error="")
    except RuntimeError as exc:
        if "input changed during the audit" not in str(exc):
            raise
        result = dict(modes=[], crossings=[], sources={}, error=str(exc))
    path = task[3].out_dir / "groups" / f"{task[0]}_N{task[1]}.json"
    path.write_text(json.dumps(result)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path, help="Live shot root containing modes, datcon and NOVA logs")
    p.add_argument("--out-dir", required=True, type=Path, help="Local diagnostic output directory")
    args = p.parse_args()
    if not args.data_root.is_dir():
        p.error("data-root must be an existing shot root")
    args.log_root = args.data_root
    args.r_min, args.r_max = .03, .75
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"groups").mkdir(exist_ok=True)
    manifest = HERE.parent / "continuum_noise_20260910/adopted_shot_summary.csv"
    with manifest.open() as handle:
        shots = [row["shot"] for row in audit.csv.DictReader(handle)]
    if len(shots) != len(set(shots)) or any(Path(s).name != s or s in (".", "..") for s in shots):
        raise ValueError("Shot manifest must contain unique exact directory basenames")
    source_paths = [Path(__file__), Path(SPEC.origin), manifest,
                    REPO/"src/cont_features.py", REPO/"src/nova_mode_loader.py",
                    REPO/"src/tae_eae_features.py", REPO/"configs/known_invalid_inputs.csv"]
    sources = {str(path): audit.sha(path) for path in source_paths}
    tasks = [(shot, n, set(), args) for shot in shots for n in (1, 2)]
    modes, crossings, group_errors = [], [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for task, result in zip(tasks, pool.map(scan_group, tasks)):
            ms, cs, hashes = result["modes"], result["crossings"], result["sources"]
            if result["error"]:
                group_errors[f"{task[0]}/N{task[1]}"] = result["error"]
            modes.extend(ms)
            crossings.extend(cs)
            sources.update(hashes)
            print(f"{task[0]}/N{task[1]}: {len(ms)} modes, "
                  f"{sum(r['status'] == 'MATCHED' for r in ms)} nonempty log matches", flush=True)
    audit.write_csv(args.out_dir/"mode_coverage.csv", modes)
    audit.write_csv(args.out_dir/"crossing_offsets.csv", crossings)
    summary = [row for row in audit.summaries(shots, [1, 2], modes, crossings)
               if row["cohort"] != "training_list"]
    for row in summary:
        row["group_error"] = group_errors.get(f"{row['shot']}/N{row['ntor']}", "")
    audit.write_csv(args.out_dir/"shot_summary.csv", summary)
    metadata = dict(schema="continuum-log-alignment-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    data_root=str(args.data_root), log_root=str(args.log_root), ntor=[1, 2],
                    r_min=args.r_min, r_max=args.r_max, frequency_rtol=1e-12,
                    continuum_preprocessing_version=audit.CONTINUUM_PREPROCESSING_VERSION,
                    shot_count=len(shots), group_errors=group_errors, source_sha256=sources)
    (args.out_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
