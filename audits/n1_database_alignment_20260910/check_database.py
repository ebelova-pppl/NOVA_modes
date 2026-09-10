"""Screen the remaining DiTw shots without running production sorting.

Example (tcsh; reruns reuse only fingerprint-verified completed groups):
  python audits/n1_database_alignment_20260910/check_database.py \
    --data-root "$NOVA_DITW_ROOT" \
    --out-dir outputs/review_n1_database_alignment_20260910
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location(
    "training_alignment", HERE.parent/"n1_training_alignment_20260910/check_alignment.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def inventory_rows(path):
    with path.open() as handle:
        rows = list(audit.csv.DictReader(handle))
    names = [row["shot"] for row in rows]
    if len(names) != len(set(names)) or any(
        Path(name).name != name or name in (".", "..") for name in names
    ):
        raise ValueError("Inventory requires unique exact shot directory basenames")
    return rows


def mode_names(directory):
    return sorted(path.name for path in directory.glob("egn*") if path.is_file())


def source_files():
    return [Path(__file__), Path(SPEC.origin), REPO/"src/cont_features.py",
            REPO/"src/nova_mode_loader.py", REPO/"src/tae_eae_features.py"]


def scan_group(task):
    shot, n, args, code_hashes = task
    directory = args.data_root/shot/f"N{n}"
    cache = args.out_dir/"groups"/f"{shot}_N{n}.json"
    names = mode_names(directory)
    if cache.is_file():
        old = json.loads(cache.read_text())
        # Missing/changed logs must also invalidate an empty-directory snapshot.
        auxiliary = [directory/name for name in (f"datcon{n}", "out_go", "out_go_prev")]
        auxiliary_paths = {str(path) for path in auxiliary if path.is_file()}
        old_auxiliary = {path for path in old.get("sources", {}) if Path(path).name in
                         (f"datcon{n}", "out_go", "out_go_prev")}
        if (not old["error"] and old["code_hashes"] == code_hashes
                and old["mode_names"] == names and auxiliary_paths == old_auxiliary
                and all(Path(path).is_file() and audit.sha(Path(path)) == digest
                        for path, digest in old["sources"].items())):
            return old, True
    started = datetime.now(timezone.utc).isoformat()
    try:
        if not (args.data_root/shot).is_dir():
            raise FileNotFoundError(f"Missing shot directory: {shot}")
        modes, crossings, sources = audit.measure_shot((shot, n, set(), args))
        if names != mode_names(directory):
            raise RuntimeError("Mode inventory changed during scan")
        result = dict(modes=modes, crossings=crossings, sources=sources, error="")
    except (RuntimeError, OSError, ValueError, ZeroDivisionError, IndexError) as exc:
        # A failed group is inconclusive, never an aligned zero-crossing result.
        result = dict(modes=[], crossings=[], sources={}, error=f"{type(exc).__name__}: {exc}")
    result.update(shot=shot, ntor=n, mode_names=names, code_hashes=code_hashes,
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat())
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps(result)+"\n")
    temporary.replace(cache)
    return result, False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path, help="Live DiTw shot root; read only")
    p.add_argument("--out-dir", required=True, type=Path, help="Local diagnostic output directory")
    p.add_argument("--inventory", type=Path, default=REPO/"audits/main_dataset_shots/shot_status.csv",
                   help="Shot inventory; scan entries outside the training/pilot cohorts")
    args = p.parse_args()
    if not args.data_root.is_dir():
        p.error("data-root must exist")
    args.log_root = args.data_root
    args.r_min, args.r_max = .03, .75
    rows = inventory_rows(args.inventory)
    selected = [row for row in rows if row["active_training_shot"] != "yes"
                and row["post_training_checked"] != "yes"]
    shots = [row["shot"] for row in selected]
    (args.out_dir/"groups").mkdir(parents=True, exist_ok=True)
    audit.write_csv(args.out_dir/"selected_shots.csv", selected)
    code_hashes = {str(path): audit.sha(path) for path in source_files()}
    input_hashes = {str(args.inventory): audit.sha(args.inventory),
                    str(REPO/"configs/known_invalid_inputs.csv"): audit.sha(REPO/"configs/known_invalid_inputs.csv")}
    results, reused = {}, 0
    tasks = [(shot, n, args, code_hashes) for shot in shots for n in (1, 2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(scan_group, task): task for task in tasks}
        for count, future in enumerate(as_completed(pending), 1):
            result, cached = future.result()
            reused += cached
            results[(result["shot"], result["ntor"])] = result
            print(f"{count}/{len(tasks)} {result['shot']}/N{result['ntor']}: "
                  f"{len(result['modes'])} modes; "
                  f"{result['error'] or ('verified cache' if cached else 'measured')}", flush=True)
    modes, crossings, sources, errors = [], [], dict(code_hashes), {}
    sources.update(input_hashes)
    for shot in shots:
        for n in (1, 2):
            result = results[(shot, n)]
            modes.extend(result["modes"])
            crossings.extend(result["crossings"])
            sources.update(result["sources"])
            if result["error"]:
                errors[f"{shot}/N{n}"] = result["error"]
    audit.write_csv(args.out_dir/"mode_coverage.csv", modes)
    audit.write_csv(args.out_dir/"crossing_offsets.csv", crossings)
    summary = [row for row in audit.summaries(shots, [1, 2], modes, crossings)
               if row["cohort"] != "training_list"]
    for row in summary:
        row["group_error"] = errors.get(f"{row['shot']}/N{row['ntor']}", "")
    audit.write_csv(args.out_dir/"shot_summary.csv", summary)
    if any(audit.sha(Path(path)) != digest for path, digest in {**code_hashes, **input_hashes}.items()):
        raise RuntimeError("Audit code or selection changed during scan")
    metadata = dict(schema="continuum-log-alignment-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    data_root=str(args.data_root), log_root=str(args.log_root), ntor=[1, 2],
                    r_min=args.r_min, r_max=args.r_max, frequency_rtol=1e-12,
                    continuum_preprocessing_version=audit.CONTINUUM_PREPROCESSING_VERSION,
                    shot_count=len(shots), group_errors=errors, cached_groups=reused,
                    source_sha256=sources)
    (args.out_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
