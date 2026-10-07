"""Stage and install the 14 user-released shots with frozen production rules.

Example: python audits/released14_rules_20261005/run_batch.py stage \
  --data-root /path/to/DiTw --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released14_rules_20261005
Use the same paths with publish after staging verifies all shots.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "previous_batch", REPO / "audits/remaining122_rules_20260921/run_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def stage(args):
    import numpy as np
    from nova_mode_loader import load_mode_from_nova
    from cont_features import load_datcon_for_mode

    release = REPO / "audits/continuum_release_20261005"
    selected = batch.read(release / "released_shots.csv")
    shots = {r["shot"] for r in selected}
    assert len(selected) == len(shots) == 14
    inv = {r["shot"]: r for r in batch.read(batch.INVENTORY)}
    for shot in shots:
        assert inv[shot]["status"] == "ready_for_rules"
        assert inv[shot]["post_training_checked"] == inv[shot]["active_training_shot"] == "no"
        assert not (args.rules_root / shot).exists(), f"Existing output: {shot}"
    review = json.loads((release / "receipt.json").read_text())
    assert all(batch.sha256_file(Path(p)) == h for p, h in review["verified_live_sources_sha256"].items())
    assert not args.runtime_dir.exists()
    for name in ("rules", "logs", "inputs", "verified"):
        (args.runtime_dir / name).mkdir(parents=True)
    config = batch.load_rule_run_configuration(batch.CONFIG)
    protected = [batch.INVENTORY, batch.INVENTORY.with_name("g_shot_status.csv"),
                 REPO / "configs/known_invalid_inputs.csv", REPO / "training_labels/tae_like_train.csv",
                 REPO / "audits/processed40_20260915/accepted_tae_modes.csv",
                 REPO / "audits/cleared_nan_rules_20261005/good_tae_final_batch.csv",
                 release / "receipt.json", release / "released_shots.csv",
                 release / "remaining_continuum_holds.csv"]
    code = (set((REPO / "scripts").glob("*.py")) | set((REPO / "src").rglob("*.py"))
            | set((REPO / "configs/rules").glob("*.yaml")) | {Path(__file__), Path(batch.__file__)})
    inputs = dict(started_utc=datetime.now(timezone.utc).isoformat(), workflow="production",
                  data_root=str(args.data_root), rules_root=str(args.rules_root),
                  runtime_dir=str(args.runtime_dir), workers=args.workers,
                  configuration=batch.CONFIG, configuration_sha256=config.sha256,
                  source_sha256={str(p.relative_to(REPO)): batch.sha256_file(p) for p in sorted(code | set(protected))},
                  shots=sorted(shots), reviewed_sources_verified=len(review["verified_live_sources_sha256"]))
    batch.save(HERE / "run_inputs.json", inputs)
    preflight = []
    for row in selected:
        shot_dir = args.data_root / row["shot"]
        files = batch.files_for(shot_dir)
        checked_continua = set()
        for path in files:
            assert np.isfinite(np.fromfile(path, dtype=np.float64)).all(), f"Nonfinite raw input: {path}"
            mode, omega, gamma, ntor = load_mode_from_nova(path)
            assert mode.shape[1] == 201 and np.isfinite(mode).all(), f"Invalid mode/grid: {path}"
            assert ntor == int(path.parent.name[1:]) and np.isfinite(omega) and omega > 0
            assert np.isfinite(gamma), f"Nonfinite gamma_d: {path}"
            if ntor not in checked_continua:
                load_datcon_for_mode(str(path), mode.shape[1])
                checked_continua.add(ntor)
        row["current_mode_files"] = len(files)
        preflight.append(dict(shot=row["shot"], current_mode_files=len(files), nr=201,
                              nonfinite_files=0, n_groups=len(checked_continua)))
        print(f"Input check passed: {row['shot']}: {len(files)} modes", flush=True)
    batch.write(HERE / "selection.csv", selected)
    batch.save(HERE / "preflight.json", preflight)
    verified, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(batch.run_one, args, row, config.sha256): row["shot"]
                   for row in sorted(selected, key=lambda r: r["current_mode_files"], reverse=True)}
        for future in as_completed(pending):
            shot = pending[future]
            try:
                result = future.result()
                verified.append(result)
                print(f"{len(verified)}/14 verified: {shot}: {result['selected_good']} GOOD, "
                      f"{result['bad']} BAD ({result['elapsed_seconds']:.0f}s)", flush=True)
            except Exception as exc:
                failed.append(dict(shot=shot, error=f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {shot}: {exc}", flush=True)
            batch.save(HERE / "stage_results.json", dict(verified=verified, failed=failed))
    assert not failed and len(verified) == 14, failed
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    batch.write(HERE / "shot_summary.csv", [{k: v for k, v in r.items() if k != "output_sha256"}
                for r in sorted(verified, key=lambda r: r["shot"])])
    good = []
    for r in sorted(verified, key=lambda r: r["shot"]):
        for mode in batch.read(args.runtime_dir / "rules" / r["shot"] / "good_tae_final.csv"):
            compact = {k: mode.get(k, "") for k in batch.COMPACT}
            compact.update(path=mode["mode_key"], label="good", review_status="not_visually_reviewed")
            good.append(compact)
    batch.write(HERE / "good_tae_final_batch.csv", good, batch.COMPACT)
    counts = {k: sum(r[k] for r in verified) for k in
              ("input_modes", "tae_like", "mixed", "eae_like", "invalid", "good_before_dedup", "selected_good", "bad")}
    assert counts["selected_good"] == len(good)
    batch.save(HERE / "verification.json", dict(status="verified", shots=14, counts=counts,
               all_nr=201, all_raw_fingerprints_verified=True, configuration_sha256=config.sha256,
               artifacts_sha256={name: batch.sha256_file(HERE / name) for name in
                   ("selection.csv", "preflight.json", "shot_summary.csv", "good_tae_final_batch.csv", "stage_results.json")}))
    print(f"All 14 shots verified: {counts}", flush=True)


def publish(args):
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    verification = json.loads((HERE / "verification.json").read_text())
    results = json.loads((HERE / "stage_results.json").read_text())
    assert verification["status"] == "verified" and not results["failed"]
    assert not (HERE / "publication.json").exists()
    assert str(args.rules_root) == inputs["rules_root"] and str(args.runtime_dir) == inputs["runtime_dir"]
    assert str(args.data_root) == inputs["data_root"]
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    assert all(batch.sha256_file(HERE / p) == h for p, h in verification["artifacts_sha256"].items())
    assert {r["shot"] for r in results["verified"]} == set(inputs["shots"])
    for result in results["verified"]:
        shot = result["shot"]
        assert not (args.rules_root / shot).exists()
        assert not (args.rules_root / f".{shot}.installing_released14_20261005").exists()
        assert batch.tree(args.runtime_dir / "rules" / shot) == result["output_sha256"]
        before = json.loads((args.runtime_dir / "inputs" / f"{shot}.json").read_text())
        assert batch.fingerprints(args.data_root / shot) == before
    installed = []
    receipt = dict(status="installing", rules_root=str(args.rules_root), installed=installed,
                   existing_outputs_replaced=0, started_utc=datetime.now(timezone.utc).isoformat())
    batch.save(HERE / "publication.json", receipt)
    for result in sorted(results["verified"], key=lambda r: r["shot"]):
        shot = result["shot"]
        temporary = args.rules_root / f".{shot}.installing_released14_20261005"
        shutil.copytree(args.runtime_dir / "rules" / shot, temporary)
        assert batch.tree(temporary) == result["output_sha256"]
        temporary.rename(args.rules_root / shot)
        installed.append(shot)
        batch.save(HERE / "publication.json", receipt)
        print(f"Installed {shot}", flush=True)
    receipt.update(status="installed", finished_utc=datetime.now(timezone.utc).isoformat())
    batch.save(HERE / "publication.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "publish"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rules-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    for name in ("data_root", "rules_root", "runtime_dir"):
        setattr(args, name, getattr(args, name).resolve())
    globals()[args.phase](args)
