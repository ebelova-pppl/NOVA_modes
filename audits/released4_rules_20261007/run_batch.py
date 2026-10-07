"""Stage and install the four released shots with frozen production rules.

Example: python audits/released4_rules_20261007/run_batch.py stage \
  --data-root /path/to/DiTw --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released4_rules_20261007
Use the same arguments with publish after successful staging.
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
    "batch", REPO / "audits/remaining122_rules_20260921/run_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
SHOTS = ("nstxuG121123N75", "nstxuG142301B85", "nstxuG142301F83", "nstxuG142301K79")
RELEASES = (REPO / "audits/continuum_release_n75_b85_20261007",
            REPO / "audits/continuum_release_f83_k79_20261007")


def stage(args):
    import numpy as np
    from nova_mode_loader import load_mode_from_nova
    from cont_features import load_datcon_for_mode

    inv = {r["shot"]: r for r in batch.read(batch.INVENTORY)}
    assert {r["shot"] for d in RELEASES for r in batch.read(d / "released_shots.csv")} == set(SHOTS)
    reviewed = [r for d in RELEASES for r in batch.read(d / "reviewed_modes.csv")]
    assert len(reviewed) == 44
    for row in reviewed:
        path = args.data_root / row["path"]
        assert batch.input_fingerprint(path, batch.datcon_path_for_mode(path)) == row["input_fingerprint"]
    for shot in SHOTS:
        assert inv[shot]["status"] == "ready_for_rules"
        assert inv[shot]["post_training_checked"] == inv[shot]["active_training_shot"] == "no"
        assert not (args.rules_root / shot).exists(), f"Existing output: {shot}"
    assert not args.runtime_dir.exists()
    for name in ("rules", "logs", "inputs", "verified"):
        (args.runtime_dir / name).mkdir(parents=True)
    config = batch.load_rule_run_configuration(batch.CONFIG)
    sources = (set((REPO / "scripts").glob("*.py")) | set((REPO / "src").rglob("*.py"))
               | set((REPO / "configs/rules").glob("*.yaml"))
               | {Path(__file__), Path(batch.__file__), batch.INVENTORY,
                  batch.INVENTORY.with_name("g_shot_status.csv"), REPO / "configs/known_invalid_inputs.csv",
                  REPO / "training_labels/tae_like_train.csv",
                  REPO / "audits/processed40_20260915/accepted_tae_modes.csv",
                  REPO / "audits/released14_manual_review_20261007/accepted_tae_modes.csv"})
    sources |= {d / name for d in RELEASES for name in ("released_shots.csv", "reviewed_modes.csv", "receipt.json")}
    inputs = dict(started_utc=datetime.now(timezone.utc).isoformat(), workflow="production",
                  data_root=str(args.data_root), rules_root=str(args.rules_root), runtime_dir=str(args.runtime_dir),
                  workers=args.workers, configuration=batch.CONFIG, configuration_sha256=config.sha256,
                  source_sha256={str(p.relative_to(REPO)): batch.sha256_file(p) for p in sorted(sources)},
                  shots=list(SHOTS), reviewed_mode_fingerprints_verified=len(reviewed))
    batch.save(HERE / "run_inputs.json", inputs)
    selected, preflight = [], []
    for shot in SHOTS:
        files = batch.files_for(args.data_root / shot)
        checked_n = set()
        for path in files:
            assert np.isfinite(np.fromfile(path, dtype=np.float64)).all(), f"Nonfinite raw input: {path}"
            mode, omega, gamma, n = load_mode_from_nova(path)
            assert mode.shape[1] == 201 and np.isfinite(mode).all(), f"Invalid mode/grid: {path}"
            assert n == int(path.parent.name[1:]) and np.isfinite(omega) and omega > 0 and np.isfinite(gamma)
            if n not in checked_n:
                load_datcon_for_mode(str(path), 201)
                checked_n.add(n)
        selected.append(dict(shot=shot, current_mode_files=len(files)))
        preflight.append(dict(shot=shot, mode_files=len(files), nr=201, nonfinite_files=0, n_groups=len(checked_n)))
        print(f"Input check passed: {shot}: {len(files)} modes", flush=True)
    batch.write(HERE / "selection.csv", selected)
    batch.save(HERE / "preflight.json", preflight)
    verified, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(batch.run_one, args, row, config.sha256): row["shot"] for row in selected}
        for future in as_completed(pending):
            shot = pending[future]
            try:
                result = future.result()
                verified.append(result)
                print(f"Verified {shot}: {result['selected_good']} GOOD, {result['bad']} BAD", flush=True)
            except Exception as exc:
                failed.append(dict(shot=shot, error=f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {shot}: {exc}", flush=True)
            batch.save(HERE / "stage_results.json", dict(verified=verified, failed=failed))
    assert not failed and len(verified) == 4, failed
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    batch.write(HERE / "shot_summary.csv", [{k: v for k, v in r.items() if k != "output_sha256"}
                for r in sorted(verified, key=lambda r: r["shot"])])
    good, review_results = [], []
    reviewed_by_key = {r["path"]: r for r in reviewed}
    for shot in SHOTS:
        output = args.runtime_dir / "rules" / shot
        assert not batch.read(output / "bae_like.csv")
        for mode in batch.read(output / "good_tae_final.csv"):
            row = {k: mode.get(k, "") for k in batch.COMPACT}
            row.update(path=mode["mode_key"], label="good", review_status="not_visually_reviewed")
            good.append(row)
        for mode in batch.read(output / "all_modes_rules.csv"):
            if mode["mode_key"] in reviewed_by_key:
                prior = reviewed_by_key[mode["mode_key"]]
                assert mode["input_fingerprint"] == prior["input_fingerprint"]
                row = {k: mode.get(k, "") for k in batch.COMPACT}
                row.update(path=mode["mode_key"], label="", review_status=prior["assessment"])
                review_results.append(row)
    assert len(review_results) == 44
    batch.write(HERE / "good_tae_final_batch.csv", good, batch.COMPACT)
    batch.write(HERE / "reviewed_tae_mode_results.csv", review_results, batch.COMPACT)
    counts = {k: sum(r[k] for r in verified) for k in
              ("input_modes", "tae_like", "mixed", "eae_like", "invalid", "good_before_dedup", "selected_good", "bad")}
    assert counts["selected_good"] == len(good)
    batch.save(HERE / "verification.json", dict(status="verified", shots=4, counts=counts, all_nr=201,
        all_raw_fingerprints_verified=True, configuration_sha256=config.sha256,
        artifacts_sha256={p.name: batch.sha256_file(p) for p in HERE.glob("*.csv")}))
    print(f"All four shots verified: {counts}", flush=True)


def publish(args):
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    verification = json.loads((HERE / "verification.json").read_text())
    results = json.loads((HERE / "stage_results.json").read_text())
    assert verification["status"] == "verified" and not results["failed"]
    assert not (HERE / "publication.json").exists()
    assert all(str(getattr(args, key)) == inputs[key] for key in ("data_root", "rules_root", "runtime_dir"))
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    assert all(batch.sha256_file(HERE / p) == h for p, h in verification["artifacts_sha256"].items())
    assert {r["shot"] for r in results["verified"]} == set(SHOTS)
    for result in results["verified"]:
        shot = result["shot"]
        assert not (args.rules_root / shot).exists()
        assert not (args.rules_root / f".{shot}.installing_released4_20261007").exists()
        assert batch.tree(args.runtime_dir / "rules" / shot) == result["output_sha256"]
        before = json.loads((args.runtime_dir / "inputs" / f"{shot}.json").read_text())
        assert batch.fingerprints(args.data_root / shot) == before
    installed = []
    receipt = dict(status="installing", rules_root=str(args.rules_root), installed=installed,
                   existing_outputs_replaced=0, started_utc=datetime.now(timezone.utc).isoformat())
    batch.save(HERE / "publication.json", receipt)
    for result in sorted(results["verified"], key=lambda r: r["shot"]):
        shot = result["shot"]
        temporary = args.rules_root / f".{shot}.installing_released4_20261007"
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
