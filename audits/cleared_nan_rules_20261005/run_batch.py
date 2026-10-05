"""Stage and verify the five authorized NaN-cleared shots.

Example: python audits/cleared_nan_rules_20261005/run_batch.py \
  --data-root "$NOVA_DITW_ROOT" --workflow rules-cli \
  --runtime-dir outputs/review_cleared_nan_rules_20261005
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("previous_batch", REPO / "audits/remaining122_rules_20260921/run_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def check_calibration(output, before, shot_dir, config_hash):
    rows = batch.read(output / "all_modes_rules.csv")
    keyed = {r["mode_key"]: r for r in rows}
    assert len(keyed) == len(rows) == len(before)
    assert {k: r["input_fingerprint"] for k, r in keyed.items()} == before
    assert batch.fingerprints(shot_dir) == before
    s = dict(batch.csv.reader((output / "shot_summary.csv").open()))
    assert s["rule_configuration_sha256"] == config_hash
    assert s["continuum_preprocessing_version"] == "datcon-monotonic-tail-v1"
    assert s["n_invalid"] == s["n_final_good"] == s["n_final_good_before_clustering"] == "0"
    assert s["n_manual_override_rows"] == s["n_overrides_applied"] == "0"
    assert s["n_severity_unavailable"] == "0"
    assert not batch.read(output / "resolution_warnings.csv")
    for r in rows:
        assert r["nr"] == "201" and not r["diagnostic_message"]
        if r["processing_status"] == "ROUTED_EAE":
            assert r["gap_region"] == "eae_like"
        else:
            assert r["processing_status"] == "RULE_EVALUATED" and r["severity_complete"] == "True"
            assert r["rule_decision"] in ("BAD", "REVIEW")
            assert r["final_decision"] == r["rule_decision"]
    for filename, decision in [("bad_tae_like.csv", "BAD"), ("review_tae_like.csv", "REVIEW"),
                               ("good_tae_final.csv", "GOOD"), ("rejected_modes.csv", "INVALID")]:
        exported = batch.read(output / filename)
        expected = [r for r in rows if r["final_decision"] == decision]
        assert len(exported) == len(expected)
        assert {r["mode_key"] for r in exported} == {r["mode_key"] for r in expected}
        assert all(r == keyed[r["mode_key"]] for r in exported)
    assert int(s["n_total_files"]) == len(rows)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--runtime-dir", type=Path, required=True)
    p.add_argument("--workflow", choices=("rules-cli", "production"), required=True)
    args = p.parse_args()
    args.data_root = args.data_root.resolve()
    args.runtime_dir = args.runtime_dir.resolve()
    config = batch.load_rule_run_configuration(batch.CONFIG)
    selected = batch.read(HERE / "selection.csv")
    preflight = {r["shot"]: r for r in json.loads((HERE / "preflight.json").read_text())}
    assert len(selected) == len(preflight) == 5
    protected = json.loads((HERE / "protected_before.json").read_text())
    assert all(batch.sha256_file(REPO / path) == h for path, h in protected.items())
    inputs = dict(created_utc=datetime.now(timezone.utc).isoformat(), workflow=args.workflow,
                  data_root=str(args.data_root), runtime_dir=str(args.runtime_dir),
                  configuration=batch.CONFIG, configuration_sha256=config.sha256,
                  protected_sha256=protected,
                  code_sha256={str(path.relative_to(REPO)): batch.sha256_file(path)
                               for path in sorted(set((REPO / "scripts").glob("*.py")) | set((REPO / "src").rglob("*.py")))})
    batch.save(HERE / "run_inputs.json", inputs)

    def run(item):
        shot = item["shot"]
        shot_dir = args.data_root / shot
        output = args.runtime_dir / "rules" / shot
        assert not output.exists(), output
        before = batch.fingerprints(shot_dir)
        assert len(before) == preflight[shot]["current_mode_files"]
        batch.save(args.runtime_dir / "inputs" / f"{shot}.json", before)
        cmd = [sys.executable, str(REPO / "scripts" / ("sort_shot_mixed.py" if args.workflow == "production" else "sort_shot_rules.py"))]
        if args.workflow == "production":
            cmd += ["--method", "rules"]
        cmd += ["--rule_config", batch.CONFIG, "--shot_dir", str(shot_dir), "--out_dir", str(output)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        start = time.monotonic()
        with (args.runtime_dir / "logs" / f"{shot}.log").open("w") as log:
            subprocess.run(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        checker = batch.check_output if args.workflow == "production" else check_calibration
        s = checker(output, before, shot_dir, config.sha256)
        result = dict(shot=shot, command=cmd, input_modes=len(before), workflow=args.workflow,
                      tae_like=int(s["n_tae_like"]), mixed=int(s["n_mixed"]), eae_like=int(s["n_eae_like"]),
                      bad=int(s["n_final_bad"]), review=int(s["n_final_review"]), invalid=int(s["n_invalid"]),
                      good_before_dedup=int(s["n_final_good_before_clustering"]), selected_good=int(s["n_final_good"]),
                      elapsed_seconds=round(time.monotonic()-start, 2), output_sha256=batch.tree(output))
        batch.save(args.runtime_dir / "verified" / f"{shot}.json", result)
        return result

    verified, failed = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(run, item): item["shot"] for item in selected}
        for future in as_completed(pending):
            shot = pending[future]
            try:
                result = future.result()
                verified.append(result)
                print({k: v for k, v in result.items() if k not in ("command", "output_sha256")}, flush=True)
            except Exception as exc:
                failed.append(dict(shot=shot, error=f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {shot}: {exc}", flush=True)
            batch.save(HERE / "stage_results.json", dict(verified=verified, failed=failed))
    assert not failed, failed
    assert all(batch.sha256_file(REPO / path) == h for path, h in inputs["code_sha256"].items())
    assert all(batch.sha256_file(REPO / path) == h for path, h in protected.items())
    batch.write(HERE / "shot_summary.csv", [{k: v for k, v in r.items() if k not in ("command", "output_sha256")} for r in sorted(verified, key=lambda r: r["shot"])])
    print(f"Verified {len(verified)} shots; ready for installation.", flush=True)


if __name__ == "__main__":
    main()
