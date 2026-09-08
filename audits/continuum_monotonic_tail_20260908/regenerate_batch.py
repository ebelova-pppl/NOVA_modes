"""Regenerate the reviewed 27-shot membership using both canonical methods.

Writes fresh local exports; publication and backups are separate operations.
Example:
  python audits/continuum_monotonic_tail_20260908/regenerate_batch.py \
    --data-root /path/to/DiTw --out-root outputs/continuum_tail_adopted_20260908 \
    --rf-model models/nova_mode_classifier.joblib --cnn-model models/nova_cnn_raw.pt
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from cont_features import CONTINUUM_PREPROCESSING_VERSION
from tae_rule_config import load_rule_run_configuration
from tae_rule_io import sha256_file


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data-root", "out-root", "rf-model", "cnn-model"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=2, help="Concurrent CPU sorter processes"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = load_rule_run_configuration("tae_rules_production_v6")
    memberships = [
        REPO / "audits/regression15_v5/shot_summary.csv",
        REPO / "audits/pilot12_v5_20260908/shot_summary.csv",
    ]
    shots = [row for p in memberships for row in read_csv(p)]
    assert len(shots) == len({row["shot"] for row in shots}) == 27
    code = sorted((REPO / "src").glob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    provenance = {
        "continuum_preprocessing_version": CONTINUUM_PREPROCESSING_VERSION,
        "rule_configuration_sha256": config.sha256,
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in code},
        "membership_sha256": {
            str(p.relative_to(REPO)): sha256_file(p) for p in memberships
        },
        "rf_model_sha256": sha256_file(args.rf_model),
        "cnn_model_sha256": sha256_file(args.cnn_model),
        "data_root": str(args.data_root.resolve()),
    }
    signature = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    def run(shot, method):
        directory = args.out_root / method / shot["shot"]
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "regeneration_complete.json"
        if marker.exists():
            result = json.loads(marker.read_text())
            if result["signature"] == signature:
                return result
        command = [
            sys.executable,
            str(REPO / "scripts/sort_shot_mixed.py"),
            "--method",
            method,
            "--shot_dir",
            str(args.data_root / shot["shot"]),
            "--out_dir",
            str(directory),
            "--rf_model",
            str(args.rf_model),
        ]
        if method == "rf-cnn":
            command += ["--cnn_model", str(args.cnn_model), "--device", "cpu"]
        started = time.monotonic()
        with (directory / "regeneration.log").open("w") as log:
            subprocess.run(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        summary = read_csv(directory / "shot_summary_wide.csv")[0]
        assert int(summary["n_total_files"]) == int(shot["input_modes"]), directory
        assert (
            summary["continuum_preprocessing_version"]
            == CONTINUUM_PREPROCESSING_VERSION
        )
        assert float(summary["fraction_direct_eae_threshold"]) == 0.2
        assert summary["method"] == method
        result = dict(
            shot=shot["shot"],
            method=method,
            signature=signature,
            input_modes=int(summary["n_total_files"]),
            final_good=int(summary["n_final_good"]),
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        marker.write_text(json.dumps(result, indent=2) + "\n")
        return result

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(run, shot, method)
            for shot in shots
            for method in ("rules", "rf-cnn")
        ]
        for job in as_completed(jobs):
            result = job.result()
            completed.append(result)
            print(
                f"{len(completed)}/54 {result['shot']} {result['method']}: "
                f"{result['final_good']} selected GOOD ({result['elapsed_seconds']}s)",
                flush=True,
            )
    for path in code:
        assert (
            sha256_file(path)
            == provenance["source_sha256"][str(path.relative_to(REPO))]
        ), path
    assert sha256_file(args.rf_model) == provenance["rf_model_sha256"]
    assert sha256_file(args.cnn_model) == provenance["cnn_model_sha256"]
    (args.out_root / "regeneration_runs.json").write_text(
        json.dumps(sorted(completed, key=lambda r: (r["shot"], r["method"])), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
