"""Install verified shot outputs into new directories without replacing anything.

Example: python audits/cleared_nan_rules_20261005/install.py \
  --destination-root /path/to/sort_outputs/calibration_20261005
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from run_batch import HERE, REPO, batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination-root", required=True, type=Path)
    args = p.parse_args()
    destination = args.destination_root.resolve()
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    results = json.loads((HERE / "stage_results.json").read_text())
    assert not results["failed"] and len(results["verified"]) == 5
    assert not (HERE / "publication.json").exists()
    assert all(batch.sha256_file(REPO / path) == h for path, h in inputs["protected_sha256"].items())
    assert all(batch.sha256_file(REPO / path) == h for path, h in inputs["code_sha256"].items())
    runtime, data = Path(inputs["runtime_dir"]), Path(inputs["data_root"])
    for result in results["verified"]:
        shot = result["shot"]
        assert not (destination / shot).exists(), f"Existing output: {shot}"
        assert batch.tree(runtime / "rules" / shot) == result["output_sha256"]
        before = json.loads((runtime / "inputs" / f"{shot}.json").read_text())
        assert batch.fingerprints(data / shot) == before, f"Changed inputs: {shot}"
    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="installing",
                   destination_root=str(destination), workflow=inputs["workflow"], installed=installed,
                   existing_outputs_replaced=0)
    batch.save(HERE / "publication.json", receipt)
    for result in sorted(results["verified"], key=lambda r: r["shot"]):
        shot = result["shot"]
        temporary = destination / f".{shot}.installing_20261005"
        assert not temporary.exists()
        shutil.copytree(runtime / "rules" / shot, temporary)
        assert batch.tree(temporary) == result["output_sha256"]
        temporary.rename(destination / shot)
        installed.append(shot)
        batch.save(HERE / "publication.json", receipt)
        print(f"Installed {shot}", flush=True)
    receipt["status"] = "installed"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    batch.save(HERE / "publication.json", receipt)


if __name__ == "__main__":
    main()
