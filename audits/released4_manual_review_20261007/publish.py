"""Install the verified N75 correction, preserving its previous output.

Example: python audits/released4_manual_review_20261007/publish.py
Locations and expected hashes come from this review's receipts.
"""
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


def main():
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    verified = json.loads((HERE / "verification.json").read_text())
    assert verified["status"] == "verified" and not (HERE / "publication.json").exists()
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    assert all(batch.sha256_file(HERE / p) == h for p, h in verified["files_sha256"].items())
    rules = Path(inputs["rules_root"])
    runtime = Path(inputs["runtime_dir"])
    shot = inputs["changed_shot"]
    assert shot == "nstxuG121123N75"
    for name in inputs["shots"]:
        assert batch.tree(rules / name) == inputs["old_trees"][name]
    assert batch.fingerprints(Path(inputs["data_root"]) / shot) == json.loads(
        (runtime / "input_fingerprints.json").read_text())
    source = runtime / "rules" / shot
    assert batch.tree(source) == verified["new_trees"][shot]
    backup = rules / "before_n75_manual_review_20261007" / shot
    staging = rules / f".{shot}.installing_manual_review_20261007"
    assert not backup.exists() and not staging.exists()
    shutil.copytree(source, staging)
    assert batch.tree(staging) == verified["new_trees"][shot]
    backup.parent.mkdir(exist_ok=True)
    target = rules / shot
    target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        backup.rename(target)
        raise
    assert batch.tree(backup) == inputs["old_trees"][shot]
    for name in inputs["shots"]:
        assert batch.tree(rules / name) == verified["new_trees"][name]
    batch.save(HERE / "publication.json", dict(status="installed", installed=[shot],
        backup=str(backup), unchanged_shots=3, finished_utc=datetime.now(timezone.utc).isoformat()))
    print("Installed N75 manual correction; previous output backed up, other three shots unchanged.")


if __name__ == "__main__":
    main()
