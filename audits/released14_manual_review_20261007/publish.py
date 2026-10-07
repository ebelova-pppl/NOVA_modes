"""Install the two verified manual-review outputs, retaining previous trees.

Example: python audits/released14_manual_review_20261007/publish.py
Input/output locations come from the verified run_inputs.json receipt.
"""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "publisher", REPO / "audits/continuum_monotonic_tail_20260908/publish_regenerated.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def main():
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    verified = json.loads((HERE / "verification.json").read_text())
    assert verified["status"] == "verified"
    assert not (HERE / "publication.json").exists()
    assert all(publisher.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    assert all(publisher.sha256_file(HERE / p) == h for p, h in verified["files_sha256"].items())
    rules = Path(inputs["rules_root"])
    runtime = Path(inputs["runtime_dir"])
    backup = rules / "before_released14_manual_review_20261007"
    staging = rules / ".staging_released14_manual_review_20261007"
    assert not backup.exists() and not staging.exists()
    assert len(inputs["changed_shots"]) == 2
    for shot in inputs["shots"]:
        assert publisher.tree_digest(rules / shot) == inputs["old_trees"][shot]
    for shot in inputs["changed_shots"]:
        source = runtime / "rules" / shot
        assert publisher.tree_digest(source) == verified["new_trees"][shot]
        shutil.copytree(source, staging / shot)
        assert publisher.tree_digest(staging / shot) == verified["new_trees"][shot]
    backup.mkdir()
    installed = []
    receipt = dict(status="installing", installed=installed, backup_root=str(backup),
                   started_utc=datetime.now(timezone.utc).isoformat())
    for shot in inputs["changed_shots"]:
        target = rules / shot
        assert publisher.tree_digest(target) == inputs["old_trees"][shot]
        target.rename(backup / shot)
        try:
            (staging / shot).rename(target)
        except Exception:
            (backup / shot).rename(target)
            raise
        assert publisher.tree_digest(target) == verified["new_trees"][shot]
        assert publisher.tree_digest(backup / shot) == inputs["old_trees"][shot]
        installed.append(shot)
        (HERE / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Installed {shot}; previous output backed up", flush=True)
    for shot in inputs["shots"]:
        assert publisher.tree_digest(rules / shot) == verified["new_trees"][shot]
    staging.rmdir()
    receipt.update(status="installed", unchanged_shots=12,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    (HERE / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
