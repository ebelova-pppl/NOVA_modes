"""Publish verified batch exports while preserving each previous shot directory.

Example:
  python audits/continuum_monotonic_tail_20260908/publish_regenerated.py \
    --out-root outputs/continuum_tail_adopted_20260908 \
    --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
    --audit-dir audits/continuum_monotonic_tail_20260908 \
    --backup-name before_continuum_tail_20260908
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from tae_rule_io import sha256_file


def tree_digest(root):
    """Hash file paths and content within this one known shot-export directory."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in export: {path}")
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(sha256_file(path).encode() + b"\0")
            count += 1
    return dict(files=count, tree_sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("out-root", "rules-root", "ai-root", "audit-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument(
        "--backup-name",
        required=True,
        help="New backup subdirectory within each output root",
    )
    args = parser.parse_args()
    if Path(args.backup_name).name != args.backup_name or args.backup_name in {
        ".",
        "..",
    }:
        parser.error("--backup-name must be a single directory name")
    verification = json.loads(
        (args.audit_dir / "adoption_verification.json").read_text()
    )
    assert verification["status"] == "adopted_and_regenerated_locally"
    assert (
        verification["canonical_runs"] == 54
        and verification["checks"]["paired_valid_mode_matches"] == 19228
    )
    provenance = json.loads((args.out_root / "run_provenance.json").read_text())
    assert provenance == verification["regeneration_provenance"]
    for name, digest in provenance["source_sha256"].items():
        assert sha256_file(REPO / name) == digest, name
    runs = json.loads((args.out_root / "regeneration_runs.json").read_text())
    assert len(runs) == 54
    destinations = {"rules": args.rules_root, "rf-cnn": args.ai_root}
    plans = []
    for run in runs:
        method, shot = run["method"], run["shot"]
        source = args.out_root / method / shot
        target = destinations[method] / shot
        backup = destinations[method] / args.backup_name / shot
        staging = destinations[method] / (".staging_" + args.backup_name) / shot
        assert target.is_dir() and not backup.exists() and not staging.exists(), target
        assert json.loads((source / "regeneration_complete.json").read_text()) == run
        plans.append(
            dict(
                method=method,
                shot=shot,
                source=source,
                target=target,
                backup=backup,
                staging=staging,
                new=tree_digest(source),
                old=tree_digest(target),
            )
        )

    # Prepare and verify every copy before replacing any current export.
    for plan in plans:
        plan["staging"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(plan["source"], plan["staging"])
        assert tree_digest(plan["staging"]) == plan["new"]
    published = []
    for plan in plans:
        target, backup, staging = plan["target"], plan["backup"], plan["staging"]
        assert tree_digest(target) == plan["old"], target
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)
        try:
            staging.rename(target)
        except Exception:
            backup.rename(target)
            raise
        assert tree_digest(target) == plan["new"]
        assert tree_digest(backup) == plan["old"]
        published.append(
            dict(
                method=plan["method"],
                shot=plan["shot"],
                output=str(target),
                backup=str(backup),
                current=plan["new"],
                previous=plan["old"],
            )
        )
        print(plan["method"], plan["shot"], "published and backed up", flush=True)
    for root in destinations.values():
        (root / (".staging_" + args.backup_name)).rmdir()
    (args.audit_dir / "publication.json").write_text(
        json.dumps(published, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
