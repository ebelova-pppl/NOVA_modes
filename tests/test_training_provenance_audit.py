import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_training_provenance import (  # noqa: E402
    SCHEMA_VERSION,
    build_manifest,
    build_shot_summary,
    read_training_rows,
    run_audit,
)


def _write_mode(
    path: Path,
    *,
    omega: float,
    ntor: int,
    gamma: float = 0.01,
    nr: int = 5,
    nhar: int = 2,
    offset: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.arange(3 * nr * nhar, dtype=np.float64) + offset
    values = np.concatenate(
        (
            np.array([omega], dtype=np.float64),
            payload,
            np.array([nr, gamma, ntor], dtype=np.float64),
        )
    )
    values.tofile(path)


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "validity"))
        writer.writerows(rows)


class TrainingProvenanceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.training = self.root / "training"
        self.reference = self.root / "reference"
        self.training.mkdir()
        self.reference.mkdir()
        self.train_csv = self.root / "train.csv"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _populate(self) -> None:
        identical = "shotA/N1/egn01w.1000E+00"
        changed = "shotA/N1/egn01w.2000E+00"
        missing_both = "shotB/N2/egn02w.3000E+00"
        _write_csv(
            self.train_csv,
            ((identical, "good"), (changed, "bad"), (missing_both, "bad")),
        )

        _write_mode(self.training / identical, omega=1.0, ntor=1)
        _write_mode(self.reference / identical, omega=1.0, ntor=1)
        _write_mode(self.training / changed, omega=2.0, ntor=1)
        _write_mode(self.reference / changed, omega=2.2, ntor=1, offset=1.0)
        _write_mode(
            self.training / "shotA/N1/egn01w.4000E+00",
            omega=4.0,
            ntor=1,
        )
        _write_mode(
            self.reference / "shotA/N1/egn01w.5000E+00",
            omega=5.0,
            ntor=1,
        )

        training_n = self.training / "shotA/N1"
        reference_n = self.reference / "shotA/N1"
        (training_n / "datcon1").write_text("old continuum\n")
        (reference_n / "datcon1").write_text("new continuum\n")
        (training_n / "datcon1_old").write_text("older continuum\n")
        (reference_n / "datcon_gf.txt").write_text("auxiliary\n")

    def test_manifest_tracks_mode_differences_and_datcon_backup(self) -> None:
        self._populate()
        training_rows = read_training_rows(self.train_csv)
        manifest = build_manifest(self.training, self.reference, training_rows)
        indexed = {
            (row["file_kind"], row["training_relative_path"]): row
            for row in manifest
        }

        identical = indexed[("mode", "shotA/N1/egn01w.1000E+00")]
        self.assertEqual(identical["status"], "identical")
        self.assertEqual(identical["training_label"], "good")

        changed = indexed[("mode", "shotA/N1/egn01w.2000E+00")]
        self.assertEqual(changed["status"], "different")
        self.assertEqual(changed["mode_structure_equal"], "false")
        self.assertAlmostEqual(changed["omega_relative_change_abs"], 0.1)
        self.assertGreater(changed["mode_max_abs_delta"], 0.0)

        self.assertEqual(
            indexed[("datcon", "shotA/N1/datcon1")]["status"],
            "different",
        )
        backup = indexed[("datcon_backup", "shotA/N1/datcon1_old")]
        self.assertEqual(backup["reference_relative_path"], "shotA/N1/datcon1")
        self.assertEqual(backup["status"], "different")
        self.assertEqual(
            indexed[("datcon_gf", "shotA/N1/datcon_gf.txt")]["status"],
            "reference_only",
        )

        summary = {row["shot"]: row for row in build_shot_summary(manifest, training_rows)}
        self.assertEqual(summary["shotA"]["canonical_rows"], 2)
        self.assertEqual(summary["shotA"]["canonical_identical"], 1)
        self.assertEqual(summary["shotA"]["canonical_different"], 1)
        self.assertEqual(summary["shotA"]["mode_training_only"], 1)
        self.assertEqual(summary["shotA"]["mode_reference_only"], 1)
        self.assertEqual(summary["shotA"]["datcon_backup_count"], 1)
        self.assertEqual(
            summary["shotA"]["overall_status"],
            "mode_and_continuum_mismatch",
        )
        self.assertEqual(summary["shotB"]["canonical_missing_both"], 1)
        self.assertEqual(summary["shotB"]["mode_missing_both"], 1)

    def test_run_audit_writes_self_describing_versioned_artifacts(self) -> None:
        self._populate()
        out_dir = self.root / "audit_v1"
        metadata = run_audit(
            training_root=self.training,
            reference_root=self.reference,
            train_csv=self.train_csv,
            out_dir=out_dir,
            audit_id="unit-test-v1",
            generated_at="2026-08-20T12:00:00+00:00",
        )

        self.assertEqual(metadata["schema_version"], SCHEMA_VERSION)
        self.assertEqual(metadata["canonical_row_count"], 3)
        for filename in (
            "file_manifest.csv",
            "differences.csv",
            "shot_summary.csv",
            "report.md",
            "run_metadata.json",
            "SHA256SUMS",
        ):
            self.assertTrue((out_dir / filename).is_file(), filename)

        report = (out_dir / "report.md").read_text()
        self.assertIn("unit-test-v1", report)
        self.assertIn("Canonical same-name mode files differ", report)
        self.assertIn("Preserved `datconN_old` files", report)

        persisted = json.loads((out_dir / "run_metadata.json").read_text())
        self.assertEqual(persisted["generated_at"], "2026-08-20T12:00:00+00:00")
        checksums = {}
        for line in (out_dir / "SHA256SUMS").read_text().splitlines():
            digest, filename = line.split("  ", 1)
            checksums[filename] = digest
        for filename, expected in checksums.items():
            actual = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)

        with self.assertRaises(FileExistsError):
            run_audit(
                training_root=self.training,
                reference_root=self.reference,
                train_csv=self.train_csv,
                out_dir=out_dir,
                audit_id="unit-test-v1",
                generated_at="2026-08-20T12:00:00+00:00",
            )

    def test_training_csv_rejects_parent_traversal(self) -> None:
        _write_csv(self.train_csv, (("../shot/N1/egn01w.1000E+00", "good"),))
        with self.assertRaisesRegex(ValueError, "unsafe relative path"):
            read_training_rows(self.train_csv)


if __name__ == "__main__":
    unittest.main()
