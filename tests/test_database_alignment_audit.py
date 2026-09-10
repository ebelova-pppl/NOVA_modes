"""Protect diagnostic snapshots from stale inputs and failed live scans."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "audits/n1_database_alignment_20260910/check_database.py"
SPEC = importlib.util.spec_from_file_location("database_alignment", SCRIPT)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class DatabaseAlignmentTests(unittest.TestCase):
    def test_inventory_rejects_duplicates_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.csv"
            for names in ("shot\nshot\n", "../shot\n", "/shot\n"):
                path.write_text("shot\n" + names)
                with self.assertRaises(ValueError):
                    driver.inventory_rows(path)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.args = SimpleNamespace(data_root=root / "data", out_dir=root / "output")
        self.directory = self.args.data_root / "shot/N1"
        self.directory.mkdir(parents=True)
        (self.args.out_dir / "groups").mkdir(parents=True)
        self.mode = self.directory / "egn01w.test"
        self.mode.write_bytes(b"original")
        self.task = ("shot", 1, self.args, {"source": "unchanged"})

    def measurement(self, _):
        sources = {str(p): driver.audit.sha(p) for p in self.directory.iterdir() if p.is_file()}
        return [dict(path="shot/N1/egn01w.test")], [], sources

    def test_cache_requires_unchanged_bytes_and_file_inventory(self):
        with patch.object(driver.audit, "measure_shot", side_effect=self.measurement) as measure:
            self.assertFalse(driver.scan_group(self.task)[1])
            self.assertTrue(driver.scan_group(self.task)[1])
            self.mode.write_bytes(b"changed")
            self.assertFalse(driver.scan_group(self.task)[1])
            (self.directory / "out_go_prev").write_text("new log")
            self.assertFalse(driver.scan_group(self.task)[1])
            (self.directory / "egn01w.added").write_bytes(b"new mode")
            self.assertFalse(driver.scan_group(self.task)[1])
            self.assertEqual(measure.call_count, 4)

    def test_failed_group_is_explicit_and_never_reused(self):
        with patch.object(driver.audit, "measure_shot", side_effect=RuntimeError("input changed")):
            result, cached = driver.scan_group(self.task)
        self.assertFalse(cached)
        self.assertIn("input changed", result["error"])
        self.assertEqual(result["modes"], [])
        with patch.object(driver.audit, "measure_shot", side_effect=self.measurement):
            result, cached = driver.scan_group(self.task)
        self.assertFalse(cached)
        self.assertEqual(result["error"], "")

    def test_new_mode_during_scan_invalidates_whole_group(self):
        def changed(task):
            measured = self.measurement(task)
            (self.directory / "egn01w.added").write_bytes(b"new mode")
            return measured
        with patch.object(driver.audit, "measure_shot", side_effect=changed):
            result, _ = driver.scan_group(self.task)
        self.assertIn("Mode inventory changed", result["error"])
        self.assertEqual(result["modes"], [])


if __name__ == "__main__":
    unittest.main()
