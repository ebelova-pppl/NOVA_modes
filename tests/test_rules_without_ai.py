"""Production rules must finish duplicate selection without AI dependencies."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

# A fresh isolated interpreter prevents imports made by other tests from
# hiding an accidental dependency. Record attempts even if a fallback catches
# ModuleNotFoundError, since a successful run must not rely on that fallback.
GUARDED_RUNNER = """
import importlib.abc
from pathlib import Path
import runpy
import sys

blocked = {'joblib', 'sklearn', 'torch', 'matplotlib', 'pandas', 'narwhals'}
attempted = []

class BlockOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            attempted.append(fullname)
            raise ModuleNotFoundError('Optional package unavailable: ' + fullname)

sys.meta_path.insert(0, BlockOptionalPackages())
script = Path(sys.argv[1])
sys.path.insert(0, str(script.parent))
sys.argv = sys.argv[1:]
try:
    runpy.run_path(str(script), run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
assert not attempted, 'Unexpected optional-package imports: ' + repr(attempted)
assert not (blocked & {name.split('.')[0] for name in sys.modules})
"""


class RulesWithoutAITests(unittest.TestCase):
    def run_without_ai(self, script, *args, cwd):
        completed = subprocess.run(
            [
                sys.executable, "-I", "-B", "-c", GUARDED_RUNNER,
                str(REPO_ROOT / "scripts" / script), *map(str, args),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed

    def test_production_rules_complete_severity_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = root / "synthetic_shot"
            n_dir = shot / "N1"
            n_dir.mkdir(parents=True)
            nr = 201
            r = np.linspace(0.0, 1.0, nr)
            envelope = np.exp(-((r - 0.65) / 0.08) ** 2)
            # Two resolved, structurally equivalent modes with close omega.
            # Their small axis defects pass the gates but distinguish severity.
            for name, omega, axis_amplitude in (
                ("a", 1.0, 0.10), ("b", 1.005, 0.05)
            ):
                payload = np.zeros((3, 4, nr), dtype=float)
                payload[0, 1] = envelope
                payload[0, 2] = 0.5 * envelope
                payload[0, 0, 2] = axis_amplitude
                np.concatenate(
                    ([omega], payload.reshape(-1), [nr, 0.01, 1])
                ).tofile(n_dir / f"egn01w.{name}")
            (n_dir / "datcon1").write_text(
                f"1 {nr}\n" + "0.25 2.25\n" * nr
            )
            output = root / "results"
            self.run_without_ai(
                "sort_shot_mixed.py", "--method", "rules",
                "--shot_dir", shot, "--out_dir", output, cwd=root,
            )
            with (output / "good_tae_unchecked.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with (output / "good_tae_final.csv").open() as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual(len(selected), 1)
            self.assertTrue(selected[0]["path"].endswith("egn01w.b"))
            self.assertEqual(selected[0]["duplicate_rank_source"], "rule_severity")
            with (output / "shot_summary_wide.csv").open() as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(
                summary["duplicate_processing_status"], "COMPLETED_RULE_SEVERITY"
            )

    def test_legacy_sorter_help_does_not_import_ai(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = self.run_without_ai("sort_shot.py", "--help", cwd=temporary)
        self.assertIn("--model", completed.stdout)


if __name__ == "__main__":
    unittest.main()
