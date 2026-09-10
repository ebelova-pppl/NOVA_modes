"""Guard frequency/log pairing and interpretation of the alignment screen."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "audits/n1_training_alignment_20260910/check_alignment.py"
SPEC = importlib.util.spec_from_file_location("continuum_log_alignment", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AlignmentTests(unittest.TestCase):
    def test_log_block_cannot_leak_across_frequencies(self):
        text = """Singularities are expected at 1 points:
 ixmax vs ising 100 81 4.0D-01 0
 hhh,om 9.0 2.0D+00
 hhh,om 9.0 3.0D+00
 Singularities are expected at 0 points:
 hhh,om 9.0 4.0D+00
 Singularities are expected at 2 points:
 ixmax vs ising 100 81 0.4 0
 hhh,om 9.0 5.0D+00
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out_go"
            path.write_text("\x00" + text)
            records = audit.read_log(path)
        self.assertEqual([r["omega2"] for r in records], [2,4,5])
        self.assertEqual(audit.match_log(records,3)[0],"NO_FREQUENCY_MATCH")
        self.assertEqual(audit.match_log(records,4)[0],"NO_LOGGED_SINGULARITIES")
        self.assertEqual(audit.match_log(records,5)[0],"INCOMPLETE_LOG_BLOCK")

    def test_matching_requires_exact_frequency_and_unambiguous_radii(self):
        record = dict(omega2=2.,radii=[.4,.6],complete=True)
        self.assertEqual(audit.match_log([record],2+1e-13)[0],"MATCHED")
        self.assertEqual(audit.match_log([record],2+1e-5)[0],"NO_FREQUENCY_MATCH")
        self.assertEqual(audit.match_log([record,dict(record,radii=[.6,.4])],2)[0],"MATCHED")
        self.assertEqual(audit.match_log([record,dict(record,radii=[.45,.6])],2)[0],"CONFLICTING_LOG_RECORDS")

    def test_nearest_distance_sign_and_resolution(self):
        crossings = [dict(boundary="high",r_cross=.43)]
        first = audit.crossing_offsets(crossings,[.4,.8],201)[0]
        finer = audit.crossing_offsets(crossings,[.4,.8],401)[0]
        self.assertAlmostEqual(first["offset_grid"],6)
        self.assertAlmostEqual(finer["offset_grid"],12)
        self.assertAlmostEqual(first["offset_r"],finer["offset_r"])
        self.assertLess(audit.crossing_offsets(crossings,[.46],201)[0]["offset_grid"],0)

    def test_empty_cohort_has_no_false_alignment(self):
        rows = audit.summaries(["shot"],[1],[],[])
        self.assertEqual(len(rows),3)
        self.assertTrue(all(row["median_abs_grid"] == "" for row in rows))
        self.assertTrue(all(row["n_interior_crossings"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
