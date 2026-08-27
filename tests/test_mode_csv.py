import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mode_csv import read_mode_csv_entries  # noqa: E402


class ModeCsvTests(unittest.TestCase):
    def test_reads_sorter_audit_mode_key_and_training_validity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "packet_new_rejections.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    ["shot", "mode_key", "training_validity", "ntor"]
                )
                writer.writerow(
                    [
                        "nstx_135388",
                        "nstx_135388/N8/egn08w.1998E+02",
                        "bad",
                        "8",
                    ]
                )

            entries = read_mode_csv_entries(csv_path, data_root=root / "data")

        self.assertEqual(
            entries,
            [
                (
                    str(
                        (
                            root
                            / "data/nstx_135388/N8/egn08w.1998E+02"
                        ).resolve()
                    ),
                    "bad",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
