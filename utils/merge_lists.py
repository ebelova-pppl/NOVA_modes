import csv
import os
import sys
from collections import OrderedDict
from typing import Tuple, List


def parse_input_spec(spec: str) -> Tuple[str, str]:
    """
    Accepts:
      "file.csv"                    -> (file.csv, "")
      "file.csv@/abs/base/dir"      -> (file.csv, "/abs/base/dir")
    """
    if "@" in spec:
        csv_path, base = spec.split("@", 1)
        return csv_path, base.rstrip("/")
    return spec, ""


def normalize_path(p: str, base: str) -> str:
    p = p.strip()

    # Strip surrounding quotes if any
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()

    # Already absolute -> keep
    if os.path.isabs(p):
        return p

    # Remove leading "./"
    if p.startswith("./"):
        p = p[2:]

    if not base:
        # no base provided; return as-is (still relative)
        return p

    return os.path.join(base, p)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python merge_lists.py OUT.csv  IN1.csv[@BASE1]  IN2.csv[@BASE2] ...\n\n"
            "Examples:\n"
            "  python merge_lists.py train_master.csv \\\n"
            "    old_train_list.csv \\\n"
            "    nstx_120113/nstx_120113_labels.csv@/global/cfs/cdirs/m314/nova/nstx_120113 \\\n"
            "    nstx_135388/nstx_135388_labels.csv@/global/cfs/cdirs/m314/nova/nstx_135388 \\\n"
            "    nstx_141711/nstx_141711_labels.csv@/global/cfs/cdirs/m314/nova/nstx_141711\n\n"
            "Notes:\n"
            "  - Input CSV rows: path,label  (label must be good or bad)\n"
            "  - Relative paths are joined with BASE if provided.\n"
            "  - Later inputs override earlier ones for the same normalized path.\n"
        )
        sys.exit(1)

    out_csv = sys.argv[1]
    specs = sys.argv[2:]

    merged = OrderedDict()  # normalized_path -> label (latest wins)

    n_read = 0
    n_keep = 0
    n_skip_label = 0

    for spec in specs:
        csv_path, base = parse_input_spec(spec)
        print(f"Reading {csv_path}  base='{base}'")

        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row or len(row) < 2:
                    continue
                raw_path = row[0].strip()
                lab = row[1].strip().lower()
                n_read += 1

                if lab not in ("good", "bad"):
                    n_skip_label += 1
                    continue

                norm_path = normalize_path(raw_path, base)
                merged[norm_path] = lab
                n_keep += 1

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        for p, lab in merged.items():
            w.writerow([p, lab])

    print(f"\nWrote: {out_csv}")
    print(f"Rows read: {n_read}")
    print(f"Kept (good/bad): {n_keep}   Unique paths: {len(merged)}")
    print(f"Skipped (non good/bad): {n_skip_label}")

    # Optional: warn if any paths are still relative (base missing / mistakes)
    rel = [p for p in merged.keys() if not os.path.isabs(p)]
    if rel:
        print(f"\nWARNING: {len(rel)} paths are still relative (missing base?). Example:")
        print("  ", rel[0])


if __name__ == "__main__":
    main()

