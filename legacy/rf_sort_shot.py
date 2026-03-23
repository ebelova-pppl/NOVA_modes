#!/usr/bin/env python3
import argparse
import os
import shutil
import glob
import csv
from pathlib import Path

import numpy as np
import joblib

# Assuming that this script is in the same /RF/ directory as 
# nova_mode_loader.py / mode_features.py,
from nova_mode_loader import load_mode_from_nova
from mode_features import compute_features_for_mode

"""
This script walks a shot directory like .../nstx_123456/
* finds N1 … N10 subdirs (or whatever exists)
* scans all files matching egn* in each N#
* runs existing RF joblib classifier on each mode file
* writes a per-shot CSV list: path,label,p_good
* optionally moves bad modes into N#/out/ (creating it if needed)
NOTE, threshold =0.5 means that mode is bad if p_good < 0.5

Run as (to preview):
python rf_sort_shot.py /global/cfs/cdirs/m314/nova/nstx_123456 \
  --model nova_mode_classifier.joblib \
  --threshold 0.5 \
  --move_bad --dry_run

Run as (to actually move bad modes in /out/*):
python rf_sort_shot.py /global/cfs/cdirs/m314/nova/nstx_123456 \
  --model nova_mode_classifier.joblib \
  --threshold 0.5 \
  --move_bad
"""

def iter_n_dirs(shot_dir: str, n_min=1, n_max=10):
    shot = Path(shot_dir)
    for n in range(n_min, n_max + 1):
        ndir = shot / f"N{n}"
        if ndir.is_dir():
            yield n, ndir


def classify_file(clf, mode_path: str):
    mode, omega, gamma_d, ntor = load_mode_from_nova(mode_path)

    extra_info = {
        "path": mode_path,
        "omega": float(omega),
        "gamma_d": float(gamma_d),
        "ntor": float(ntor),
    }
    x = compute_features_for_mode(mode, extra_info).reshape(1, -1)

    # Works for sklearn Pipeline or plain estimator
    if hasattr(clf, "predict_proba"):
        p_good = float(clf.predict_proba(x)[0, 1])
    else:
        # fallback: use decision_function -> sigmoid-ish
        if hasattr(clf, "decision_function"):
            z = float(clf.decision_function(x)[0])
            p_good = 1.0 / (1.0 + np.exp(-z))
        else:
            p_good = float(clf.predict(x)[0])

    return p_good


def main():
    ap = argparse.ArgumentParser(
        description="Classify NOVA egn* modes in a shot directory with an RF model; optionally move bad modes to N#/out/."
    )
    ap.add_argument("shot_dir", help="Shot directory, e.g. /global/cfs/cdirs/.../nstx_123456")
    ap.add_argument("--model", required=True, help="RF model joblib, e.g. nova_mode_classifier.joblib")
    ap.add_argument("--out_csv", default=None, help="Output CSV path. Default: <shot_dir>/rf_sorted.csv")
    ap.add_argument("--threshold", type=float, default=0.5, help="Good if p_good >= threshold (default 0.5)")
    ap.add_argument("--move_bad", action="store_true", help="Move bad modes to N#/out/")
    ap.add_argument("--dry_run", action="store_true", help="Do not move files; just report and write CSV")
    ap.add_argument("--n_min", type=int, default=1)
    ap.add_argument("--n_max", type=int, default=10)
    ap.add_argument("--pattern", default="egn*", help="Glob pattern for mode files (default egn*)")
    args = ap.parse_args()

    shot_dir = Path(args.shot_dir).resolve()
    if not shot_dir.is_dir():
        raise SystemExit(f"Shot dir not found: {shot_dir}")

    out_csv = Path(args.out_csv) if args.out_csv else shot_dir / "rf_sorted.csv"

    clf = joblib.load(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Shot dir: {shot_dir}")
    print(f"Threshold: {args.threshold}  (good if p_good >= thr)")
    if args.move_bad:
        print("Will move BAD modes to N#/out/  (use --dry_run to preview)")

    rows = []
    n_total = 0
    n_good = 0
    n_bad = 0
    n_err = 0

    for n, ndir in iter_n_dirs(str(shot_dir), args.n_min, args.n_max):
        files = sorted(glob.glob(str(ndir / args.pattern)))
        if not files:
            continue

        out_dir = ndir / "out"
        if args.move_bad and not args.dry_run:
            out_dir.mkdir(exist_ok=True)

        print(f"\nN{n}: found {len(files)} files in {ndir}")

        for f in files:
            n_total += 1
            try:
                p_good = classify_file(clf, f)
                label = "good" if p_good >= args.threshold else "bad"
            except Exception as e:
                n_err += 1
                rows.append([f, "error", "", str(e)])
                print(f"  ERROR {f}: {e}")
                continue

            rows.append([f, label, f"{p_good:.6f}", ""])
            if label == "good":
                n_good += 1
            else:
                n_bad += 1
                if args.move_bad:
                    dest = out_dir / Path(f).name
                    if args.dry_run:
                        print(f"  would move BAD: {f} -> {dest}")
                    else:
                        # avoid overwrite
                        if dest.exists():
                            # add suffix
                            stem = dest.stem
                            suf = dest.suffix
                            k = 1
                            while True:
                                alt = out_dir / f"{stem}__dup{k}{suf}"
                                if not alt.exists():
                                    dest = alt
                                    break
                                k += 1
                        shutil.move(f, dest)

        print(f"  N{n} done. cumulative: total={n_total} good={n_good} bad={n_bad} err={n_err}")

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["path", "label", "p_good", "error"])
        w.writerows(rows)

    print("\n=== Summary ===")
    print(f"Total: {n_total} | Good: {n_good} | Bad: {n_bad} | Errors: {n_err}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

