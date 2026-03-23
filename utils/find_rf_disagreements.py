import csv
import sys
import numpy as np
import joblib

from mode_features import compute_features_for_mode
from pathlib import Path
"""
Compare manual sorting list with RF list
use as:
python find_rf_disagreements.py \
  train_master.csv \
  nova_mode_classifier.joblib \
  rf_vs_manual_disagreements.csv   <-- output table
"""

# ---- NOVA loader ----
def load_mode_from_nova(path: str):
    f1 = np.fromfile(path)
    omega = float(f1[0])
    nr = int(f1[-3])
    gamma_d = float(f1[-2])
    ntor = int(round(float(f1[-1])))

    nhar = int((f1.size - 4) / (3 * nr))
    f11 = f1[1:-3].reshape(3, nhar, nr)
    mode = f11[0, :, :]   # xi_psi
    return mode, omega, gamma_d, ntor

def make_X_for_model(clf, mode, omega, gamma_d, ntor, path):
    """
    Build a feature vector matching the trained RF model.
    """
    expected = getattr(clf, "n_features_in_", None)

    extra = {"omega": omega, "gamma_d": gamma_d, "ntor": ntor, "path": path}

    feats = compute_features_for_mode(mode, extra_info=extra).astype(float)

    if expected is None or feats.size == expected:
        return feats.reshape(1, -1), extra

    raise ValueError(
        f"Feature length mismatch for {path}. Model expects {expected}, got {feats.size}. "
        f"Extras used: {extra}"
    )



def main():
    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python find_rf_disagreements.py manual_labels.csv "
            "nova_mode_classifier.joblib out_disagreements.csv\n"
        )
        sys.exit(1)

    manual_csv = sys.argv[1]
    rf_model_path = sys.argv[2]
    out_csv = sys.argv[3]

    clf = joblib.load(rf_model_path)
    print(f"Loaded RF model: {rf_model_path}")

    disagreements = []
    total = 0
    skipped = 0

    with open(manual_csv, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue

            path = row[0].strip()
            manual_lab = row[1].strip().lower()
            if manual_lab not in ("good", "bad"):
                continue

            total += 1

            try:
                mode, omega, gamma_d, ntor = load_mode_from_nova(path)
                X, used_extra = make_X_for_model(clf, mode, omega, gamma_d, ntor, path)
                p_good = float(clf.predict_proba(X)[0, 1])
                rf_lab = "good" if p_good >= 0.5 else "bad"

            except Exception as e:
                skipped += 1
                continue

            if rf_lab != manual_lab:
                disagreements.append(
                    (path, manual_lab, rf_lab, p_good)
                )

    # Sort by confidence (strong disagreements first)
    disagreements.sort(key=lambda x: abs(x[3] - 0.5), reverse=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "manual_label", "rf_label", "p_good"])
        for row in disagreements:
            w.writerow(row)

    print(f"Total checked modes: {total}")
    print(f"Skipped (read/feature errors): {skipped}")
    print(f"Disagreements found: {len(disagreements)}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

