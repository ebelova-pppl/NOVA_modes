#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from nova_mode_loader import load_mode_from_nova      # returns (mode, omega, gamma_d, ntor)
from mode_features import compute_features_for_mode   # includes full extra_info dict


def read_train_csv(csv_path: str):
    paths = []
    y = []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            p = row[0].strip()
            lab = row[1].strip().lower()
            if lab not in ("good", "bad"):
                continue
            paths.append(p)
            y.append(1 if lab == "good" else 0)
    return paths, np.asarray(y, dtype=int)


def build_X(paths):
    X = []
    bad = 0
    for p in paths:
        try:
            mode, omega, gamma_d, ntor = load_mode_from_nova(p)
            extra = {
                "path": p,
                "omega": float(omega),
                "gamma_d": float(gamma_d),
                "ntor": int(round(float(ntor))),
            }
            feats = compute_features_for_mode(mode, extra_info=extra).astype(float)
            X.append(feats)
        except Exception as e:
            bad += 1
            raise RuntimeError(f"Failed building features for {p}: {e}") from e

    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        i, j = np.argwhere(~np.isfinite(X))[0]
        raise ValueError(f"Non-finite feature at row {i}, col {j}, path={paths[i]}")
    return X


def oof_predict_proba(clf, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p_oof = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = clone(clf)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        p_oof[te] = p
        print(f"Fold {fold}: train={len(tr)} test={len(te)}")

    if not np.isfinite(p_oof).all():
        raise RuntimeError("OOF predictions contain NaNs (bug in CV loop).")
    return p_oof


def write_oof_table(out_csv, paths, y, p_oof, thr=0.5):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "manual_label", "p_good_oof", "oof_pred"])
        for p, yi, pi in zip(paths, y, p_oof):
            manual = "good" if yi == 1 else "bad"
            pred = "good" if pi >= thr else "bad"
            w.writerow([p, manual, f"{pi:.6f}", pred])


def write_suspects(out_csv, paths, y, p_oof, thr_low=0.2, thr_high=0.8):
    rows = []
    for p, yi, pi in zip(paths, y, p_oof):
        manual = "good" if yi == 1 else "bad"
        if yi == 1 and pi < thr_low:
            # manual good but model very confident it's bad
            rows.append((p, manual, pi, "manual_good_but_oof_bad", (thr_low - pi)))
        elif yi == 0 and pi > thr_high:
            # manual bad but model very confident it's good
            rows.append((p, manual, pi, "manual_bad_but_oof_good", (pi - thr_high)))

    # Sort by "how strong" the disagreement is
    rows.sort(key=lambda t: t[-1], reverse=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "manual_label", "p_good_oof", "flag"])
        for p, manual, pi, flag, _strength in rows:
            w.writerow([p, manual, f"{pi:.6f}", flag])

    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="OOF (StratifiedKFold) label-audit for NOVA RF classifier.")
    ap.add_argument("train_csv", help="CSV with rows: path,label (good/bad)")
    ap.add_argument("--model_in", default="nova_mode_classifier.joblib",
                    help="Existing trained model (joblib) used as a template for hyperparams/pipeline.")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr", type=float, default=0.5, help="threshold for printing OOF confusion matrix")
    ap.add_argument("--thr_low", type=float, default=0.2, help="suspect if manual=good and p<thr_low")
    ap.add_argument("--thr_high", type=float, default=0.8, help="suspect if manual=bad and p>thr_high")
    ap.add_argument("--out_oof", default="oof_table.csv")
    ap.add_argument("--out_suspects", default="oof_suspects.csv")
    args = ap.parse_args()

    paths, y = read_train_csv(args.train_csv)
    print(f"Loaded labels: {len(paths)} modes  (good={int(y.sum())}, bad={len(y)-int(y.sum())})")

    # Use model as a template. (Assumes it’s a Pipeline ending in RF with predict_proba)
    clf = joblib.load(args.model_in)
    print(f"Loaded model template: {args.model_in}")

    X = build_X(paths)
    print("Built X:", X.shape)

    p_oof = oof_predict_proba(clf, X, y, n_splits=args.splits, seed=args.seed)

    # Basic OOF metrics
    y_pred = (p_oof >= args.thr).astype(int)
    print("\nOOF confusion matrix (rows=actual [bad,good], cols=pred [bad,good]):")
    print(confusion_matrix(y, y_pred))
    print("\nOOF classification report:")
    print(classification_report(y, y_pred, target_names=["bad", "good"]))

    write_oof_table(args.out_oof, paths, y, p_oof, thr=args.thr)
    n_susp = write_suspects(args.out_suspects, paths, y, p_oof, thr_low=args.thr_low, thr_high=args.thr_high)

    print(f"\nWrote OOF table: {args.out_oof}")
    print(f"Wrote suspects:  {args.out_suspects}  (n={n_susp})")


if __name__ == "__main__":
    main()

