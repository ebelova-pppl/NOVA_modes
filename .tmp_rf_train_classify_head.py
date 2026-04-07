import numpy as np
import csv
import os
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from mode_features import compute_features_for_mode
from path_utils import resolve_mode_csv_path


# ========================
# 1. DATA LOADING
# ========================

def load_mode_from_nova(path):
    """
    It returns a 2D numpy array: mode[m_index, r_index]

    Here f1 is 1D array of size 3*nr*nhar + 4,
    it contains 3 perturbations: xi_psi, delta_p, xi_surf,
    and 4 scalar parameters: 
    f1[0]= omega, 
    f1[-3]= nr (=number of radial points),
    f1[-2]= gamma_d of continuum decay, 
    f1[-1]= ntor

    """
    f1 = np.fromfile(path)

    omega = float(f1[0])
    nr = int(f1[-3])
    gamma_d = float(f1[-2])
    ntor = int(round(float(f1[-1])))

    payload = f1.size - 4
    denom = 3 * nr
    if payload % denom != 0:
        raise ValueError(
            f"Bad NOVA file size for {path}: (size-4)={payload} not divisible by 3*nr={denom}"
        )
    nhar = payload // denom        # number of poloidal harmonics

    f11 = f1[1:-3].reshape(3, nhar, nr)
    mode = f11[0, :, :]  # already (nhar, nr)

    return mode, omega, gamma_d, ntor


def load_labeled_modes(csv_path):
    """
    csv_path: CSV with lines 'filepath,label' where label is 'good' or 'bad'.
    Returns:
        modes: list of 2D numpy arrays
        y:     list of integer labels (0 = bad, 1 = good)
        paths: list of filepaths
    """
    modes = []
    labels = []
    paths = []
    extra_infos = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            file_path = resolve_mode_csv_path(row[0])
            label_str = row[1].strip().lower()
            mode, omega, gamma_d, ntor = load_mode_from_nova(file_path)
            extra_info = {
               "omega": omega,
               "gamma_d": gamma_d,
               "ntor": ntor,
               "path": file_path,
            }

            feats = compute_features_for_mode(mode, extra_info)

            # map labels: good -> 1, bad -> 0
            if label_str not in ("good", "bad"):
                raise ValueError(f"Unknown label {label_str} in {csv_path}")
            label = 1 if label_str == "good" else 0

            modes.append(mode)
            labels.append(label)
            paths.append(file_path)
            extra_infos.append(extra_info)

    return modes, np.array(labels), paths, extra_infos


# ========================
# 2. FEATURE ENGINEERING
# ========================
# compute_features_for_mode() was separated out and put into mode_features.py

def build_feature_matrix(modes, extra_infos):
    """
    modes: list of 2D arrays
    Returns: X 2D array (n_samples, n_features)
    """
    feats = [
        compute_features_for_mode(m, extra_info) 
        for m, extra_info in zip(modes, extra_infos)
    ]
    return np.vstack(feats)

# ========================
# 3. TRAINING
# ========================

def train_classifier(X, y):
    """
    X: (n_samples, n_features)
    y: (n_samples,)  0/1 labels
    Returns:
        clf: fitted sklearn Pipeline (StandardScaler + RandomForest)
    """

    # Pipeline: scale features -> random forest
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # With small number of samples, use cross-validation to sanity-check.
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-validation accuracies:", scores)
    print("Mean CV accuracy:", scores.mean())
    print('Trained on',y.size,'modes total')

    # Fit on full dataset (or hold out a small test set if you like)
    clf.fit(X, y)
    return clf


def print_feature_importance(clf, feature_names):

    # Access the RF inside the pipeline
    rf = clf.named_steps["rf"]
    importances = rf.feature_importances_
    
    print("\n=== Feature Importances ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"{name:25s}  {imp:.4f}")


def evaluate_classifier(clf, X, y):
    """
    Optional: split out a test set and print metrics.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred))

    # Re-fit on full dataset for deployment
    clf.fit(X, y)
    return clf


# ========================
# 4. SAVING / LOADING / INFERENCE
# ========================

def save_model(clf, path):
    joblib.dump(clf, path)
    print(f"Saved classifier to {path}")


def load_model(path):
    clf = joblib.load(path)
    return clf


def classify_mode_file(clf, path):
    """
    Classify a single new NOVA output file.
    Returns:
        prob_good: probability it's a good (physical) mode
        label:     "good" or "bad"
    """
    mode, omega, gamma_d, ntor = load_mode_from_nova(path)
    extra_info = {
        "omega": omega,
        "gamma_d": gamma_d,
        "ntor": ntor,
        "path": path,
    }
    X = compute_features_for_mode(mode, extra_info).reshape(1, -1)
    prob_good = clf.predict_proba(X)[0, 1]
    label = "good" if prob_good >= 0.5 else "bad"
    return prob_good, label

# ========================
# 5. MAIN ENTRY POINT
# ========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or apply an ML classifier to sort NOVA TAE modes (good vs bad)."
    )
    parser.add_argument("--train_csv", type=str, help="CSV file with 'path,label' for training.")
    parser.add_argument("--model_out", type=str, default="nova_mode_classifier.joblib",
                        help="Path to save trained model.")
    parser.add_argument("--classify", type=str, default=None,
                        help="If set, classify this NOVA output file using an existing model.")
    parser.add_argument("--model_in", type=str, default=None,
                        help="Path to existing model to use for --classify.")

    args = parser.parse_args()

    if args.train_csv:
        # TRAINING MODE
        modes, y, paths, extra_infos = load_labeled_modes(args.train_csv)
        X = build_feature_matrix(modes, extra_infos)
        feature_names = [
            "mean_amp",
            "std_amp",
            "rad_loc",
            "rad_width",
            "max_to_mean",
            "max_to_median",
            "mean_abs_d1_mode",
            "max_abs_d1_abs",
            "mean_abs_d2_mode",
            "max_abs_d2_abs",
            "std_per_m_max",
            "max_per_m_mean",
            "std_per_m_mean",
            #"n_spikes",
            "spikes_per_m",
            "frac_spikes",
        ]
        #feature_names += ["omega", "gamma_d"]
        feature_names += ["omega", "gamma_d", "ntor"]
        #feature_names += ["has_intersection", "delta2_eff", "S", "W_star"]
        feature_names += ["r_star", "delta2_eff", "S", "W_star"]

        clf = train_classifier(X, y)
        print_feature_importance(clf, feature_names)

        # Optional extra evaluation
        clf = evaluate_classifier(clf, X, y)

        save_model(clf, args.model_out)

        joblib.dump(
            {
                "model": clf,
                "X_train": X,
                "y_train": y,
                "feature_names": feature_names,
            },
            "nova_mode_classifier_bundle.joblib"
        )

    if args.classify:
        if args.model_in is None:
            raise ValueError("You must specify --model_in to classify a new mode.")
        clf = load_model(args.model_in)
        prob_good, label = classify_mode_file(clf, args.classify)
        print(f"File: {args.classify}")
        print(f"Predicted label: {label}  (P(good) = {prob_good:.3f})")
        print('=======================================================')
        print('')


