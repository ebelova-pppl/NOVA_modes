import numpy as np
import os
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from mode_features import (
    compute_features_for_mode,
    get_feature_names,
    get_feature_schema_version,
)
from mode_csv import read_mode_csv_entries


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
    csv_path: CSV with path plus good/bad labels and an optional header row.
    Label/validity values must be 'good' or 'bad'.
    Returns:
        modes: list of 2D numpy arrays
        y:     list of integer labels (0 = bad, 1 = good)
        paths: list of filepaths
    """
    modes = []
    labels = []
    paths = []
    extra_infos = []

    for file_path, raw_label in read_mode_csv_entries(csv_path):
        label_str = (raw_label or "").strip().lower()
        mode, omega, gamma_d, ntor = load_mode_from_nova(file_path)
        extra_info = {
           "omega": omega,
           "gamma_d": gamma_d,
           "ntor": ntor,
           "path": file_path,
        }

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

def build_feature_matrix(
    modes,
    extra_infos,
    include_crossing_features=False,
    r_shear0=0.2,
):
    """
    modes: list of 2D arrays
    Returns: X 2D array (n_samples, n_features)
    """
    feats = [
        compute_features_for_mode(
            m,
            extra_info,
            include_crossing_features=include_crossing_features,
            r_shear0=r_shear0,
        )
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
    if len(feature_names) != importances.size:
        raise ValueError(
            f"Feature-name count {len(feature_names)} does not match "
            f"RF importance count {importances.size}"
        )
    
    print("\n=== Feature Importances ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"{name:25s}  {imp:.4f}")


def evaluate_classifier(clf, X, y, paths):
    """
    Optional: split out a test set and print metrics.
    """

    X_train, X_test, y_train, y_test, _paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.10, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    # Check FP/FN cases
    wrong = np.where(y_pred != y_test)[0]
    print("\nMisclassified modes:")
    print("Path,  true_label,  pred_label,  p_good")
    for i in wrong:
        true_lab = "good" if y_test[i] == 1 else "bad"
        pred_lab = "good" if y_pred[i] == 1 else "bad"
        print(f"{paths_test[i]}, {true_lab}, {pred_lab}, {probs[i]:.3f}")

    print("\nConfusion matrix (test):")
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


def attach_feature_metadata(
    clf,
    feature_names,
    include_crossing_features=False,
    r_shear0=0.2,
):
    """Attach lightweight schema metadata while keeping a plain sklearn pipeline."""
    clf.nova_feature_names_ = list(feature_names)
    clf.nova_feature_schema_version_ = get_feature_schema_version(
        include_crossing_features
    )
    clf.nova_include_crossing_features_ = bool(include_crossing_features)
    clf.nova_r_shear0_ = float(r_shear0)
    return clf


def validate_model_feature_schema(
    clf,
    feature_names,
    include_crossing_features=False,
    r_shear0=0.2,
):
    expected_count = getattr(clf, "n_features_in_", None)
    if expected_count is not None and expected_count != len(feature_names):
        if expected_count == 28 and len(feature_names) == 22:
            action = "Add --crossing-features for this experimental model."
        elif expected_count == 22 and len(feature_names) == 28:
            action = "Remove --crossing-features for the production model."
        else:
            action = "Select the feature schema used to train this model."
        raise ValueError(
            f"RF model expects {expected_count} features, but the selected schema "
            f"builds {len(feature_names)}. {action}"
        )

    saved_names = getattr(clf, "nova_feature_names_", None)
    if saved_names is not None and list(saved_names) != list(feature_names):
        raise ValueError(
            "Selected feature names do not match the schema stored in the RF model."
        )

    saved_crossing = getattr(clf, "nova_include_crossing_features_", None)
    if saved_crossing is not None and bool(saved_crossing) != bool(
        include_crossing_features
    ):
        raise ValueError(
            "The RF model crossing-feature setting does not match the CLI. "
            "Add or remove --crossing-features as appropriate."
        )

    saved_r_shear0 = getattr(clf, "nova_r_shear0_", None)
    if (
        include_crossing_features
        and saved_r_shear0 is not None
        and not np.isclose(float(saved_r_shear0), float(r_shear0))
    ):
        raise ValueError(
            f"The RF model was trained with r_shear0={saved_r_shear0}, "
            f"but classification requested {r_shear0}."
        )


def classify_mode_file(
    clf,
    path,
    include_crossing_features=False,
    r_shear0=0.2,
):
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
    feature_names = get_feature_names(include_crossing_features)
    validate_model_feature_schema(
        clf,
        feature_names,
        include_crossing_features=include_crossing_features,
        r_shear0=r_shear0,
    )
    X = compute_features_for_mode(
        mode,
        extra_info,
        include_crossing_features=include_crossing_features,
        r_shear0=r_shear0,
    ).reshape(1, -1)
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
    parser.add_argument(
        "--train_csv",
        type=str,
        help="Training CSV with paths plus good/bad labels, with or without a header row.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default=None,
        help=(
            "Path to save the trained model. Defaults to "
            "nova_mode_classifier.joblib for the production schema or "
            "nova_mode_classifier_crossing.joblib with --crossing-features."
        ),
    )
    parser.add_argument(
        "--bundle_out",
        type=str,
        default=None,
        help=(
            "Path to save training arrays and schema metadata. By default, "
            "'_bundle.joblib' is appended to the model filename stem."
        ),
    )
    parser.add_argument("--classify", type=str, default=None,
                        help="If set, classify this NOVA output file using an existing model.")
    parser.add_argument("--model_in", type=str, default=None,
                        help="Path to existing model to use for --classify.")
    parser.add_argument(
        "--crossing-features",
        action="store_true",
        help=(
            "Opt in to the experimental 28-feature RF schema. The production "
            "schema already includes W_star_max; this adds the other six "
            "continuum-boundary-crossing features."
        ),
    )
    parser.add_argument(
        "--r_shear0",
        type=float,
        default=0.2,
        help="Radial offset for the experimental shear proxy (default: 0.2).",
    )

    args = parser.parse_args()
    if not np.isfinite(args.r_shear0):
        raise ValueError(f"--r_shear0 must be finite, got {args.r_shear0}")

    model_out = args.model_out
    bundle_out = args.bundle_out
    if args.train_csv:
        if model_out is None:
            model_out = (
                "nova_mode_classifier_crossing.joblib"
                if args.crossing_features
                else "nova_mode_classifier.joblib"
            )
        model_out_path = Path(model_out)
        active_model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "nova_mode_classifier.joblib"
        )
        if (
            args.crossing_features
            and model_out_path.expanduser().resolve() == active_model_path.resolve()
        ):
            raise ValueError(
                "Refusing to overwrite the active legacy RF checkpoint with an "
                "experimental crossing-feature model. Choose a different --model_out."
            )
        if bundle_out is None:
            bundle_out = str(
                model_out_path.with_name(f"{model_out_path.stem}_bundle.joblib")
            )

    if args.train_csv:
        # TRAINING MODE
        modes, y, paths, extra_infos = load_labeled_modes(args.train_csv)
        X = build_feature_matrix(
            modes,
            extra_infos,
            include_crossing_features=args.crossing_features,
            r_shear0=args.r_shear0,
        )
        feature_names = get_feature_names(args.crossing_features)
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Feature-name count {len(feature_names)} does not match "
                f"feature matrix width {X.shape[1]}"
            )

        clf = train_classifier(X, y)
        print_feature_importance(clf, feature_names)

        # Optional extra evaluation
        clf = evaluate_classifier(clf, X, y, paths)

        clf = attach_feature_metadata(
            clf,
            feature_names,
            include_crossing_features=args.crossing_features,
            r_shear0=args.r_shear0,
        )

        save_model(clf, model_out)

        joblib.dump(
            {
                "model": clf,
                "X_train": X,
                "y_train": y,
                "feature_names": feature_names,
                "feature_schema_version": get_feature_schema_version(
                    args.crossing_features
                ),
                "include_crossing_features": bool(args.crossing_features),
                "r_shear0": float(args.r_shear0),
            },
            bundle_out,
        )
        print(f"Saved classifier bundle to {bundle_out}")

    if args.classify:
        if args.model_in is None:
            raise ValueError("You must specify --model_in to classify a new mode.")
        clf = load_model(args.model_in)
        prob_good, label = classify_mode_file(
            clf,
            args.classify,
            include_crossing_features=args.crossing_features,
            r_shear0=args.r_shear0,
        )
        print(f"File: {args.classify}")
        print(f"Predicted label: {label}  (P(good) = {prob_good:.3f})")
        print('=======================================================')
        print('')
