#!/usr/bin/env python3
import os
import glob
import csv
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mode_csv import read_mode_csv_entries
from cont_features import load_datcon_for_mode, continuum_scalars


plt.ion()

# =========================
# Loader (same format)
# =========================
def load_mode_from_nova(path: str):
    """
    Returns:
      mode (nhar, nr) real array (xi_psi assumed)
      omega (float)
      gamma_d (float)
      ntor (int)
    File format:
      f1[0]   = omega
      f1[-3]  = nr
      f1[-2]  = gamma_d
      f1[-1]  = ntor
      f1[1:-3] reshapes to (3, nhar, nr): (xi_psi, delta_p, xi_surf)
    """
    f1 = np.fromfile(path)
    omega = float(f1[0])
    nr = int(f1[-3])
    gamma_d = float(f1[-2])
    ntor = int(round(float(f1[-1])))

    nhar = int((f1.size - 4) / (3 * nr))
    f11 = f1[1:-3].reshape(3, nhar, nr)
    mode = f11[0, :, :]  # xi_psi

    return mode, omega, gamma_d, ntor

# =========================
# get continuum info
# =========================
def get_r_star_for_mode(mode_path: str, mode: np.ndarray, omega: float):
    """
    Returns r_star in [0,1] or None if continuum unavailable.
    """
    n_r = mode.shape[1]
    r = np.linspace(0.0, 1.0, n_r)

    try:
        low2, high2, *_ = load_datcon_for_mode(mode_path, n_r=n_r)
        cont = continuum_scalars(mode, omega, low2, high2, r=r)
        r_star = float(cont["r_star"])
        return r_star
    except Exception:
        return None

# =========================
# CSV helpers (resume-friendly)
# =========================
def read_labels(csv_path: str) -> Dict[str, str]:
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    for path, label in read_mode_csv_entries(csv_path, resolve_paths=False):
        if label is None:
            continue
        labels[path] = label.strip().lower()
    return labels


def append_label(csv_path: str, path: str, label: str):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([path, label])


# =========================
# Plotting
# =========================
def plot_all_harmonics_1d(ax, mode: np.ndarray, r: np.ndarray, use_abs: bool,
                          max_lines: Optional[int] = None,
                          r_star: Optional[float] = None):
    """
    Overlays xi_m(r) for all m on one axis.
    If max_lines is set, plots only the strongest harmonics (by max|xi_m|).
    """
    ax.clear()
    nhar, nr = mode.shape
    y = np.abs(mode) if use_abs else mode

    # choose which m lines to plot
    strength = np.max(np.abs(mode), axis=1)  # shape (nhar,)
    order = np.argsort(strength)[::-1]  # strongest first
    if max_lines is not None:
        order = order[:max_lines]

    for k, mi in enumerate(order):
        ax.plot(r, y[mi, :], linewidth=1.0, alpha=0.9)

    # Mark nearest continuum location if provided
    if r_star is not None and np.isfinite(r_star):
        ax.axvline(
            r_star,
            color="k",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=5
        )
        ax.text(
            r_star, -0.01,
            r"$R^\ast$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10
        )

    ax.set_xlabel("r (normalized index)")
    ax.set_ylabel("|xi_m(r)|" if use_abs else "xi_m(r)")
    ax.grid(True, alpha=0.3)


def plot_m_spectrum(ax, mode: np.ndarray):
    """Simple diagnostic: max_r |xi_m| vs m-index."""
    ax.clear()
    strength = np.max(np.abs(mode), axis=1)
    m = np.arange(len(strength))
    ax.plot(m, strength, marker="o", linewidth=1.0)
    ax.set_xlabel("m index")
    ax.set_ylabel("max_r |xi_m|")
    ax.grid(True, alpha=0.3)

def plot_continuum_panel(ax, r: np.ndarray, omega: float,
                         low2: np.ndarray, high2: np.ndarray,
                         title: str = "Continuum"):
    """
    Plots sqrt(low2) and sqrt(high2) vs r, plus a horizontal line at omega.
    low2/high2 can contain NaNs on skipped edge points.
    """
    ax.clear()

    # Safe sqrt: ignore NaNs; clamp tiny negatives if any numerical noise
    low = np.sqrt(np.maximum(low2, 0.0))
    high = np.sqrt(np.maximum(high2, 0.0))

    ax.plot(r, low, linewidth=1.2)
    ax.plot(r, high, linewidth=1.2)
    ax.axhline(omega, linestyle="--", linewidth=1.2)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\omega$")

    ax.grid(True, alpha=0.3)

def load_rf_classifier(rf_model_path: str):
    try:
        import joblib
    except ImportError as exc:
        print(f"WARNING: joblib is not available; RF disabled ({exc}).")
        return None

    model_path = Path(rf_model_path).expanduser()
    if not model_path.exists():
        print(f"WARNING: RF model not found: {model_path}; RF disabled.")
        return None

    try:
        clf = joblib.load(model_path)
    except Exception as exc:
        print(f"WARNING: failed to load RF model {model_path}: {exc}; RF disabled.")
        return None

    print(f"Loaded RF model: {model_path}")
    return clf


def rf_opinion(clf, mode, omega, gamma_d, ntor, file_path):
    from mode_features import compute_features_for_mode

    X = compute_features_for_mode(
        mode,
        extra_info={"omega": omega, "gamma_d": gamma_d, "ntor": ntor, "path": file_path}
        #extra_info={"omega": omega, "gamma_d": gamma_d, "path": file_path}
    ).reshape(1, -1)
    p_good = float(clf.predict_proba(X)[0, 1])
    lab = "good" if p_good >= 0.5 else "bad"
    return p_good, lab


# =========================
# Main interactive labeling
# =========================

@dataclass
class Config:
    data_dir: Optional[str] = None
    pattern: str = "egn*"
    out_csv: str = "mode_labels.csv"
    mode_list: Optional[str] = None
    sort: bool = True
    use_rf: bool = True
    rf_model: str = "nova_mode_classifier.joblib"

    # plotting choices
    use_abs: bool = False          # if True, plot |mode| instead of signed
    max_lines: Optional[int] = 30  # None = plot all m; else plot strongest N m-lines only
    r_is_uniform_0_1: bool = True  # r = linspace(0,1,nr)
    show_m_spectrum: bool = True   # add a small 2nd panel


def resolve_mode_dir(mode_dir: str, data_dir: Optional[str]) -> Path:
    raw_path = Path(mode_dir).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    if data_dir:
        return (Path(data_dir).expanduser() / raw_path).resolve()

    raise ValueError(
        f"Relative mode directory '{mode_dir}' requires --data_dir or $NOVA_DATA."
    )


def path_key(path: str, data_dir: Optional[str] = None) -> str:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute() or data_dir is None:
        return str(raw_path.resolve())
    return str((Path(data_dir).expanduser() / raw_path).resolve())


def path_match_keys(path: str, data_dir: Optional[str] = None) -> set[str]:
    """
    Build exact and suffix keys for matching mode-list rows to scanned files.

    Split outputs may contain absolute paths from a different filesystem root
    than the current run, especially across Perlmutter/Flux. The
    shot/N/file suffix is stable across those roots.
    """
    keys = {path_key(path, data_dir)}
    raw_path = Path(path).expanduser()
    resolved_path = Path(next(iter(keys)))

    for candidate in (raw_path, resolved_path):
        parts = candidate.parts
        if len(parts) >= 3:
            keys.add(str(Path(*parts[-3:])))
        if len(parts) >= 2:
            keys.add(str(Path(*parts[-2:])))

    return keys


def read_mode_list_keys(csv_path: str, data_dir: Optional[str]) -> set[str]:
    keys: set[str] = set()
    for path, _label in read_mode_csv_entries(csv_path, data_root=data_dir):
        keys.update(path_match_keys(path))
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Fast interactive labeling of NOVA modes"
    )
    parser.add_argument(
        "mode_dir",
        help=(
            "Directory containing mode files. Absolute paths are used directly; "
            "relative paths are resolved under --data_dir or $NOVA_DATA "
            "(e.g. 'nstx_120113/N5')."
        )
    )
    parser.add_argument(
        "--csv_out",
        default="mode_labels.csv",
        dest="csv_out",
        help="Output CSV for labels (default: mode_labels.csv)"
    )
    parser.add_argument(
        "--csv",
        dest="csv_out",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode-list",
        help=(
            "Optional CSV list of candidate modes to label, such as "
            "training_labels/tae_like.csv or training_labels/eae_like.csv. "
            "Only files in mode_dir whose resolved path or shot/N/file suffix "
            "appears in this list are shown."
        )
    )
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("NOVA_DATA"),
        help="Data directory for relative mode_dir paths (default: $NOVA_DATA)"
    )
    parser.add_argument(
        "--pattern",
        default="egn*",
        help="Glob pattern for mode files inside mode_dir (default: egn*)"
    )
    parser.add_argument(
        "--rf-model",
        default="nova_mode_classifier.joblib",
        help="Random Forest model path used for optional guidance (default: nova_mode_classifier.joblib)"
    )
    parser.add_argument(
        "--no-rf",
        action="store_true",
        help="Disable Random Forest evaluation/display."
    )
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        pattern=args.pattern,
        out_csv=args.csv_out,
        mode_list=args.mode_list,
        use_rf=not args.no_rf,
        rf_model=args.rf_model,
    )

    try:
        mode_dir = resolve_mode_dir(args.mode_dir, cfg.data_dir)
    except ValueError as exc:
        parser.error(str(exc))

    clf = load_rf_classifier(cfg.rf_model) if cfg.use_rf else None
    if cfg.use_rf and clf is None:
        print("Continuing without RF guidance.")
    elif not cfg.use_rf:
        print("RF guidance disabled by --no-rf.")

    input_glob = str(mode_dir / cfg.pattern)
    files = glob.glob(input_glob)

    if cfg.sort:
        files = sorted(files)

    mode_list_keys = None
    files_before_mode_list_filter = None
    if cfg.mode_list:
        try:
            mode_list_keys = read_mode_list_keys(cfg.mode_list, cfg.data_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(f"Could not read --mode-list {cfg.mode_list!r}: {exc}")
        files_before_mode_list_filter = len(files)
        files = [p for p in files if path_match_keys(p, cfg.data_dir) & mode_list_keys]
        print(
            f"Mode list filter: {Path(cfg.mode_list).expanduser()} "
            f"({len(mode_list_keys)} entries)"
        )
        print(
            f"Matched {len(files)} of {files_before_mode_list_filter} files in mode_dir "
            "against the mode list."
        )

    labels = read_labels(cfg.out_csv)
    labeled_keys = set()
    for p in labels:
        labeled_keys.update(path_match_keys(p, cfg.data_dir))
    files_to_do = [p for p in files if not (path_match_keys(p, cfg.data_dir) & labeled_keys)]
    labeled_in_scope = len(files) - len(files_to_do)

    print(f"Mode directory: {mode_dir}")
    print(f"File pattern:    {cfg.pattern}")
    print(f"Input glob:      {input_glob}")
    if cfg.mode_list:
        print(f"Mode list:       {Path(cfg.mode_list).expanduser()}")
    print(f"Output CSV:      {Path(cfg.out_csv).expanduser()}")
    print(
        f"Found {len(files)} files; already labeled {labeled_in_scope}; "
        f"remaining {len(files_to_do)}"
    )
    print("Keys: g=good, b=bad, s=skip, u=undo, q=quit")

    if not files:
        if cfg.mode_list and files_before_mode_list_filter:
            print(
                "No mode files matched --mode-list. Check that the list and "
                "mode_dir refer to the same shot/N directory."
            )
        else:
            print("No mode files found. Check mode_dir, --data_dir, and --pattern.")
        return

    if not files_to_do:
        print("Nothing to label.")
        return

    history: List[Tuple[str, str]] = []

    if cfg.show_m_spectrum:
        fig, (ax1, axC, ax2) = plt.subplots(
            3, 1, figsize=(9, 9),
            height_ratios=[3, 1, 1],
            constrained_layout=True
        )
    else:
        fig, (ax1, axC) = plt.subplots(
            2, 1, figsize=(9, 7),
            height_ratios=[3, 1],
            constrained_layout=True
        )
        ax2 = None


    i = 0
    while i < len(files_to_do):
        path = files_to_do[i]
        base = os.path.basename(path)

        try:
            mode, omega, gamma_d, ntor = load_mode_from_nova(path)
        except Exception as e:
            print(f"\nERROR reading {path}: {e}\nMarking as skip.")
            append_label(cfg.out_csv, path, "skip")
            labels[path] = "skip"
            i += 1
            plt.pause(0.01)
            continue

        nhar, nr = mode.shape
        r = np.linspace(0.0, 1.0, nr) if cfg.r_is_uniform_0_1 else np.arange(nr)

        rf_text = ""
        if clf is not None:
            try:
                p_rf, rf_lab = rf_opinion(clf, mode, omega, gamma_d, ntor, path)
                rf_text = f"  RF:{rf_lab} (P={p_rf:.3f})"
            except Exception as exc:
                print(f"WARNING: RF evaluation failed for {path}: {exc}")

        title = (f"{base}  n={ntor}  omega={omega:.4g}  g_d={gamma_d:.3g}  "
                 f"{rf_text}  "
                 #f"shape=({nhar},{nr})   "
                 f"[g/b/s, u=undo, q=quit]")


        #plot_all_harmonics_1d(ax1, mode, r, use_abs=cfg.use_abs, max_lines=cfg.max_lines)
        r_star = get_r_star_for_mode(path, mode, omega)  # returns None if no datcon

        plot_all_harmonics_1d(
            ax1,
            mode,
            r,
            use_abs=True,
            max_lines=20,
            r_star=r_star
        )

        ax1.set_title(title)

        # Continuum mini-panel
        n_r = mode.shape[1]
        r = np.linspace(0.0, 1.0, n_r)
        try:
            low2, high2, *_ = load_datcon_for_mode(path, n_r=n_r)
            low2 = low2.copy()
            high2 = high2.copy()
            low2[low2 > 999] = np.nan
            high2[high2 > 999] = np.nan
            plot_continuum_panel(axC, r, omega, low2, high2, title="Alfvén continuum")
        except Exception:
            axC.clear()
            axC.text(0.5, 0.5, "no datcon", ha="center", va="center", transform=axC.transAxes)
            axC.set_axis_off()

        if ax2 is not None:
            plot_m_spectrum(ax2, mode)

        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.01)

        key = input("\nLabel (g=good, b=bad, s=skip, u=undo, q=quit): ").strip().lower()[:1]

        if key == "q":
            print("Quitting. Progress saved.")
            break

        if key == "u":
            if not history:
                print("Nothing to undo.")
                continue
            last_path, last_label = history.pop()
            print(f"Undo: {os.path.basename(last_path)} was {last_label}")
            if last_path in labels:
                del labels[last_path]
            # Put it back right before current item
            files_to_do.insert(i, last_path)
            continue

        if key not in ("g", "b", "s"):
            print("Unrecognized key. Use g/b/s/u/q.")
            continue

        label = {"g": "good", "b": "bad", "s": "skip"}[key]
        append_label(cfg.out_csv, path, label)
        labels[path] = label
        history.append((path, label))
        print(f"Labeled: {base} -> {label}")
        i += 1
        plt.pause(0.01)

        fig.canvas.draw_idle()
        plt.pause(0.01)


    plt.close(fig)
    plt.ioff()

    # de-duplicate (latest label wins) into a clean CSV
    clean_csv = os.path.splitext(cfg.out_csv)[0] + "_clean.csv"
    final_labels = read_labels(cfg.out_csv)
    with open(clean_csv, "w", newline="") as f:
        w = csv.writer(f)
        for p in sorted(final_labels.keys()):
            w.writerow([p, final_labels[p]])
    print(f"\nWrote de-duplicated labels to: {clean_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
