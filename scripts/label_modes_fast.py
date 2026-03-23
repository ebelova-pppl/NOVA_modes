#!/usr/bin/env python3
import os
import glob
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import joblib
from mode_features import compute_features_for_mode
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
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            labels[row[0]] = row[1].strip().lower()
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

def rf_opinion(clf, mode, omega, gamma_d, ntor, file_path):
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
    base_dir: str = "/global/cfs/cdirs/m314/nova"     # root directory
    out_csv: str = "mode_labels.csv"
    sort: bool = True

    # plotting choices
    use_abs: bool = False          # if True, plot |mode| instead of signed
    max_lines: Optional[int] = 30  # None = plot all m; else plot strongest N m-lines only
    r_is_uniform_0_1: bool = True  # r = linspace(0,1,nr)
    show_m_spectrum: bool = True   # add a small 2nd panel

def main():
    
    rf_model_path = "nova_mode_classifier.joblib"     # Load RF classifier
    clf = joblib.load(rf_model_path)
    print(f"Loaded RF model: {rf_model_path}")

    parser = argparse.ArgumentParser(
        description="Fast interactive labeling of NOVA modes"
    )
    parser.add_argument(
        "shot",
        help="device shot number (subdirectory name, e.g. 'D3D/202020')"
    )
    parser.add_argument(
        "--csv",
        default="mode_labels.csv",
        help="Output CSV for labels (default: mode_labels.csv)"
    )
    args = parser.parse_args()

    cfg = Config(out_csv=args.csv)

    input_glob = os.path.join(cfg.base_dir, args.shot, "egn*")
    files = glob.glob(input_glob)

    if cfg.sort:
        files = sorted(files)

    labels = read_labels(cfg.out_csv)
    files_to_do = [p for p in files if p not in labels]

    print(f"Found {len(files)} files; already labeled {len(labels)}; remaining {len(files_to_do)}")
    print("Keys: g=good, b=bad, s=skip, u=undo, q=quit")

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

        p_rf, rf_lab = rf_opinion(clf, mode, omega, gamma_d, ntor, path)  # RF input

        title = (f"{base}  n={ntor}  omega={omega:.4g}  g_d={gamma_d:.3g}  "
                 f"RF:{rf_lab} (P={p_rf:.3f})  "
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
            low2[low2 > 999] = 300
            high2[high2 > 999] = 300
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

