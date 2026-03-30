#!usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from nova_mode_loader import load_mode_from_nova        # returns (mode, omega, gamma_d, ntor)
from cont_features import load_datcon_for_mode, continuum_scalars
from path_utils import resolve_mode_csv_path


def read_mode_csv(csv_path: str):
    """
    Accepts CSV with either:
      - path,label
      - path
    Returns:
      paths:  list[str]
      labels: list[Optional[str]]  # "good"/"bad"/None
    """
    paths, labels = [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            p = resolve_mode_csv_path(row[0])
            if not p or p.lower().startswith("path"):
                continue

            lab = None
            if len(row) >= 2:
                s = row[1].strip().lower()
                if s in ("good", "bad", "g", "b"):
                    lab = "good" if s in ("good", "g") else "bad"

            paths.append(p)
            labels.append(lab)
    return paths, labels


def topk_harmonics(mode: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k harmonics by max(|mode|) over r."""
    strength = np.max(np.abs(mode), axis=1)
    order = np.argsort(strength)[::-1]
    k = min(k, mode.shape[0])
    return order[:k]


def plot_mode_panel(ax, mode: np.ndarray, r: np.ndarray, kind: str, topk: int, use_abs: bool):
    """
    kind: 'lines' or 'contour'
    """
    ax.clear()
    nhar, nr = mode.shape
    y = np.abs(mode) if use_abs else mode

    if kind == "contour":
        # contour expects 2D; x=r, y=m_index
        m_idx = np.arange(nhar)
        # Use imshow for speed/clarity
        ax.imshow(
            y,
            aspect="auto",
            origin="lower",
            extent=[r[0], r[-1], m_idx[0], m_idx[-1]],
        )
        ax.set_xlabel("r (normalized)")
        ax.set_ylabel("m index")
        ax.set_title("Mode structure (m,r)" + (" |abs|" if use_abs else ""))
        return

    # lines
    idx = topk_harmonics(mode, topk)
    for mi in idx:
        ax.plot(r, y[mi, :], linewidth=1.0, alpha=0.9)
    ax.set_xlabel("r (normalized)")
    ax.set_ylabel("|xi_m(r)|" if use_abs else "xi_m(r)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Top-{len(idx)} harmonic radial profiles")


def plot_continuum_panel(ax, mode_path: str, n_r: int, r: np.ndarray, omega: float, try_rstar: bool = True):
    ax.clear()

    try:
        mode, _, _, _ = load_mode_from_nova(mode_path)
        low2, high2, *_ = load_datcon_for_mode(mode_path, n_r=n_r)
    except Exception as e:
        ax.text(
            0.02, 0.5,
            f"No datcon (or read error)\n{Path(mode_path).parent}\n{e}",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
        )
        ax.set_axis_off()
        return

    low = np.sqrt(np.clip(low2, 0.0, np.inf))
    high = np.sqrt(np.clip(high2, 0.0, np.inf))

    ax.plot(r, low, linewidth=1.2, label="sqrt(low2)")
    ax.plot(r, high, linewidth=1.2, label="sqrt(high2)")
    ax.axhline(omega, linestyle="--", linewidth=1.2, label="mode omega")

    # r_star (optional)
    r_star = None
    if try_rstar:
        try:
            cont = continuum_scalars(mode, float(omega), low2, high2, r=r)
            r_star = float(cont["r_star"])
        except Exception:
            r_star = None

    if r_star is None:
        pass
    else:
        ax.axvline(r_star, linestyle="--", linewidth=1.0, label="r*")

    ax.set_xlabel("r (normalized)")
    ax.set_ylabel("frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Continuum band + mode frequency")


def plot_m_spectrum(ax, mode: np.ndarray):
    ax.clear()
    per_m = np.max(np.abs(mode), axis=1)
    m_idx = np.arange(mode.shape[0])
    ax.plot(m_idx, per_m, marker="o", linewidth=1.0, markersize=3)
    ax.set_xlabel("m index")
    ax.set_ylabel("max_r |xi_m|")
    ax.grid(True, alpha=0.3)
    ax.set_title("m-spectrum (max over r)")


def main():
    ap = argparse.ArgumentParser(description="Browse NOVA mode files listed in a CSV (no labeling).")
    ap.add_argument("csv", help="CSV containing mode paths (optionally path,label)")
    ap.add_argument("--start", type=int, default=0, help="start index")
    ap.add_argument("--topk", type=int, default=44, help="number of strongest m harmonics to plot in line mode")
    ap.add_argument("--abs", action="store_true", help="plot abs(mode) instead of signed/real mode")
    ap.add_argument("--contour", action="store_true", help="use (m,r) contour panel instead of harmonic lines")
    ap.add_argument("--no_mspec", action="store_true", help="disable m-spectrum panel")
    ap.add_argument("--no_cont", action="store_true", help="disable continuum panel")
    args = ap.parse_args()

    paths, labels = read_mode_csv(args.csv)
    if not paths:
        raise SystemExit(f"No paths found in {args.csv}")

    idx = max(0, min(args.start, len(paths) - 1))

    # Layout: mode panel on top, then continuum, then m-spectrum (optional)
    nrows = 1
    show_cont = not args.no_cont
    show_mspec = not args.no_mspec
    if show_cont:
        nrows += 1
    if show_mspec:
        nrows += 1

    fig = plt.figure(figsize=(10, 3.2 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, 1, height_ratios=[3] + ([1.2] if show_cont else []) + ([1.2] if show_mspec else []))

    row = 0
    ax_mode = fig.add_subplot(gs[row, 0]); row += 1
    ax_cont = fig.add_subplot(gs[row, 0]) if show_cont else None
    if show_cont: row += 1
    ax_mspec = fig.add_subplot(gs[row, 0]) if show_mspec else None

    help_text = (
        "Keys:  n/→ next | p/← prev | h(ome) first | e(nd) last | "
        "c toggle contour/lines | a toggle abs | q quit"
    )

    state = {
        "idx": idx,
        "use_abs": bool(args.abs),
        "contour": bool(args.contour),
    }

    def update():
        i = state["idx"]
        path = paths[i]
        lab = labels[i]
        lab_str = "unknown" if lab is None else lab
        mode, omega, gamma_d, ntor = load_mode_from_nova(path)
        nhar, nr = mode.shape
        r = np.linspace(0.0, 1.0, nr)

        title = (
            f"[{i+1}/{len(paths)}]  label={lab_str.upper()}  {path}\n"
            f"nhar={nhar} nr={nr}  ntor={ntor}  omega={omega:.6g}  gamma_d={gamma_d:.3g}"
        )
        fig.suptitle(title + "\n" + help_text, fontsize=10)

        plot_mode_panel(ax_mode, mode, r, kind=("contour" if state["contour"] else "lines"),
                        topk=args.topk, use_abs=state["use_abs"])

        if ax_cont is not None:
            # Plot continuum band and omega; also try to plot r*:
            # First do the base plot
            ax_cont.clear()
            try:
                low2, high2, *_ = load_datcon_for_mode(path, n_r=nr)
                low = np.sqrt(np.clip(low2, 0.0, np.inf))
                high = np.sqrt(np.clip(high2, 0.0, np.inf))

                ax_cont.plot(r, low, linewidth=1.2, label="sqrt(low2)")
                ax_cont.plot(r, high, linewidth=1.2, label="sqrt(high2)")
                ax_cont.axhline(float(omega), linestyle="--", linewidth=1.2, label="mode omega")

                # r* EXACTLY like label_modes_fast.py
                r_star = None
                try:
                    cont = continuum_scalars(mode, float(omega), low2, high2, r=r)
                    r_star = float(cont["r_star"])
                except Exception:
                    r_star = None

                if r_star is not None and np.isfinite(r_star):
                    ax_cont.axvline(r_star, linestyle="--", linewidth=1.0, label="r*")

                ax_cont.set_xlabel("r (normalized)")
                ax_cont.set_ylabel("frequency")
                ax_cont.grid(True, alpha=0.3)
                ax_cont.legend(loc="best", fontsize=8)
                ax_cont.set_title("Continuum band + mode frequency")

            except Exception as e:
                ax_cont.text(
                    0.02, 0.5,
                    f"No datcon (or read error)\n{Path(path).parent}\n{e}",
                    transform=ax_cont.transAxes,
                    va="center",
                    ha="left",
                    fontsize=9,
                )
                ax_cont.set_axis_off()

        if ax_mspec is not None:
            plot_m_spectrum(ax_mspec, mode)

        fig.canvas.draw_idle()

    def on_key(event):
        k = event.key
        if k in ("q", "escape"):
            plt.close(fig)
            return
        if k in ("n", "right"):
            state["idx"] = min(len(paths) - 1, state["idx"] + 1)
            update()
            return
        if k in ("p", "left"):
            state["idx"] = max(0, state["idx"] - 1)
            update()
            return
        if k == "h":
            state["idx"] = 0
            update()
            return
        if k == "e":
            state["idx"] = len(paths) - 1
            update()
            return
        if k == "c":
            state["contour"] = not state["contour"]
            update()
            return
        if k == "a":
            state["use_abs"] = not state["use_abs"]
            update()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


if __name__ == "__main__":
    main()
