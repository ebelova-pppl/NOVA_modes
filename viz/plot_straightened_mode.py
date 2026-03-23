#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mode_transform import straighten_mode_window
from nova_mode_loader import load_mode_from_nova

"""
script to plot 2d (r,m) mode structure + straightened mode (r, m_c +/- delta_m)
"""
# -------------------------
# Plotting helpers
# -------------------------
def plot_heatmap(ax, A, x, y, title="", xlabel="", ylabel="", vlim=None):
    """
    A: (Ny,Nx)
    x: (Nx,), y:(Ny,)
    """
    if vlim is None:
        vmax = np.max(np.abs(A)) + 1e-12
        vlim = (-vmax, vmax)
    im = ax.imshow(
        A,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=vlim[0], vmax=vlim[1],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

def main():
    ap = argparse.ArgumentParser(description="Plot NOVA mode and its straightened representation.")
    ap.add_argument("mode_path", help="Path to egn* binary file")
    ap.add_argument("--M", type=int, default=12, help="Half-window in m for straightened view (H=2M+1)")
    ap.add_argument("--med_k", type=int, default=7, help="Median filter window (odd) for mc(r)")
    ap.add_argument("--max_step", type=float, default=2.0, help="Slew limiter max step in mc per radial index")
    ap.add_argument("--nr_override", type=int, default=None, help="(debug) override nr grid for plotting [ignored normally]")
    ap.add_argument("--save", type=str, default=None, help="Save figure path (png/pdf). If omitted, show.")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    mode, omega, gamma_d, ntor = load_mode_from_nova(args.mode_path)
    nhar, nr = mode.shape

    # radial coordinate: normalized index [0,1]
    r = np.linspace(0.0, 1.0, nr)

    X, mc, m0 = straighten_mode_window(
        mode, 
        M=args.M, 
        median_k=args.med_k, 
        max_step=args.max_step
    )
    H = X.shape[0]
    dm = np.arange(-args.M, args.M + 1)  # delta-m axis

    # color limits based on original mode (maxabs=1 typically)
    vmax = float(np.max(np.abs(mode)) + 1e-12)
    vlim = (-vmax, vmax)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (1) original heatmap with ridge
    im0 = plot_heatmap(
        axs[0, 0], mode, r, np.arange(nhar),
        title="Original mode (m vs r)",
        xlabel="r (normalized index)", ylabel="m index",
        vlim=vlim
    )
    axs[0, 0].plot(r, mc, "w-", linewidth=0.5, alpha=0.9, label=r"$m_c(r)$")
    axs[0, 0].legend(loc="upper right", fontsize=9)

    # (2) straightened heatmap
    im1 = plot_heatmap(
        axs[0, 1], X, r, dm,
        title=f"Straightened window (Δm vs r), M={args.M}",
        xlabel="r (normalized index)", ylabel="Δm (around ridge)",
        vlim=vlim
    )

    # (3) ridge curve only
    axs[1, 0].plot(r, mc, "-", linewidth=2)
    axs[1, 0].set_title(r"Ridge $m_c(r)$")
    axs[1, 0].set_xlabel("r (normalized index)")
    axs[1, 0].set_ylabel("m index")
    axs[1, 0].grid(True, alpha=0.3)

    # (4) example straightened lineouts (a few Δm rows)
    #rows = [0, args.M//4, args.M//2, args.M*3//4, args.M, 
         #args.M + args.M//4, 
         #args.M + args.M//2, 
         #args.M + args.M*3//4, 
         #2*args.M] if args.M >= 8 else list(range(H))
    rows = [0, args.M-2, args.M-1, args.M, args.M+1, args.M+2,
         2*args.M] if args.M >= 4 else list(range(H))
    for ridx in rows:
        axs[1, 1].plot(r, X[ridx, :], linewidth=1.0, alpha=0.9, label=f"Δm={dm[ridx]}")
    axs[1, 1].set_title("Straightened lineouts (selected Δm)")
    axs[1, 1].set_xlabel("r (normalized index)")
    axs[1, 1].set_ylabel("mode value")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"{args.mode_path}\nntor={ntor:g}  omega={omega:.6g}  gamma_d={gamma_d:.6g}  nhar={nhar}  nr={nr}",
        fontsize=10
    )

    # colorbars
    fig.colorbar(im0, ax=axs[0, 0], shrink=0.85)
    fig.colorbar(im1, ax=axs[0, 1], shrink=0.85)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi)
        print(f"Saved: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
