"""Plot the two diagnostic details using the shared NOVA loaders.

Example: python audits/nstxu_204202_five_mode_review_20261025/plot_details.py \
    --shot-dir /path/to/nstxu_204202 --out-dir /path/to/figures
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _repo_bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()
from cont_features import load_datcon_for_mode
from nova_mode_loader import load_mode_from_nova


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-dir", type=Path, required=True, help="Original shot directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="PNG/PDF output directory")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, key in enumerate(("N4/egn04w.8263E+01", "N10/egn10w.1026E+02")):
        path = args.shot_dir / key
        mode, omega, _, _ = load_mode_from_nova(str(path))
        radius = np.linspace(0, 1, mode.shape[1])
        energy = np.sum(mode**2, axis=0)
        energy /= energy.max()
        selected = ((radius >= 0.43) & (radius <= 0.56)) if column == 0 else radius <= 0.065
        harmonics = (8, 9) if column == 0 else (14, 15)
        ax = axes[0, column]
        for harmonic in harmonics:
            ax.plot(radius[selected], mode[harmonic, selected], ".-", linewidth=1,
                    label=f"stored harmonic {harmonic}")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set(title=key, xlabel="Normalized radius r", ylabel="Signed amplitude")
        ax.legend()
        ax.grid(alpha=0.25)
        ax = axes[1, column]
        if column == 0:
            lower2, _, *_ = load_datcon_for_mode(str(path), n_r=len(radius))
            ax.plot(radius[selected], np.sqrt(lower2[selected]), ".-", label="Lower TAE continuum")
            ax.axhline(omega, color="k", linestyle="--", label="Mode frequency")
            ax.set(ylabel="Frequency (NOVA units)", ylim=(2.3, 3.1))
            ax.legend()
        else:
            ax.plot(radius[selected], energy[selected], ".-")
            ax.set(ylabel="W / max(W)", title="Axis spike: small contribution to total energy")
        ax.set_xlabel("Normalized radius r")
        ax.grid(alpha=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(args.out_dir / f"n4_peak_n10_axis_detail.{extension}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
