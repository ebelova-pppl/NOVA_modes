#!/usr/bin/env python3
"""Render label-free, model-free NOVA TAE-like mode diagnostics."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _blind_review_io import read_dict_csv, stable_mode_key


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from cont_features import (
        continuum_crossing_records,
        continuum_extremum_features,
        load_datcon_for_mode,
    )
    from nova_mode_loader import load_mode_from_nova
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The diagnostic renderer requires the project scientific Python "
        "environment with NumPy, Matplotlib, and SciPy."
    ) from exc


MANIFEST_FIELDS = {
    "blind_id",
    "path",
    "shot",
    "n",
    "signed_delta",
    "fraction_below_upper2",
    "gap_region",
    "error",
}
DATCON_INVALID_SENTINEL_MIN = 999.0


def resolve_mode_path(raw_path: str, data_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if data_root is None:
        raise ValueError(
            f"relative mode path requires --data-root or $NOVA_DATA: {raw_path}"
        )
    return data_root / path


def energy_fraction(
    energy: np.ndarray, r: np.ndarray, center: float, half_width: float
) -> float:
    total = float(np.trapezoid(energy, r))
    if total <= 0.0:
        return 0.0
    left = max(float(r[0]), center - half_width)
    right = min(float(r[-1]), center + half_width)
    if right <= left:
        return 0.0
    inner_r = r[(r > left) & (r < right)]
    window_r = np.concatenate(([left], inner_r, [right]))
    window_w = np.interp(window_r, r, energy)
    return float(np.trapezoid(window_w, window_r) / total)


def safe_filename(blind_id: str, mode_path: Path) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", mode_path.name)
    return f"{blind_id}_{token}.png"


def render_one(
    row: dict[str, str],
    mode_path: Path,
    output: Path,
    topk: int,
    dpi: int,
) -> None:
    mode, omega, gamma_d, ntor = load_mode_from_nova(str(mode_path))
    mode = np.asarray(mode, dtype=float)
    nhar, nr = mode.shape
    r = np.linspace(0.0, 1.0, nr)
    energy = np.sum(np.abs(mode) ** 2, axis=0)
    peak_energy = float(np.max(energy))
    normalized_energy = energy / (peak_energy + 1e-14)
    r_peak = float(r[int(np.argmax(energy))])

    strength = np.max(np.abs(mode), axis=1)
    harmonic_order = np.argsort(strength)[::-1][: min(topk, nhar)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_mode, ax_energy, ax_cont, ax_spectrum = axes.flat

    for harmonic in harmonic_order:
        ax_mode.plot(r, mode[harmonic], linewidth=0.9, alpha=0.85)
    ax_mode.axhline(0.0, color="0.5", linewidth=0.6)
    ax_mode.axvline(r_peak, color="k", linestyle="--", linewidth=1.0)
    ax_mode.set(
        xlabel="r",
        ylabel=r"$\xi_m(r)$",
        title=f"Signed harmonics ({len(harmonic_order)}/{nhar}); r_peak={r_peak:.3f}",
    )
    ax_mode.grid(alpha=0.25)

    ax_energy.plot(r, normalized_energy, color="tab:blue", linewidth=1.5)
    ax_energy.axvline(r_peak, color="k", linestyle="--", linewidth=1.0)
    axis_fraction = energy_fraction(energy, r, 0.015, 0.015)
    ax_energy.set(
        xlabel="r",
        ylabel=r"$W(r)/W_{max}$",
        title=f"Radial energy; fraction at r<=0.03: {100.0 * axis_fraction:.3f}%",
    )
    ax_energy.set_ylim(bottom=0.0)
    ax_energy.grid(alpha=0.25)

    continuum_note = "continuum unavailable"
    crossing_summary: list[str] = []
    try:
        low2, high2, *_ = load_datcon_for_mode(str(mode_path), n_r=nr)
        low2 = np.asarray(low2, dtype=float).copy()
        high2 = np.asarray(high2, dtype=float).copy()
        low2[low2 > DATCON_INVALID_SENTINEL_MIN] = np.nan
        high2[high2 > DATCON_INVALID_SENTINEL_MIN] = np.nan
        low = np.sqrt(np.where(low2 >= 0.0, low2, np.nan))
        high = np.sqrt(np.where(high2 >= 0.0, high2, np.nan))

        ax_cont.plot(r, low, label="lower", linewidth=1.3)
        ax_cont.plot(r, high, label="upper", linewidth=1.3)
        ax_cont.axhline(omega, color="k", linestyle="--", label="mode omega")
        ax_cont.axvline(r_peak, color="0.4", linestyle=":", linewidth=1.0)

        crossings = continuum_crossing_records(
            mode, float(omega), low2, high2, r=r
        )
        for crossing in crossings:
            radius = float(crossing["r_cross"])
            local_fraction = energy_fraction(energy, r, radius, 0.03)
            ax_cont.axvline(
                radius,
                color="tab:red" if crossing["boundary"] == "high" else "tab:purple",
                linewidth=0.8,
                alpha=0.6,
            )
            crossing_summary.append(
                f"{crossing['boundary']}@{radius:.3f}:"
                f"Wpk={float(crossing['W_peak']):.3g},"
                f"E±.03={100.0 * local_fraction:.3g}%"
            )

        extremum = continuum_extremum_features(
            mode, float(omega), low2, high2, r=r
        )
        continuum_note = (
            f"crossings={len(crossings)}; "
            f"nearest inner extremum: delta_r_peak={float(extremum['ext_dr']):.3g}, "
            f"gap-side delta_f/omega={float(extremum['ext_df_gap']):.3g}, "
            f"local energy={100.0 * float(extremum['ext_energy_frac']):.3g}%"
        )
        ax_cont.legend(fontsize=8, loc="best")
        ax_cont.set(
            xlabel="r",
            ylabel="absolute frequency",
            title="TAE continuum and true sign-change crossings",
        )
        ax_cont.grid(alpha=0.25)
    except Exception as exc:
        ax_cont.text(
            0.03,
            0.5,
            f"Continuum diagnostic unavailable:\n{type(exc).__name__}: {exc}",
            transform=ax_cont.transAxes,
            va="center",
        )
        ax_cont.set_axis_off()

    ax_spectrum.plot(
        np.arange(nhar), strength, marker="o", markersize=2.5, linewidth=0.9
    )
    ax_spectrum.set(
        xlabel="stored harmonic index (physical m is not stored in egn)",
        ylabel=r"$\max_r |\xi_m|$",
        title="Stored poloidal-harmonic spectrum",
    )
    ax_spectrum.grid(alpha=0.25)

    blind_id = row.get("blind_id", "")
    fig.suptitle(
        f"{blind_id}  {stable_mode_key(str(mode_path))}\n"
        f"n={ntor}, omega={omega:.7g}, gamma_d={gamma_d:.3g}; {continuum_note}",
        fontsize=11,
    )
    if crossing_summary:
        shown = crossing_summary[:6]
        suffix = (
            f" | +{len(crossing_summary) - len(shown)} more"
            if len(crossing_summary) > len(shown)
            else ""
        )
        fig.text(0.01, 0.005, " | ".join(shown) + suffix, fontsize=7)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render signed harmonics, W(r), absolute continuum, true crossings, "
            "and m-spectrum without loading any classifier or labels."
        )
    )
    parser.add_argument("manifest", help="Prepared label-free blind manifest")
    parser.add_argument("output_dir", help="Directory for one PNG per mode")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Resolve relative mode paths under this directory",
    )
    parser.add_argument(
        "--topk", type=int, default=44, help="Strongest signed harmonics to plot"
    )
    parser.add_argument("--start", type=int, default=0, help="Zero-based start row")
    parser.add_argument("--limit", type=int, help="Maximum number of modes to render")
    parser.add_argument("--dpi", type=int, default=140, help="PNG resolution")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing diagnostic PNGs"
    )
    args = parser.parse_args()

    if args.topk <= 0 or args.start < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--topk and --limit must be positive; --start must be nonnegative")

    fields, rows = read_dict_csv(args.manifest)
    unsupported = set(fields) - MANIFEST_FIELDS
    if unsupported or "path" not in fields:
        raise ValueError(
            "manifest is not the prepared label-free schema; unsupported "
            f"columns: {', '.join(sorted(unsupported))}"
        )

    selected = rows[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("no manifest rows selected")

    data_root_value = args.data_root
    if data_root_value is None:
        import os

        data_root_value = os.environ.get("NOVA_DATA")
    data_root = Path(data_root_value).expanduser() if data_root_value else None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for row in selected:
        mode_path = resolve_mode_path(row["path"], data_root)
        output = output_dir / safe_filename(row.get("blind_id", "mode"), mode_path)
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite existing diagnostic: {output}")
        outputs.append(output)

    for index, (row, output) in enumerate(zip(selected, outputs), start=1):
        mode_path = resolve_mode_path(row["path"], data_root)
        render_one(row, mode_path, output, args.topk, args.dpi)
        print(f"[{index}/{len(selected)}] {output}")

    print(f"Rendered {len(outputs)} model-free diagnostics in {output_dir}")


if __name__ == "__main__":
    main()
