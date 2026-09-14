"""Native energy concentration and local smoothness for the envelope exception.

Regions use half the global W maximum; quadrature includes partial cells at
their interpolated endpoints. High-pass stencils must fit inside each window.
"""

import math
import numpy as np

from continuum_noise import native_high_pass_energy


def integrate(r, y, left, right):
    """Integrate a piecewise-linear native density, including partial cells."""
    x = np.r_[left, r[(r > left) & (r < right)], right]
    return float(np.trapezoid(np.interp(x, r, y), x))


def measure_envelope_footprint(mode, window_dr=0.05, width_max_grid=2.0, peak_r_max=0.5):
    """Return the calibrated F_spikes and worst-window Q_local diagnostics.

    An unresolved window cannot grant the exception; its status is explicit.
    The grid is neither smoothed nor resampled. All harmonics contribute.
    """
    for name, value in (("window_dr", window_dr), ("width_max_grid", width_max_grid), ("peak_r_max", peak_r_max)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < window_dr <= 1 or peak_r_max > 1:
        raise ValueError("window_dr must be in (0, 1] and peak_r_max in [0, 1]")
    A = np.asarray(mode, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 2 or not np.isfinite(A).all():
        raise ValueError("footprint requires finite [harmonic, radius] arrays with nr>=2")
    if A.shape[1] < 3:
        return dict(status="WINDOW_UNRESOLVED", nr=A.shape[1], spikes_fraction=None, local_hf_fraction=None)
    raw, hf_power, weights, top2 = native_high_pass_energy(A)
    W = np.sum(raw, axis=0) / weights
    r = np.linspace(0, 1, W.size)
    dr = r[1] - r[0]
    peak = int(np.argmax(W))
    total = float(np.sum(raw))
    if total <= 0:
        return dict(status="ZERO_MODE_ENERGY", nr=W.size, spikes_fraction=None, local_hf_fraction=None)
    top2_energy = float(np.sum(raw[top2]))
    level = 0.5 * W[peak]
    above = W >= level
    starts = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    components = []
    main_fraction = None
    for start in starts:
        end = int(start)
        while end + 1 < W.size and above[end + 1]:
            end += 1
        left = float(r[start]) if start == 0 else float(
            r[start - 1] + dr * (level - W[start - 1]) / (W[start] - W[start - 1]))
        right = float(r[end]) if end == W.size - 1 else float(
            r[end] + dr * (W[end] - level) / (W[end] - W[end + 1]))
        energy = integrate(r, W, left, right)
        center = int(start + np.argmax(W[start:end + 1]))
        if start <= peak <= end:
            main_fraction = energy / total
        if (right - left) / dr <= width_max_grid + 1e-12 and r[center] <= peak_r_max:
            components.append(dict(left=left, right=right, center=center, energy=energy))
    # Regions are disjoint, so overlapping peak windows cannot double-count F.
    spike_energy = sum(c["energy"] for c in components)
    half = window_dr / 2
    windows = []
    for center in sorted({peak} | {c["center"] for c in components}):
        window = np.flatnonzero((r >= r[center] - half - 1e-12) & (r <= r[center] + half + 1e-12))
        if len(window) < 3:
            return dict(status="WINDOW_UNRESOLVED", nr=W.size,
                        spikes_fraction=spike_energy / total, local_hf_fraction=None)
        first, last = int(window[0]), int(window[-1])
        hf_energy = float(np.sum(hf_power[:, first + 1:last]) * dr)
        window_energy = integrate(r, W, float(r[first]), float(r[last]))
        windows.append(dict(center=center, first=first, last=last, ratio=hf_energy / window_energy))
    main_window = next(w for w in windows if w["center"] == peak)
    worst_window = max(windows, key=lambda w: w["ratio"])
    return dict(
        status="MEASURED", nr=W.size, energy_peak_r=float(r[peak]), main_spike_fraction=main_fraction,
        spikes_fraction=spike_energy / total, spikes_top2_ratio=spike_energy / top2_energy,
        n_narrow_halfmax_components=len(components),
        energy_effective_length=total**2 / float(np.sum(weights * W**2)),
        local_hf_fraction=worst_window["ratio"],
        main_peak_local_hf_fraction=main_window["ratio"],
        worst_window_peak_r=float(r[worst_window["center"]]),
        peak_window_r_start=float(r[main_window["first"]]), peak_window_r_end=float(r[main_window["last"]]),
        global_hf_top2_ratio=float(np.sum(hf_power * weights)) / top2_energy,
        top2_energy_fraction=top2_energy / total,
        components=components, windows=windows,
    )


def footprint_exception_qualifies(features, spikes_fraction_max, local_hf_fraction_max):
    """Equality or an unavailable measurement cannot grant the exception."""
    return bool(spikes_fraction_max is not None and features["status"] == "MEASURED"
                and features["spikes_fraction"] < spikes_fraction_max
                and features["local_hf_fraction"] < local_hf_fraction_max)
