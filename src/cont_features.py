import os
import numpy as np
import re
import warnings
from scipy.signal import find_peaks

_WARNED_DATCON_DIRS = set()
DATCON_INVALID_SENTINEL_MIN = 999.0
DATCON_TAIL_SPIKE_FACTOR = 3.0
DATCON_TAIL_SPIKE_ABS_MIN = 50.0
DATCON_TAIL_LOOKBACK = 4
CROSSING_FEATURE_DEFAULTS = {
    "n_cross": 0,
    "r_star_max": 0.0,
    "W_star_max": 0.0,
    "W_star_sum": 0.0,
    "r_star_high_shear": 0.0,
    "W_star_high_shear": 0.0,
    "W_star_high_shear_sum": 0.0,
}
EXTREMUM_FEATURE_DEFAULTS = {
    "ext_dr": 1.0,
    "ext_df_gap": 1.0,
    "ext_energy_frac": 0.0,
}
EXTREMUM_R_MIN = 0.03
EXTREMUM_R_MAX = 0.40
EXTREMUM_DR_SCALE = 0.02
EXTREMUM_DF_SCALE = 0.03
EXTREMUM_ENERGY_HALF_WIDTH = 0.03
EXTREMUM_SMOOTHING_KERNEL = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0

def warn_once_per_dir(mode_path: str, msg: str):
    d = os.path.dirname(os.path.abspath(mode_path))
    if d not in _WARNED_DATCON_DIRS:
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        _WARNED_DATCON_DIRS.add(d)

def get_ntor_from_path(mode_path: str) -> int:
    """
    Extract ntor from .../N5/... or .../N10/...
    """
    m = re.search(r"/N(\d+)/", mode_path)
    if not m:
        raise ValueError(f"Cannot infer ntor from path: {mode_path}")
    return int(m.group(1))


def _mask_datcon_invalid(values: np.ndarray) -> np.ndarray:
    """
    Treat the legacy datcon sentinel (~1000.000) as missing data.
    """
    arr = values.astype(float, copy=True)
    arr[arr > DATCON_INVALID_SENTINEL_MIN] = np.nan
    return arr


def _trim_trailing_datcon_spikes(values: np.ndarray) -> np.ndarray:
    """
    Trim extra bogus tail points immediately before the masked datcon sentinel.

    Some legacy datcon files contain one more unphysical trailing value before
    the explicit 1000.000 sentinel region. We only trim at the tail, and only
    when the last finite point is both:
    - several times larger than the recent finite tail values, and
    - large in absolute terms, to avoid overreacting near zero.
    """
    arr = values.astype(float, copy=True)

    while True:
        finite_idx = np.flatnonzero(np.isfinite(arr))
        if finite_idx.size < DATCON_TAIL_LOOKBACK + 1:
            break

        last = int(finite_idx[-1])
        if last == arr.size - 1:
            break

        prev_idx = finite_idx[-(DATCON_TAIL_LOOKBACK + 1):-1]
        prev = arr[prev_idx]
        prev = prev[np.isfinite(prev)]
        if prev.size < 3:
            break

        prev_max = float(np.max(prev))
        last_val = float(arr[last])

        if (
            prev_max > 0.0
            and last_val > DATCON_TAIL_SPIKE_FACTOR * prev_max
            and last_val > DATCON_TAIL_SPIKE_ABS_MIN
        ):
            arr[last] = np.nan
            continue

        break

    return arr

def load_datcon_for_mode(mode_path: str, n_r: int):
    """
    Loads datcon from the same N* directory as the mode file.

    Returns:
        low2_full  (n_r,) float, NaN where undefined
        high2_full (n_r,) float, NaN where undefined
        i1, i2     int (1-based indices from file)
    """
    #d = os.path.dirname(os.path.abspath(mode_path)) # this works for
    #datcon_path = os.path.join(d, "datcon")         # Perlmutter
    ntor = get_ntor_from_path(mode_path)
    datcon_name = f"datcon{ntor}"                    # this is needed on Flux
    datcon_path = os.path.join(os.path.dirname(mode_path), datcon_name)

    if not os.path.exists(datcon_path):
        raise FileNotFoundError(f"Missing {datcon_path}")
        raise FileNotFoundError(f"datcon not found next to mode: {datcon_path}")

    with open(datcon_path, "r") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise ValueError(f"Bad datcon header in {datcon_path}: {header}")
        i1 = int(header[0])
        i2 = int(header[1])

        data = np.loadtxt(f)  # remaining lines: 2 columns
        if data.ndim == 1:
            data = data.reshape(1, -1)

    expected = i2 - i1 + 1
    if data.shape[0] != expected or data.shape[1] < 2:
        raise ValueError(
            f"datcon shape mismatch in {datcon_path}: got {data.shape}, "
            f"expected ({expected}, 2+)"
        )

    low2 = _trim_trailing_datcon_spikes(_mask_datcon_invalid(data[:, 0]))
    high2 = _trim_trailing_datcon_spikes(_mask_datcon_invalid(data[:, 1]))

    # Build full arrays on the mode's radial grid
    low2_full = np.full(n_r, np.nan, dtype=float)
    high2_full = np.full(n_r, np.nan, dtype=float)

    # Convert 1-based inclusive [i1,i2] -> 0-based slice [i1-1 : i2]
    low2_full[i1-1:i2] = low2
    high2_full[i1-1:i2] = high2

    return low2_full, high2_full, i1, i2


def band_distance(omega2: float, low2: np.ndarray, high2: np.ndarray):
    """
    Returns dist2(r): 0 if omega2 inside [low2, high2], else distance to nearest boundary in omega^2 units.
    NaNs in low2/high2 propagate to NaN dist2.
    """
    dist = np.full_like(low2, np.nan, dtype=float)
    ok = np.isfinite(low2) & np.isfinite(high2)

    l = low2[ok]
    h = high2[ok]

    below = omega2 < l
    above = omega2 > h
    inside = (~below) & (~above)

    d = np.zeros_like(l)
    d[below] = (l[below] - omega2)
    d[above] = (omega2 - h[above])
    d[inside] = 0.0

    dist[ok] = d
    return dist


def _validate_crossing_inputs(mode, omega, low2_full, high2_full, r):
    mode = np.asarray(mode)
    if mode.ndim != 2:
        raise ValueError(f"mode must be 2D (n_m, n_r), got shape {mode.shape}")
    if mode.shape[0] < 1:
        raise ValueError("mode must contain at least one poloidal harmonic")
    if mode.shape[1] < 1:
        raise ValueError("mode must contain at least one radial point")
    if not np.all(np.isfinite(mode)):
        raise ValueError("mode contains non-finite values")

    n_r = mode.shape[1]
    low2 = np.asarray(low2_full, dtype=float)
    high2 = np.asarray(high2_full, dtype=float)
    if low2.ndim != 1 or high2.ndim != 1:
        raise ValueError(
            f"continuum arrays must be 1D, got low2={low2.shape}, high2={high2.shape}"
        )
    if low2.shape != (n_r,) or high2.shape != (n_r,):
        raise ValueError(
            "continuum arrays must match the mode radial dimension: "
            f"n_r={n_r}, low2={low2.shape}, high2={high2.shape}"
        )

    if r is None:
        radial_grid = np.linspace(0.0, 1.0, n_r)
    else:
        radial_grid = np.asarray(r, dtype=float)
        if radial_grid.ndim != 1 or radial_grid.shape != (n_r,):
            raise ValueError(
                f"r must be 1D with length {n_r}, got shape {radial_grid.shape}"
            )
    if not np.all(np.isfinite(radial_grid)):
        raise ValueError("r contains non-finite values")
    if radial_grid.size > 1 and np.any(np.diff(radial_grid) <= 0.0):
        raise ValueError("r must be strictly increasing")

    omega_value = float(omega)
    if not np.isfinite(omega_value):
        raise ValueError(f"omega must be finite, got {omega}")

    return mode, omega_value, low2, high2, radial_grid


def _boundary_crossing_records(
    boundary_type,
    boundary2,
    omega2,
    valid,
    r,
    w_peak,
    r_shear0,
):
    """
    Return crossings for one boundary without bridging invalid radial gaps.

    Exact-zero runs are represented by one crossing at the run midpoint.
    """
    f = omega2 - boundary2
    records = []
    n_r = r.size
    i = 0

    while i < n_r:
        if not valid[i]:
            i += 1
            continue

        block_end = i
        while block_end + 1 < n_r and valid[block_end + 1]:
            block_end += 1

        while i <= block_end:
            if f[i] == 0.0:
                zero_end = i
                while zero_end + 1 <= block_end and f[zero_end + 1] == 0.0:
                    zero_end += 1
                r_cross = 0.5 * (float(r[i]) + float(r[zero_end]))
                W_cross = float(np.interp(r_cross, r, w_peak))
                shear_weighted = W_cross * max(r_cross - r_shear0, 0.0) ** 2
                records.append(
                    {
                        "boundary": boundary_type,
                        "r_cross": r_cross,
                        "W_peak": W_cross,
                        "shear_weighted": float(shear_weighted),
                    }
                )
                i = zero_end + 1
                continue

            if i < block_end and f[i + 1] != 0.0 and f[i] * f[i + 1] < 0.0:
                fraction = float(-f[i] / (f[i + 1] - f[i]))
                r_cross = float(r[i] + fraction * (r[i + 1] - r[i]))
                W_cross = float(w_peak[i] + fraction * (w_peak[i + 1] - w_peak[i]))
                shear_weighted = W_cross * max(r_cross - r_shear0, 0.0) ** 2
                records.append(
                    {
                        "boundary": boundary_type,
                        "r_cross": r_cross,
                        "W_peak": W_cross,
                        "shear_weighted": float(shear_weighted),
                    }
                )

            i += 1

    return records


def continuum_crossing_records(
    mode,
    omega,
    low2_full,
    high2_full,
    r=None,
    r_shear0=0.2,
):
    """
    Return diagnostic records for lower/upper continuum-boundary crossings.

    Each record contains boundary type, interpolated radius, peak-normalized
    radial mode energy, and the shear-weighted value. Lower and upper boundary
    crossings are counted separately.
    """
    mode, omega, low2, high2, r = _validate_crossing_inputs(
        mode, omega, low2_full, high2_full, r
    )
    r_shear0 = float(r_shear0)
    if not np.isfinite(r_shear0):
        raise ValueError(f"r_shear0 must be finite, got {r_shear0}")

    radial_energy = np.sum(np.abs(mode) ** 2, axis=0)
    peak_energy = float(np.max(radial_energy))
    w_peak = radial_energy / (peak_energy + 1e-14)

    valid = np.isfinite(low2) & np.isfinite(high2)
    if not np.any(valid):
        return []

    omega2 = omega**2
    records = []
    records.extend(
        _boundary_crossing_records(
            "low", low2, omega2, valid, r, w_peak, r_shear0
        )
    )
    records.extend(
        _boundary_crossing_records(
            "high", high2, omega2, valid, r, w_peak, r_shear0
        )
    )
    return records


def continuum_crossing_features(
    mode,
    omega,
    low2_full,
    high2_full,
    r=None,
    r_shear0=0.2,
):
    """
    Compute peak-energy and shear-weighted continuum-crossing RF features.

    Ties for either maximum are resolved in favor of the largest crossing
    radius so feature values remain deterministic.
    """
    records = continuum_crossing_records(
        mode,
        omega,
        low2_full,
        high2_full,
        r=r,
        r_shear0=r_shear0,
    )
    if not records:
        return dict(CROSSING_FEATURE_DEFAULTS)

    amp_max = max(records, key=lambda item: (item["W_peak"], item["r_cross"]))
    shear_max = max(
        records, key=lambda item: (item["shear_weighted"], item["r_cross"])
    )

    return {
        "n_cross": len(records),
        "r_star_max": float(amp_max["r_cross"]),
        "W_star_max": float(amp_max["W_peak"]),
        "W_star_sum": float(sum(item["W_peak"] for item in records)),
        "r_star_high_shear": float(shear_max["r_cross"]),
        "W_star_high_shear": float(shear_max["shear_weighted"]),
        "W_star_high_shear_sum": float(
            sum(item["shear_weighted"] for item in records)
        ),
    }


def _finite_blocks(mask):
    """Yield inclusive-exclusive slices for contiguous True blocks."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return

    start = int(indices[0])
    previous = start
    for index in indices[1:]:
        index = int(index)
        if index != previous + 1:
            yield slice(start, previous + 1)
            start = index
        previous = index
    yield slice(start, previous + 1)


def _smooth_finite_blocks(values, kernel=EXTREMUM_SMOOTHING_KERNEL):
    """Smooth finite continuum blocks without bridging missing-data gaps."""
    values = np.asarray(values, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 1 or kernel.size % 2 != 1 or kernel.size < 1:
        raise ValueError("extremum smoothing kernel must have positive odd length")
    if not np.isfinite(kernel).all() or float(np.sum(kernel)) <= 0.0:
        raise ValueError("extremum smoothing kernel must be finite with positive sum")
    kernel = kernel / np.sum(kernel)

    smoothed = np.full_like(values, np.nan)
    half_width = kernel.size // 2
    for block in _finite_blocks(np.isfinite(values)):
        block_values = values[block]
        if block_values.size == 1:
            smoothed[block] = block_values
            continue
        padded = np.pad(block_values, half_width, mode="edge")
        smoothed[block] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _extremum_candidates(boundary, smoothed, r, omega, kind, r_min, r_max):
    """Return upper-minimum or lower-maximum candidates in the inner region."""
    search_mask = (
        np.isfinite(boundary)
        & np.isfinite(smoothed)
        & (r >= r_min)
        & (r <= r_max)
    )
    candidates = []
    for block in _finite_blocks(search_mask):
        block_indices = np.arange(r.size)[block]
        # Endpoints of a valid/search block are deliberately excluded by
        # find_peaks, avoiding datcon-boundary artifacts.
        signal = smoothed[block]
        if kind == "upper_min":
            signal = -signal
        peaks, properties = find_peaks(signal, prominence=0.0)
        for local_index, prominence in zip(peaks, properties["prominences"]):
            index = int(block_indices[local_index])
            frequency = float(boundary[index])
            if kind == "upper_min":
                df_gap = (frequency - omega) / omega
            else:
                df_gap = (omega - frequency) / omega
            candidates.append(
                {
                    "kind": kind,
                    "index": index,
                    "r_ext": float(r[index]),
                    "frequency": frequency,
                    "df_gap": float(df_gap),
                    "prom_rel": float(prominence / abs(omega)),
                }
            )
    return candidates


def _energy_fraction_in_window(radial_energy, r, center, half_width):
    """Return the fraction of integrated radial energy near ``center``."""
    radial_energy = np.asarray(radial_energy, dtype=float)
    if radial_energy.size == 1:
        return 1.0

    total = float(
        np.sum(
            0.5 * (radial_energy[:-1] + radial_energy[1:]) * np.diff(r)
        )
    )
    if not np.isfinite(total) or total <= 0.0:
        return 0.0

    left = max(float(r[0]), center - half_width)
    right = min(float(r[-1]), center + half_width)
    if right <= left:
        return 0.0

    interior = (r > left) & (r < right)
    r_window = np.concatenate(([left], r[interior], [right]))
    energy_window = np.interp(r_window, r, radial_energy)
    local = float(
        np.sum(
            0.5
            * (energy_window[:-1] + energy_window[1:])
            * np.diff(r_window)
        )
    )
    return float(np.clip(local / total, 0.0, 1.0))


def continuum_extremum_features(
    mode,
    omega,
    low2_full,
    high2_full,
    r=None,
    r_min=EXTREMUM_R_MIN,
    r_max=EXTREMUM_R_MAX,
    dr_scale=EXTREMUM_DR_SCALE,
    df_scale=EXTREMUM_DF_SCALE,
    energy_half_width=EXTREMUM_ENERGY_HALF_WIDTH,
):
    """Measure joint mode alignment with an inner continuum-gap extremum.

    The established radial mode-energy notation is
    ``W(r) = sum_m |xi_m(r)|^2``. The mode location used here is the maximum
    of W(r), not its centroid. Candidate extrema are minima of the physical
    upper boundary ``sqrt(high2)`` and maxima of the physical lower boundary
    ``sqrt(low2)``. The candidate with the smallest normalized joint radial
    and frequency mismatch is returned.

    ``ext_df_gap`` uses one sign convention for both boundary types: positive
    values place the mode frequency on the local gap side of the extremum,
    zero is tangency, and negative values place it on the continuum side.
    ``ext_energy_frac`` is the fraction of integrated W(r) within a fixed
    radial half-width of the matched extremum.
    """
    mode, omega, low2, high2, r = _validate_crossing_inputs(
        mode, omega, low2_full, high2_full, r
    )
    r_min = float(r_min)
    r_max = float(r_max)
    dr_scale = float(dr_scale)
    df_scale = float(df_scale)
    energy_half_width = float(energy_half_width)
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min >= r_max:
        raise ValueError(f"invalid extremum radial interval: [{r_min}, {r_max}]")
    if not np.isfinite(dr_scale) or dr_scale <= 0.0:
        raise ValueError(f"extremum dr_scale must be positive, got {dr_scale}")
    if not np.isfinite(df_scale) or df_scale <= 0.0:
        raise ValueError(f"extremum df_scale must be positive, got {df_scale}")
    if not np.isfinite(energy_half_width) or energy_half_width <= 0.0:
        raise ValueError(
            "extremum energy_half_width must be positive, "
            f"got {energy_half_width}"
        )
    if omega <= 0.0:
        raise ValueError("omega must be positive for relative extremum features")

    radial_energy = np.sum(np.abs(mode) ** 2, axis=0)
    if float(np.max(radial_energy)) <= 0.0:
        return dict(EXTREMUM_FEATURE_DEFAULTS)
    r_peak = float(r[int(np.argmax(radial_energy))])
    low = np.sqrt(np.where(low2 >= 0.0, low2, np.nan))
    high = np.sqrt(np.where(high2 >= 0.0, high2, np.nan))
    low_smoothed = _smooth_finite_blocks(low)
    high_smoothed = _smooth_finite_blocks(high)

    candidates = _extremum_candidates(
        high, high_smoothed, r, omega, "upper_min", r_min, r_max
    )
    candidates.extend(
        _extremum_candidates(
            low, low_smoothed, r, omega, "lower_max", r_min, r_max
        )
    )
    if not candidates:
        return dict(EXTREMUM_FEATURE_DEFAULTS)

    for candidate in candidates:
        candidate["dr"] = abs(r_peak - candidate["r_ext"])
        candidate["match_score"] = (
            (candidate["dr"] / dr_scale) ** 2
            + (abs(candidate["df_gap"]) / df_scale) ** 2
        )

    matched = min(
        candidates,
        key=lambda item: (
            item["match_score"],
            -item["prom_rel"],
            item["r_ext"],
            item["kind"],
        ),
    )
    return {
        "ext_dr": float(matched["dr"]),
        "ext_df_gap": float(matched["df_gap"]),
        "ext_energy_frac": _energy_fraction_in_window(
            radial_energy,
            r,
            matched["r_ext"],
            energy_half_width,
        ),
    }


def continuum_scalars(
    mode,
    omega,
    low2_full,
    high2_full,
    r=None,
    alpha=1.0,
    r_star_energy_tie=False,
):
    """
    Compute continuum-aware scalars from mode + datcon band.
    Uses omega^2 comparison because datcon stores omega_A^2.
    """
    n_m, n_r = mode.shape
    if r is None:
        r = np.linspace(0.0, 1.0, n_r)

    w = np.sum(np.abs(mode)**2, axis=0)  # (n_r,)
    wsum = np.sum(w) + 1e-14

    r0 = float(np.sum(r * w) / wsum)
    rw = float(np.sqrt(np.sum(((r - r0)**2) * w) / wsum) + 1e-12)

    omega2 = float(omega)**2
    dist2 = band_distance(omega2, low2_full, high2_full)  # (n_r,), NaN outside valid band region

    ok = np.isfinite(dist2)
    if not np.any(ok):
        # Should not happen if datcon is valid; return safe defaults
        return {
            "has_intersection": 0.0,
            "delta2_min": 1e30,
            "delta2_eff": 1e30,
            "r_star": r0,
            "S": 1e30,
            "W_star": 0.0,
            "r0": r0,
            "rw": rw,
        }

    # Always-defined distances
    delta2_min = float(np.nanmin(dist2))
    delta2_eff = float(np.nansum(dist2 * w) / wsum)

    # Closest gap-distance radius. The legacy behavior takes the first tied
    # minimum; the opt-in rule selects the tied point with the largest W(r).
    if r_star_energy_tie:
        tied = np.flatnonzero(np.isfinite(dist2) & (dist2 == delta2_min))
        i_star = int(max(tied, key=lambda index: (w[index], r[index])))
    else:
        i_star = int(np.nanargmin(dist2))
    r_star = float(r[i_star])

    # Does it intersect the band anywhere?
    has_intersection = float(delta2_min == 0.0)

    # Distance in widths
    S = float(abs(r_star - r0) / rw)

    # Mode weight near r_star within window alpha*rw
    L = alpha * rw
    mask = np.abs(r - r_star) <= L
    W_star = float(np.sum(w[mask]) / wsum)

    return {
        "has_intersection": has_intersection,
        "delta2_min": delta2_min,
        "delta2_eff": delta2_eff,
        "r_star": r_star,
        "S": S,
        "W_star": W_star,
        "r0": r0,
        "rw": rw,
    }
