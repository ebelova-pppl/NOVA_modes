import os
import numpy as np
import re
import warnings

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


def continuum_scalars(mode, omega, low2_full, high2_full, r=None, alpha=1.0):
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

    # Closest approach radius (global)
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
