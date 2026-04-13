import os
import numpy as np
import re
import warnings

_WARNED_DATCON_DIRS = set()
DATCON_INVALID_SENTINEL_MIN = 999.0

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

    low2 = _mask_datcon_invalid(data[:, 0])
    high2 = _mask_datcon_invalid(data[:, 1])

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
