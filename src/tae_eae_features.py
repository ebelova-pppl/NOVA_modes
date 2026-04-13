from __future__ import annotations

import numpy as np

from cont_features import load_datcon_for_mode
from nova_mode_loader import load_mode_from_nova


def mode_weight_profile(mode: np.ndarray) -> np.ndarray:
    """
    Match the amplitude-squared radial weight used in cont_features.py.
    """
    return np.sum(np.abs(mode) ** 2, axis=0)


def upper2_scalars(
    mode: np.ndarray,
    omega: float,
    upper2_full: np.ndarray,
) -> dict[str, float]:
    """
    Compute simple TAE/EAE split scalars relative to the upper TAE gap boundary.

    Returns:
        signed_delta:
            Mode-weighted average of (upper2 - omega^2). Positive means mostly
            below the upper TAE boundary.
        fraction_below_upper2:
            Weighted fraction of mode energy at radii where upper2 > omega^2.
    """
    w = mode_weight_profile(mode)
    omega2 = float(omega) ** 2
    dist = upper2_full - omega2

    mask = np.isfinite(dist) & np.isfinite(w)
    if not np.any(mask):
        raise ValueError("No finite overlap between upper2 and radial mode weights")

    w_valid = w[mask]
    dist_valid = dist[mask]

    wsum = float(np.nansum(w_valid))
    if not np.isfinite(wsum) or wsum <= 0.0:
        raise ValueError("Zero valid mode weight for upper2 split")

    signed_delta = float(np.nansum(dist_valid * w_valid) / wsum)
    fraction_below_upper2 = float(np.nansum(w_valid[dist_valid > 0.0]) / wsum)
    fraction_below_upper2 = float(np.clip(fraction_below_upper2, 0.0, 1.0))

    return {
        "signed_delta": signed_delta,
        "fraction_below_upper2": fraction_below_upper2,
    }


def load_upper2_scalars_for_mode(mode_path: str) -> dict[str, float]:
    """
    Load a NOVA mode file plus its datcon file and compute upper-gap scalars.
    """
    mode, omega, gamma_d, ntor = load_mode_from_nova(mode_path)
    _low2_full, upper2_full, *_ = load_datcon_for_mode(mode_path, n_r=mode.shape[1])
    scalars = upper2_scalars(mode, omega, upper2_full)
    scalars.update(
        {
            "omega": float(omega),
            "gamma_d": float(gamma_d),
            "ntor": int(ntor),
        }
    )
    return scalars
