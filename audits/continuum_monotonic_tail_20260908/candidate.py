"""Experimental tail detector; does not alter the production continuum loader.

Read native datcon values with read_raw(path, nr), then compare repair(...,
fill="last") with fill="mean". The new detector precedes existing cleanup,
which remains the fallback for isolated spikes and sentinels.
"""

from pathlib import Path
import numpy as np
from cont_features import (
    _mask_datcon_invalid,
    _repair_joint_trailing_datcon_spikes,
    _trim_trailing_datcon_spikes,
)


def read_raw(path, nr=201):
    with Path(path).open() as f:
        i1, i2 = map(int, f.readline().split())
        data = np.loadtxt(f)
    low = _mask_datcon_invalid(data[:, 0])
    high = _mask_datcon_invalid(data[:, 1])
    return low, high, np.arange(i1 - 1, i2) / (nr - 1)


def onset(low2, high2, r, ratio=5.0, slope_floor=100.0):
    low = np.sqrt(np.where(low2 >= 0, low2, np.nan))
    high = np.sqrt(np.where(high2 >= 0, high2, np.nan))
    valid = np.isfinite(low) & np.isfinite(high) & (high >= low)
    ids = np.flatnonzero(valid)
    if len(ids) < 6:
        return None
    end = int(ids[-1])
    start = end
    while (
        start > 0
        and valid[start - 1]
        and low[start] > low[start - 1]
        and high[start] > high[start - 1]
    ):
        start -= 1
    # Confirm a sustained tail against the interval preceding the whole rise.
    if end - start < 2 or start < 3 or not np.all(valid[start - 3 : start + 1]):
        return None
    gradients = [
        np.diff(x[start : end + 1]) / np.diff(r[start : end + 1]) for x in [low, high]
    ]
    for x, g in zip([low, high], gradients):
        reference = np.median(
            np.abs(
                np.diff(x[start - 3 : start + 1]) / np.diff(r[start - 3 : start + 1])
            )
        )
        if np.max(g) <= max(slope_floor, ratio * reference):
            return None
    # Backtrack to the first fast step, with continuation confirmed above.
    fast = np.flatnonzero(np.maximum(*gradients) > slope_floor)
    if not len(fast):
        return None
    j = start + 1 + int(fast[0])
    return j if r[end] - r[j] <= 0.08 else None


def repair(low2, high2, r, fill="last", ratio=5.0, slope_floor=100.0):
    j = onset(low2, high2, r, ratio, slope_floor)
    low, high = low2.copy(), high2.copy()
    if j is not None:
        for x in [low, high]:
            value = (
                np.sqrt(x[j - 1]) if fill == "last" else np.mean(np.sqrt(x[j - 4 : j]))
            )
            ids = np.arange(len(x)) >= j
            x[ids & np.isfinite(x)] = value**2
    low, high = _repair_joint_trailing_datcon_spikes(low, high)
    return _trim_trailing_datcon_spikes(low), _trim_trailing_datcon_spikes(high), j
