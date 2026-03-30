import numpy as np
"""
Modifies NOVA mode data from a full (n_m, n_r) array to a reduced (straighten) array 
centered around m_c(r) +/- M modes. Returns: mode_m[2M+1, n_r].
It will re-sample mode if n_r <> 201, interpolating to 201 grid.
"""

def resample_r(mode2d: np.ndarray, R_target: int = 201) -> np.ndarray:
    """
    mode2d: (n_m, n_r) on r in [0,1] uniform (assumed)
    Returns: (n_m, R_target) on uniform [0,1]
    """
    n_m, n_r = mode2d.shape
    if n_r == R_target:
        return mode2d.astype(np.float32, copy=False)

    r_old = np.linspace(0.0, 1.0, n_r, dtype=np.float32)
    r_new = np.linspace(0.0, 1.0, R_target, dtype=np.float32)

    out = np.empty((n_m, R_target), dtype=np.float32)
    for i in range(n_m):
        out[i, :] = np.interp(r_new, r_old, mode2d[i, :]).astype(np.float32)
    return out

def median_filter_1d_int(x: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple median filter for 1D int array. k must be odd.
    Edge handling: clamp window to valid range.
    """
    assert k % 2 == 1 and k >= 1
    n = len(x)
    h = k // 2
    y = np.empty_like(x)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n, i + h + 1)
        y[i] = int(np.median(x[lo:hi]))
    return y

def slew_limit_int(x: np.ndarray, max_step: int = 2) -> np.ndarray:
    """
    Enforce |x[i] - x[i-1]| <= max_step by clipping changes.
    Keeps sharp changes if they persist over multiple points.
    """
    y = x.astype(int).copy()
    for i in range(1, len(y)):
        d = y[i] - y[i-1]
        if d > max_step:
            y[i] = y[i-1] + max_step
        elif d < -max_step:
            y[i] = y[i-1] - max_step
    return y

def straighten_mode_window(
    mode: np.ndarray,
    M: int = 8,
    center_power: float = 2.0,
    median_k: int = 3,
    max_step: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mode: (n_m, n_r) real (signed) ndarray
    Returns: (2M+1, n_r) windowed & straightened signed mode.

    Steps:
      - compute w_m(r) = |mode|^center_power
      - weighted-mean center m_c(r) = sum(m*w)/sum(w)
      - round to int
      - median filter (k=median_k)
      - slew limit with max_step
      - extract signed window around m_c for each r, pad with zeros
    """
    mode = np.asarray(mode, dtype=np.float32)
    n_m, n_r = mode.shape
    m_idx = np.arange(n_m, dtype=np.float32)[:, None]  # (n_m,1)

    # weights per (m,r)
    w = np.abs(mode) ** center_power                  # (n_m,n_r)
    wsum = np.sum(w, axis=0) + 1e-12                  # (n_r,)

    # Identify radii where mode energy is effectively zero
    eps_w = 1e-4 * float(np.max(wsum))  # relative threshold
    valid = wsum > eps_w

    mc_int = np.zeros(n_r, dtype=int)

    if np.any(valid):
        mc = np.sum(m_idx * w, axis=0) / (wsum + 1e-8)  # (n_r,) float
        mc_tmp = np.rint(mc).astype(int)
        mc_tmp = np.clip(mc_tmp, 0, n_m - 1)

        # Fill only valid locations first
        mc_int[valid] = mc_tmp[valid]

        # Forward-fill then backward-fill invalid regions
        last = mc_int[valid][0]
        for j in range(n_r):
            if valid[j]:
                last = mc_int[j]
            else:
                mc_int[j] = last
        last = mc_int[valid][-1]
        for j in range(n_r - 1, -1, -1):
            if valid[j]:
                last = mc_int[j]
            else:
                mc_int[j] = last
    else:
        # Entire mode is basically zero (shouldn't happen), default to mid m
        mc_int[:] = n_m // 2

    # keep within bounds
    mc_int = np.clip(mc_int, 0, n_m - 1)

    # mild edge-preserving smoothing
    if median_k and median_k >= 3:
        mc_int = median_filter_1d_int(mc_int, k=median_k)

    # slew limiter to remove 1-point glitches
    if max_step and max_step >= 1:
        mc_int = slew_limit_int(mc_int, max_step=max_step)

    # extract window
    H = 2 * M + 1
    out = np.zeros((H, n_r), dtype=np.float32)

    for j in range(n_r):
        c = int(mc_int[j])
        m0 = c - M
        m1 = c + M + 1

        src0 = max(0, m0)
        src1 = min(n_m, m1)

        dst0 = src0 - m0
        dst1 = dst0 + (src1 - src0)

        if src1 > src0:
            out[dst0:dst1, j] = mode[src0:src1, j]

    mc_plot = mc_int.astype(np.float32)

    return out, mc_plot, mc_int
