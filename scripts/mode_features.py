import numpy as np
import os

from cont_features import load_datcon_for_mode, continuum_scalars, warn_once_per_dir

def radial_centroid(mode, r):
    """
    mode: (n_m, n_r)
    r:    (n_r,) normalized to [0,1]
    """
    w = np.abs(mode)**2
    num = np.sum(w * r[None, :])
    den = np.sum(w) + 1e-14
    return num / den

def radial_width(mode, r, r_centroid=None):
    w = np.abs(mode)**2
    if r_centroid is None:
        r_centroid = radial_centroid(mode, r)
    num = np.sum(w * (r[None,:] - r_centroid)**2)
    den = np.sum(w) + 1e-14
    return np.sqrt(num / den)

def compute_features_for_mode(mode, extra_info=None):
    """
    mode: 2D numpy array of shape (n_m, n_r).
    We compute a set of simple 'roughness / singularity' features
    aggregated over poloidal harmonics.

    Assumes that radial grid is uniform on [0,1]

    Returns:
        1D numpy array of features for this mode.
        Now also adds continuum features.
    """

    n_m, n_r = mode.shape
    # Normalized uniform minor radius
    r = np.linspace(0.0, 1.0, n_r)

    mode_abs = np.abs(mode)

    # Normalize radial direction to [0, 1] if needed for derivatives
    # (here we assume uniform grid, so we just treat dr = 1)
    # First and second radial derivatives along r (axis=1)
    d1 = np.diff(mode, axis=1)
    d2 = np.diff(mode, n=2, axis=1)

    # Basic amplitude statistics
    mean_amp = mode_abs.mean()
    std_amp = mode_abs.std()
    max_amp = mode_abs.max()
    median_amp = np.median(mode_abs)

    eps = 1e-12
    max_to_mean = max_amp / (mean_amp + eps)
    max_to_median = max_amp / (median_amp + eps)

    # Roughness / singularity measures
    mean_abs_d1 = np.mean(np.abs(d1))
    mean_abs_d2 = np.mean(np.abs(d2))

    d1_abs = np.diff(np.abs(mode), axis=1)
    d2_abs = np.diff(np.abs(mode), n=2, axis=1)
    max_abs_d1 = np.max(np.abs(d1_abs))
    max_abs_d2 = np.max(np.abs(d2_abs))

    # Variation across poloidal harmonics (now just the top K)
    # (Singular modes might have one or a few m's blowing up.)
    K = 32
    per_m_max = mode_abs.max(axis=1)
    order = np.argsort(per_m_max)[::-1]
    top = order[:min(K, n_m)]
    per_m_mean = mode_abs.mean(axis=1)

    # statistics over m
    std_per_m_max = per_m_max[top].std()
    std_per_m_mean = per_m_mean[top].std()
    max_per_m_mean = per_m_mean[top].max()

    # Count of "spiky" points above some threshold
    thr_high = median_amp + 4 * std_amp
    n_spikes = int((mode_abs > thr_high).sum())  # changed because n_m varies for diff n
    spikes_per_m = n_spikes / n_m
    frac_spikes = n_spikes / mode_abs.size

    # Radial location and width
    rc = radial_centroid(mode, r)
    rw = radial_width(mode, r, rc)

    # Basic structure-based features
    features = np.array([
        mean_amp,
        std_amp,
        rc,
        rw,
        max_to_mean,
        max_to_median,
        mean_abs_d1,
        max_abs_d1,
        mean_abs_d2,
        max_abs_d2,
        std_per_m_max,
        max_per_m_mean,
        std_per_m_mean,
        #n_spikes,
        spikes_per_m,
        frac_spikes,
    ], dtype=float)

    # Now optional physics-based features added
    if extra_info is not None:
        if "omega" in extra_info:
            features = np.append(features, float(extra_info["omega"]))
        if "gamma_d" in extra_info:
            features = np.append(features, float(extra_info["gamma_d"]))
        if "ntor" in extra_info:
            features = np.append(features, float(extra_info["ntor"]))

    # Get continuum related scalars (optional) 02/04/26
    CONT_FALLBACK = [0.0, 1e30, 1e30, 0.0]  # [r_star, delta2_eff, S, W_star]

    mode_path = extra_info.get("path") if extra_info else None
    omega = float(extra_info.get("omega")) if extra_info and "omega" in extra_info else None

    if mode_path is not None and omega is not None:
        try:
            #low2, high2, i1, i2 = load_datcon_for_mode(mode_path, n_r=n_r)  # load datcon
            low2, high2, *_ = load_datcon_for_mode(mode_path, n_r=n_r)  # load datcon
            cont = continuum_scalars(mode, omega, low2, high2, r=r)
            features = np.append(features, [
                #cont["has_intersection"],
                cont["r_star"],
                cont["delta2_eff"],
                cont["S"],
                cont["W_star"],
            ])
        except FileNotFoundError:
            warn_once_per_dir(
                mode_path,
                f"   \n"
                f"========================================================================\n"
                f"[NOVA-RF] Continuum file not found in directory:\n"
                f"  {os.path.dirname(os.path.abspath(mode_path))}\n"
                f"Continuum-related features will be DISABLED for modes in this directory.\n"
                f"Expected a datcon file ('datcon<N>') alongside mode files.\n"
                f"========================================================================"
            )
            features = np.append(features, CONT_FALLBACK)
        except Exception as e:
            warn_once_per_dir(
                mode_path,
                f"[NOVA-RF] Continuum feature computation failed in directory:\n"
                f"  {os.path.dirname(os.path.abspath(mode_path))}\n"
                f"Continuum-related features will be DISABLED for modes in this directory.\n"
                f"Reason: {type(e).__name__}: {e}"
            )
            features = np.append(features, CONT_FALLBACK)
    else:
        features = np.append(features, CONT_FALLBACK)


    return features

