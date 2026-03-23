#!/usr/bin/env python3
"""
sort_shot.py
============

Extend rf_sort_shot.py by adding a post-processing stage that removes very
closely spaced near-duplicate modes from the GOOD list, while keeping distinct
mode types even when their frequencies are close.

Current behavior preserved:
- walk shot/N1..N10 directories
- classify each egn* file with an RF classifier
- optionally move BAD modes to N#/out/
- write CSV with all classified modes

New behavior:
- collect lightweight dictionaries for GOOD modes
- group by ntor
- cluster by close frequency
- compare signed ridge profiles inside each cluster
- keep one representative per inferred mode type
- write GOOD list, final selected list, and a cluster report

Expected local modules (same directory as this script):
- nova_mode_loader.py
- mode_features.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np

from nova_mode_loader import load_mode_from_nova
from mode_features import compute_features_for_mode


# -----------------------------------------------------------------------------
# Filesystem / classification helpers
# -----------------------------------------------------------------------------

def iter_n_dirs(shot_dir: str, n_min: int = 1, n_max: int = 10) -> Iterable[Tuple[int, Path]]:
    shot = Path(shot_dir)
    for n in range(n_min, n_max + 1):
        ndir = shot / f"N{n}"
        if ndir.is_dir():
            yield n, ndir


def classify_mode_rf(clf: Any, mode_path: str) -> Tuple[float, np.ndarray, float, float, int]:
    """
    RF classification wrapper.

    Returns
    -------
    p_good, mode, omega, gamma_d, ntor
    """
    mode, omega, gamma_d, ntor = load_mode_from_nova(mode_path)

    extra_info = {
        "path": mode_path,
        "omega": float(omega),
        "gamma_d": float(gamma_d),
        "ntor": float(ntor),
    }
    x = compute_features_for_mode(mode, extra_info).reshape(1, -1)

    if hasattr(clf, "predict_proba"):
        p_good = float(clf.predict_proba(x)[0, 1])
    elif hasattr(clf, "decision_function"):
        z = float(clf.decision_function(x)[0])
        p_good = 1.0 / (1.0 + np.exp(-z))
    else:
        p_good = float(clf.predict(x)[0])

    return p_good, mode, float(omega), float(gamma_d), int(round(float(ntor)))


# -----------------------------------------------------------------------------
# Mode/ridge helpers
# -----------------------------------------------------------------------------

def median_filter_1d_int(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        raise ValueError("median filter window must be odd")

    n = len(x)
    h = k // 2
    y = np.empty_like(x)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n, i + h + 1)
        y[i] = int(np.median(x[lo:hi]))
    return y



def slew_limit_int(x: np.ndarray, max_step: int = 2) -> np.ndarray:
    if max_step <= 0:
        return x.copy()
    y = x.astype(int).copy()
    for i in range(1, len(y)):
        d = y[i] - y[i - 1]
        if d > max_step:
            y[i] = y[i - 1] + max_step
        elif d < -max_step:
            y[i] = y[i - 1] - max_step
    return y



def compute_mc_int(
    mode: np.ndarray,
    center_power: float = 2.0,
    median_k: int = 3,
    max_step: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ridge center m_c(r) estimated by weighted mean in m.

    Returns
    -------
    mc_float : (nr,)
        floating point center
    mc_int : (nr,)
        rounded/smoothed integer center actually used for ridge envelope
    """
    mode = np.asarray(mode, dtype=np.float32)
    n_m, n_r = mode.shape
    m_idx = np.arange(n_m, dtype=np.float32)[:, None]

    w = np.abs(mode) ** center_power
    wsum = np.sum(w, axis=0) + 1e-12

    # float ridge
    mc_float = np.sum(m_idx * w, axis=0) / wsum

    # guard regions with vanishing amplitude
    eps_w = 1e-6 * float(np.max(wsum)) if np.max(wsum) > 0 else 1e-12
    valid = wsum > eps_w

    mc_int = np.zeros(n_r, dtype=int)
    if np.any(valid):
        mc_tmp = np.rint(mc_float).astype(int)
        mc_tmp = np.clip(mc_tmp, 0, n_m - 1)
        mc_int[valid] = mc_tmp[valid]

        # forward fill + backward fill invalid regions
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
        mc_int[:] = n_m // 2

    mc_int = np.clip(mc_int, 0, n_m - 1)
    # mild smoothing of the ridge index
    if median_k and median_k >= 3:
        mc_int = median_filter_1d_int(mc_int, k=median_k)
    if max_step and max_step >= 1:
        mc_int = slew_limit_int(mc_int, max_step=max_step)
    mc_int = np.clip(mc_int, 0, n_m - 1)

    return mc_float.astype(np.float32), mc_int.astype(np.int32)



def ridge_envelope_profile(mode: np.ndarray, mc_int: np.ndarray, dm_band: int = 1) -> np.ndarray:
    """
    was: a(r) = sqrt( sum_{|m-mc(r)|<=dm_band} A(m,r)^2 )
    now: a(r) = sum_{|m-mc(r)|<=dm_band} (|A| * A)/(sum(|A|) + eps) )
    """
    mode = np.asarray(mode, dtype=np.float32)
    n_m, n_r = mode.shape
    a = np.zeros(n_r, dtype=np.float32)
    for j in range(n_r):
        c = int(mc_int[j])
        lo = max(0, c - dm_band)
        hi = min(n_m, c + dm_band + 1)
        band = mode[lo:hi, j]

        # signed weighted average: numerator = sum(|A| * A), denominator = sum(|A|)
        denom = np.sum(np.abs(band)) + 1e-14
        a[j] = np.sum(np.abs(band) * band) / denom
        #a[j] = float(np.sqrt(np.sum(band * band)))

    return a

def ridge_envelope_profile_v2(  # not used
    mode: np.ndarray,
    r: np.ndarray,
    dm_band: int = 1,
    center_power: float = 2.0,
    median_k: int = 3,
    max_step: int = 2,
):
    """
    Build a signed ridge profile a(r) from a 2D mode A(m,r).

    Parameters
    ----------
    mode : ndarray, shape (n_m, n_r)
        Signed mode amplitude array A(m,r).
    r : ndarray, shape (n_r,)
        Radial grid.
    dm_band : int
        Number of harmonics around m_c(r) to include on each side.
    center_power : float
        Power used for ridge-center weighting, usually 2.0.
    median_k : int
        Median filter window for m_c(r).
    max_step : int
        Slew limiter for m_c(r).

    Returns
    -------
    profile : ndarray, shape (n_r,)
        Signed weighted ridge profile a(r).
    mc : ndarray, shape (n_r,)
        Float ridge center m_c(r).
    mc_int : ndarray, shape (n_r,)
        Integer ridge center used for band extraction.
    """
    mode = np.asarray(mode, dtype=float)
    n_m, n_r = mode.shape

    if r.shape[0] != n_r:
        raise ValueError(f"r has length {r.shape[0]}, but mode has n_r={n_r}")

    m_idx = np.arange(n_m, dtype=float)[:, None]

    # Weight for ridge center location
    w = np.abs(mode) ** center_power
    wsum = np.sum(w, axis=0)

    eps_w = 1e-6 * float(np.max(wsum) + 1e-30)
    valid = wsum > eps_w

    mc_int = np.zeros(n_r, dtype=int)

    if np.any(valid):
        mc = np.sum(m_idx * w, axis=0) / (wsum + 1e-12)
        mc_tmp = np.rint(mc).astype(int)
        mc_tmp = np.clip(mc_tmp, 0, n_m - 1)

        mc_int[valid] = mc_tmp[valid]

        # forward fill + backward fill invalid regions
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
        mc_int[:] = n_m // 2

    # mild smoothing of the ridge index
    if median_k and median_k >= 3:
        mc_int = median_filter_1d_int(mc_int, k=median_k)

    if max_step and max_step >= 1:
        mc_int = slew_limit_int(mc_int, max_step=max_step)

    mc_int = np.clip(mc_int, 0, n_m - 1)
    mc = mc_int.astype(float)

    # signed weighted ridge profile
    profile = np.zeros(n_r, dtype=float)

    for j in range(n_r):
        c = mc_int[j]
        lo = max(0, c - dm_band)
        hi = min(n_m, c + dm_band + 1)

        band = mode[lo:hi, j]

        # signed weighted average:
        # numerator = sum(|A| * A), denominator = sum(|A|)
        denom = np.sum(np.abs(band)) + 1e-14
        profile[j] = np.sum(np.abs(band) * band) / denom

    return profile, mc, mc_int


def radial_centroid(r: np.ndarray, a: np.ndarray) -> Tuple[float, float]:
    """
    Compute centroid from envelope a(r).Uses a(r)^2 weights.
    """
    a2 = np.asarray(a, dtype=np.float64) ** 2
    wsum = float(np.sum(a2)) + 1e-14
    r0 = float(np.sum(r * a2) / wsum)
    #dr = float(np.sqrt(np.sum(((r - r0) ** 2) * a2) / wsum))  # Not used
    return r0 #, dr

def quantile_width(
    a: np.ndarray,
    r: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
):
    """
    Compute radial quantile width of a 1D profile (a) using profile^2 as weight.
    a : ndarray, shape (n_r,) - signed or unsigned 1D profile a(r).
    r : ndarray, shape (n_r,)
    q_low, q_high : float -  Lower and upper cumulative energy quantiles.

    Returns
    width : float -  Quantile width = r(q_high) - r(q_low)
    """

    a = np.asarray(a, dtype=float)
    r = np.asarray(r, dtype=float)

    if a.shape[0] != r.shape[0]:
        raise ValueError("a and r must have same length")
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("Need 0 <= q_low < q_high <= 1")

    w = a ** 2
    wsum = np.sum(w)

    if wsum <= 1e-14:
        return 0.0, r[0], r[0]

    cdf = np.cumsum(w) / wsum

    r_lo = np.interp(q_low, cdf, r)
    r_hi = np.interp(q_high, cdf, r)
    width = float(r_hi - r_lo)

    return width, float(r_lo), float(r_hi)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-14 or nb < 1e-14:
        return 0.0
    cos = float(np.dot(aa, bb) / (na * nb))

    return abs(cos)



def resample_profile_to_grid(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x_old, y_old).astype(np.float32)



def same_mode_type(
    mode_i: Dict[str, Any],
    mode_j: Dict[str, Any],
    sim_threshold: float,
    r_tol: float,
    width_tol: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Compare two modes using ridge envelope profile, centroid, q_width.
    Profiles are interpolated to a common r-grid if needed.
    """
    ri = mode_i["r"]
    rj = mode_j["r"]
    ai = mode_i["ridge_profile"]
    aj = mode_j["ridge_profile"]

    if len(ri) != len(rj) or not np.allclose(ri, rj):
        aj_cmp = resample_profile_to_grid(rj, aj, ri)
        ai_cmp = ai
    else:
        ai_cmp = ai
        aj_cmp = aj

    sim = cosine_similarity(ai_cmp, aj_cmp)
    r0i, dri = mode_i["r0"], mode_i["dr"]
    r0j, drj = mode_j["r0"], mode_j["dr"]

    out = {
        "cosine": sim,
        "dr0": abs(r0i - r0j),
        "ddr": abs(dri - drj),
    }
    ok = (sim >= sim_threshold) and (out["dr0"] < r_tol) and (out["ddr"] < width_tol)
    return ok, out



def build_mode_dict(
    path: str,
    shot: str,
    ntor: int,
    omega: float,
    score: float,
    mode: np.ndarray,
    dm_band: int,
    center_power: float,
    median_k: int,
    max_step: int,
) -> Dict[str, Any]:
    mode = np.asarray(mode, dtype=np.float32)
    nhar, nr = mode.shape
    r = np.linspace(0.0, 1.0, nr, dtype=np.float32)
    mc_float, mc_int = compute_mc_int(mode, center_power=center_power, median_k=median_k, max_step=max_step)
    a = ridge_envelope_profile(mode, mc_int, dm_band=dm_band)
    r0 = radial_centroid(r, a)
    w90, r_lo, r_hi = quantile_width(a, r, q_low=0.10, q_high=0.90)
    dr = w90

    return {
        "path": path,
        "shot": shot,
        "ntor": int(ntor),
        "omega": float(omega),
        "score": float(score),
        "r": r,
        "nhar": int(nhar),
        "mode": mode,
        "mc_float": mc_float,
        "mc_int": mc_int,
        "ridge_profile": a,
        "r0": float(r0),
        "dr": float(dr),
    }


# -----------------------------------------------------------------------------
# Clustering / post-processing
# -----------------------------------------------------------------------------

def relative_freq_close(omega_i: float, omega_j: float, rel_tol: float) -> bool:
    denom = max(abs(omega_i), 1e-12)
    return abs(omega_j - omega_i) / denom < rel_tol



def cluster_modes_by_frequency(modes: Sequence[Dict[str, Any]], rel_tol: float = 0.02) -> List[List[Dict[str, Any]]]:
    if not modes:
        return []

    ms = sorted(modes, key=lambda m: m["omega"])
    clusters: List[List[Dict[str, Any]]] = [[ms[0]]]

    for m in ms[1:]:
        prev = clusters[-1][-1]
        if relative_freq_close(prev["omega"], m["omega"], rel_tol):
            clusters[-1].append(m)
        else:
            clusters.append([m])
    return clusters



def resolve_cluster(
    cluster: Sequence[Dict[str, Any]],
    sim_threshold: float,
    r_tol: float,
    width_tol: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Group a close-frequency cluster into inferred mode types.

    Returns
    -------
    kept_modes : list of representatives (one per type)
    type_groups : list of dicts with representative + members + pair metrics
    """
    if len(cluster) == 1:
        m = cluster[0]
        return [m], [{"rep": m, "members": [m], "comparisons": []}]

    # sort by score descending so representative rule is automatic
    work = sorted(cluster, key=lambda x: x["score"], reverse=True)
    type_groups: List[Dict[str, Any]] = []

    for mode in work:
        placed = False
        for grp in type_groups:
            rep = grp["rep"]
            ok, met = same_mode_type(mode, rep, sim_threshold=sim_threshold, r_tol=r_tol, width_tol=width_tol)
            grp["comparisons"].append((mode["path"], rep["path"], met, ok))
            if ok:
                grp["members"].append(mode)
                placed = True
                break
        if not placed:
            type_groups.append({"rep": mode, "members": [mode], "comparisons": []})

    kept = [grp["rep"] for grp in type_groups]
    return kept, type_groups



def postprocess_good_modes(
    good_modes: Sequence[Dict[str, Any]],
    rel_freq_tol: float,
    sim_threshold: float,
    r_tol: float,
    width_tol: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns
    -------
    selected_modes : flattened list of kept representatives
    cluster_records : list of per-cluster records for report writing
    """
    by_n: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for md in good_modes:
        by_n[int(md["ntor"])] += [md]

    selected: List[Dict[str, Any]] = []
    cluster_records: List[Dict[str, Any]] = []

    for ntor in sorted(by_n):
        clusters = cluster_modes_by_frequency(by_n[ntor], rel_tol=rel_freq_tol)
        for ic, cluster in enumerate(clusters, start=1):
            kept, type_groups = resolve_cluster(
                cluster,
                sim_threshold=sim_threshold,
                r_tol=r_tol,
                width_tol=width_tol,
            )
            selected.extend(kept)
            cluster_records.append({
                "ntor": ntor,
                "cluster_index": ic,
                "cluster": list(cluster),
                "kept": kept,
                "type_groups": type_groups,
            })

    selected.sort(key=lambda m: (m["ntor"], m["omega"]))
    return selected, cluster_records


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(list(header))
        w.writerows(rows)

def write_cluster_csv(path: Path, cluster_records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fp:   # Close-frequency cluster csv list
        for rec in cluster_records:
            cluster = rec["cluster"]
            kept = {m["path"] for m in rec["kept"]}
            if len(cluster) == 1:   # only if more than one mode in cluster
               continue

            for itg, grp in enumerate(rec["type_groups"], start=1):
                rep = grp["rep"]
                for m in grp["members"]:
                    label = "KEEP" if m["path"] in kept else "DROP"
                    fp.write(f"{m['path']},{label}\n")


def write_cluster_report(
    path: Path, cluster_records: Sequence[Dict[str, Any]],
    rel_freq_tol: float,
    sim_threshold: float,
    r_tol: float,
    width_tol: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        fp.write("Close-frequency cluster report for selected NOVA modes\n")
        fp.write(
            f"  Used parameters: rel_freq_tol={rel_freq_tol}" 
            f"  sim_tol={sim_threshold}  r_tol={r_tol}  width_tol={width_tol}\n"
        )
        fp.write("=" * 75 + "\n\n")

        for rec in cluster_records:
            ntor = rec["ntor"]
            ic = rec["cluster_index"]
            cluster = rec["cluster"]
            kept = {m["path"] for m in rec["kept"]}
            # Report only if more than one mode in cluster
            if len(cluster) == 1:
               continue

            wmin = min(m["omega"] for m in cluster)
            wmax = max(m["omega"] for m in cluster)
            fp.write(
                f"n={ntor}  cluster={ic}  size={len(cluster)}"
                f"  omega_range=[{wmin:.6g}, {wmax:.6g}]"  
                f"  d_omega/omega={2*(wmax-wmin)/(wmax+wmin):.6g}\n"
            )

            for itg, grp in enumerate(rec["type_groups"], start=1):
                rep = grp["rep"]
                fp.write(f"  type_group={itg}\n") #  representative={rep['path']}  score={rep['score']:.6f}\n")
                for m in grp["members"]:
                    status = "KEEP" if m["path"] in kept else "DROP"
                    fp.write(f"    [{status}] path={m['path']}\n") 
                    fp.write(f"               omega={m['omega']:.6g}  score={m['score']:.6f}  r0={m['r0']:.4f}  dr={m['dr']:.4f}\n"
                    )
                for comp in grp.get("comparisons", []):
                    p1, p2, met, same = comp

                    cos = met["cosine"]
                    dr0 = met["dr0"]
                    ddr = met["ddr"]

                    fp.write(
                        f"  Compare: {Path(p1).name}  vs  {Path(p2).name}  "
                        f"cos={cos:.3f}  dr0={dr0:.4f}  ddr={ddr:.4f}  same={same}\n"
                    )

            fp.write("-" * 75 + "\n\n")
         

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
"""
# Calibrated on example shots: nstxu_204202, nstx_120113
# different modes: cosine <= ~0.75
# duplicate/same-type modes: cosine >= ~0.93
# d_r0 and d_dr matter less, but are ~<0.02 for similar modes
"""

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classify NOVA modes in a shot directory and post-process close-frequency GOOD modes."
    )
    ap.add_argument("shot_dir", help="Shot directory, e.g. /global/cfs/cdirs/.../nstx_123456")
    ap.add_argument("--model", required=True, help="RF model joblib, e.g. nova_mode_classifier.joblib")

    # Preserve current rf_sort_shot.py output behavior
    ap.add_argument("--out_csv", default=None,
                    help="CSV of all classified modes. Default: <shot_dir>/rf_sorted.csv")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Mode is GOOD if p_good >= threshold (default 0.5)")
    ap.add_argument("--move_bad", action="store_true", help="Move bad modes to N#/out/")
    ap.add_argument("--dry_run", action="store_true", help="Do not move files; just report and write CSV")
    ap.add_argument("--n_min", type=int, default=1)
    ap.add_argument("--n_max", type=int, default=10)
    ap.add_argument("--pattern", default="egn*", help="Glob pattern for mode files (default egn*)")

    # New outputs
    ap.add_argument("--good_csv", default=None,
                    help="CSV of all GOOD modes before post-processing. Default: <shot_dir>/good_modes.csv")
    ap.add_argument("--selected_csv", default=None,
                    help="CSV of selected representative modes after post-processing. Default: <shot_dir>/selected_modes.csv")
    ap.add_argument("--cluster_report", default=None,
                    help="Cluster report text file. Default: <shot_dir>/cluster_report.txt")

    # Post-processing parameters
    ap.add_argument("--rel_freq_tol", type=float, default=0.02,
                    help="Relative frequency tolerance for close-frequency clustering (default 0.02)")
    ap.add_argument("--dm_band", type=int, default=1,
                    help="Half-band in m around ridge for envelope a(r) (default 1)")
    ap.add_argument("--sim_threshold", type=float, default=0.90,
                    help="Cosine similarity threshold for same-type test (default 0.90)")
    ap.add_argument("--r_tol", type=float, default=0.10,
                    help="Absolute tolerance in radial centroid r0 (default 0.10)")
    ap.add_argument("--width_tol", type=float, default=0.05,
                    help="Absolute tolerance in radial width dr (default 0.05)")
    ap.add_argument("--center_power", type=float, default=2.0,
                    help="Power used in ridge-center weighting |A|^p (default 2)")
    ap.add_argument("--median_k", type=int, default=3,
                    help="Median filter window for ridge center m_c(r) (default 3)")
    ap.add_argument("--max_step", type=int, default=2,
                    help="Slew limiter for ridge center m_c(r) (default 2)")

    args = ap.parse_args()

    shot_dir = Path(args.shot_dir).resolve()
    if not shot_dir.is_dir():
        raise SystemExit(f"Shot dir not found: {shot_dir}")

    shot_name = shot_dir.name
    out_csv = Path(args.out_csv) if args.out_csv else shot_dir / "rf_sorted.csv"
    good_csv = Path(args.good_csv) if args.good_csv else shot_dir / "good_modes.csv"
    selected_csv = Path(args.selected_csv) if args.selected_csv else shot_dir / "selected_modes.csv"
    cluster_report = Path(args.cluster_report) if args.cluster_report else shot_dir / "cluster_report.txt"
    cluster_csv = shot_dir / "cluster.csv"


    clf = joblib.load(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Shot dir: {shot_dir}")
    print(f"Threshold: {args.threshold}  (good if p_good >= thr)")
    print(f"Post-process: rel_freq_tol={args.rel_freq_tol}, dm_band={args.dm_band}, sim>{args.sim_threshold}, r_tol={args.r_tol}, width_tol={args.width_tol}")
    if args.move_bad:
        print("Will move BAD modes to N#/out/  (use --dry_run to preview)")

    rows_all: List[List[Any]] = []
    good_modes: List[Dict[str, Any]] = []

    n_total = 0
    n_good = 0
    n_bad = 0
    n_err = 0

    for n, ndir in iter_n_dirs(str(shot_dir), args.n_min, args.n_max):
        files = sorted(glob.glob(str(ndir / args.pattern)))
        if not files:
            continue

        out_dir = ndir / "out"
        if args.move_bad and not args.dry_run:
            out_dir.mkdir(exist_ok=True)

        print(f"\nN{n}: found {len(files)} files in {ndir}")

        for f in files:
            n_total += 1
            try:
                p_good, mode, omega, gamma_d, ntor = classify_mode_rf(clf, f)
                label = "good" if p_good >= args.threshold else "bad"
            except Exception as e:
                n_err += 1
                rows_all.append([f, "error", "", str(e)])
                print(f"  ERROR {f}: {e}")
                continue

            rows_all.append([f, label, f"{p_good:.6f}", ""])

            if label == "good":
                n_good += 1
                md = build_mode_dict(
                    path=f,
                    shot=shot_name,
                    ntor=ntor,
                    omega=omega,
                    score=p_good,
                    mode=mode,
                    dm_band=args.dm_band,
                    center_power=args.center_power,
                    median_k=args.median_k,
                    max_step=args.max_step,
                )
                good_modes.append(md)
            else:
                n_bad += 1
                if args.move_bad:
                    dest = out_dir / Path(f).name
                    if args.dry_run:
                        print(f"  would move BAD: {f} -> {dest}")
                    else:
                        if dest.exists():
                            stem = dest.stem
                            suf = dest.suffix
                            k = 1
                            while True:
                                alt = out_dir / f"{stem}__dup{k}{suf}"
                                if not alt.exists():
                                    dest = alt
                                    break
                                k += 1
                        shutil.move(f, dest)

        print(f"  N{n} done. cumulative: total={n_total} good={n_good} bad={n_bad} err={n_err}")

    # Post-process GOOD modes
    selected_modes, cluster_records = postprocess_good_modes(
        good_modes,
        rel_freq_tol=args.rel_freq_tol,
        sim_threshold=args.sim_threshold,
        r_tol=args.r_tol,
        width_tol=args.width_tol,
    )

    # Write outputs
    write_csv(
        out_csv,
        header=["path", "label", "p_good", "error"],
        rows=rows_all,
    )

    write_csv(
        good_csv,
        header=["path", "shot", "ntor", "omega", "p_good", "r0", "dr", "nhar"],
        rows=[
            [m["path"], m["shot"], m["ntor"], f"{m['omega']:.8g}", f"{m['score']:.6f}", f"{m['r0']:.6f}", f"{m['dr']:.6f}", m["nhar"]]
            for m in sorted(good_modes, key=lambda x: (x["ntor"], x["omega"]))
        ],
    )

    write_csv(
        selected_csv,
        header=["path", "shot", "ntor", "omega", "p_good", "r0", "dr", "nhar"],
        rows=[
            [m["path"], m["shot"], m["ntor"], f"{m['omega']:.8g}", f"{m['score']:.6f}", f"{m['r0']:.6f}", f"{m['dr']:.6f}", m["nhar"]]
            for m in selected_modes
        ],
    )

    write_cluster_report(
        cluster_report, cluster_records, 
        rel_freq_tol=args.rel_freq_tol,
        sim_threshold=args.sim_threshold,
        r_tol=args.r_tol,
        width_tol=args.width_tol,
    )

    write_cluster_csv(cluster_csv, cluster_records) # list of paths

    print("\n=== Summary ===")
    print(f"Total: {n_total} | Good: {n_good} | Bad: {n_bad} | Errors: {n_err}")
    print(f"Good modes before post-processing: {len(good_modes)}")
    print(f"Selected modes after post-processing: {len(selected_modes)}")
    print(f"Wrote all classified modes: {out_csv}")
    print(f"Wrote GOOD-mode list:      {good_csv}")
    print(f"Wrote selected-mode list:  {selected_csv}")
    print(f"Wrote cluster report:      {cluster_report}")
    print(f"Wrote cluster paths:      {cluster_csv}")


if __name__ == "__main__":
    main()
