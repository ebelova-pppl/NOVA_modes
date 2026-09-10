"""Continuum-side grid-scale energy with extent in normalized radius.

Measure signed second differences before masking. Each contiguous above-upper
or below-lower region is independent, and only complete stencils wholly in
that region contribute to its candidate metrics. All energies use the same
native-grid trapezoidal node weights (half weight at domain endpoints).
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from cont_features import _validate_crossing_inputs


SCHEMA_VERSION = "continuum-side-noise-v2"
DEFAULT_TOP2_MIN = 0.01
DEFAULT_LOCAL_MIN = 0.20
DEFAULT_RADIAL_LENGTH_MIN = 0.04
BAD_EXTENDED_CONTINUUM_NOISE = "BAD_EXTENDED_CONTINUUM_NOISE"


@dataclass(frozen=True)
class ContinuumNoiseThresholds:
    """Inclusive production cuts; harmonic extent is audit-only."""

    top2_min: float | None = DEFAULT_TOP2_MIN
    local_min: float = DEFAULT_LOCAL_MIN
    radial_length_min: float = DEFAULT_RADIAL_LENGTH_MIN

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "top2_min" and value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.radial_length_min <= 1:
            raise ValueError("radial_length_min must be in (0, 1] normalized radius")

    @property
    def enabled(self):
        return self.top2_min is not None


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _participation(energy):
    total = float(np.sum(energy))
    # Scale before squaring to preserve very small, finite noise measurements.
    return float(1 / np.sum((energy / total) ** 2)) if total > 0 else None


def measure_continuum_noise(mode, omega, low2, high2):
    """Return per-region measurements on the full native, uniform radial grid.

    Full-domain harmonic energies rank the top two with stable lower-index tie
    breaking, just as in the crossing-tail gate. No adjacency requirement or
    physical poloidal-m offset is inferred. Unknown, reversed, or negative
    continuum bounds separate regions and cannot count as outside the gap.

    Regional raw energy is sum(weights * W * region_mask), not an integral
    with interpolated crossing endpoints. HF energy uses those same weights
    on eligible stencil centers. There is no smoothing, resampling, volume
    weighting, or division by dr**2. A zero-energy denominator is undefined.
    """
    A, omega, low, high, r = _validate_crossing_inputs(
        mode, omega, low2, high2, None
    )
    nr = A.shape[1]
    if nr < 3:
        raise ValueError("continuum noise requires at least three radial samples")
    if omega <= 0 or not np.isfinite(omega * omega):
        raise ValueError("continuum noise requires positive finite squared frequency")
    peak = float(np.max(np.abs(A)))
    A = A / peak if peak > 0 else A
    weights = np.full(nr, 1.0 / (nr - 1))
    weights[[0, -1]] *= 0.5
    raw_energy = A * A * weights
    harmonic_energy = np.sum(raw_energy, axis=1)
    total = float(np.sum(harmonic_energy))
    top2_indices = np.argsort(-harmonic_energy, kind="stable")[:2]
    top2 = float(np.sum(harmonic_energy[top2_indices]))
    hf = np.zeros_like(A)
    hf[:, 1:-1] = (A[:, 2:] - 2 * A[:, 1:-1] + A[:, :-2]) / 4
    hf_energy = hf * hf * weights
    valid = np.isfinite(low) & np.isfinite(high) & (low >= 0) & (high >= low)
    side = np.zeros(nr, dtype=np.int8)
    side[valid & (omega * omega < low)] = -1
    side[valid & (omega * omega > high)] = 1
    starts = np.flatnonzero((side != 0) & np.r_[True, side[1:] != side[:-1]])
    records = []
    for start in starts:
        stop = int(start) + 1
        while stop < nr and side[stop] == side[start]:
            stop += 1
        # A region's first/last samples have a neighbor outside the region,
        # or touch the domain boundary: neither has a complete eligible stencil.
        centers = np.arange(start + 1, stop - 1)
        all_complete = np.arange(max(1, start), min(nr - 1, stop))
        border = np.setdiff1d(all_complete, centers, assume_unique=True)
        known_neighbors = valid[border - 1] & valid[border + 1]
        crossing_border = border[known_neighbors]
        unknown_border = border[~known_neighbors]
        energy = hf_energy[:, centers]
        radial_hf = np.sum(energy, axis=0)
        harmonic_hf = np.sum(energy, axis=1)
        hf_out = float(np.sum(radial_hf))
        raw_out = float(np.sum(raw_energy[:, start:stop]))
        radial_extent = _participation(radial_hf)
        records.append(
            {
                "region_id": len(records),
                "side": "above_upper" if side[start] == 1 else "below_lower",
                "start_index": int(start),
                "end_index": stop - 1,
                "r_start": float(r[start]),
                "r_end": float(r[stop - 1]),
                "outside_sample_count": stop - int(start),
                "eligible_stencil_count": len(centers),
                "crossing_stencil_count": len(crossing_border),
                "unknown_neighbor_stencil_count": len(unknown_border),
                "hf_crossing_stencil_energy": float(np.sum(hf_energy[:, crossing_border])),
                "hf_unknown_neighbor_energy": float(np.sum(hf_energy[:, unknown_border])),
                "raw_out_energy": raw_out,
                "raw_out_total_fraction": _ratio(raw_out, total),
                "hf_out_energy": hf_out,
                "hf_out_top2_ratio": _ratio(hf_out, top2),
                "hf_out_total_fraction": _ratio(hf_out, total),
                "hf_out_local_fraction": _ratio(hf_out, raw_out),
                "hf_out_radial_extent": radial_extent,
                "hf_out_radial_length": radial_extent / (nr - 1) if radial_extent is not None else None,
                "hf_out_harmonic_extent": _participation(harmonic_hf),
                "hf_positive_radial_count": int(np.count_nonzero(radial_hf)),
                "hf_positive_harmonic_count": int(np.count_nonzero(harmonic_hf)),
                "hf_peak_r": float(r[centers[np.argmax(radial_hf)]]) if hf_out > 0 else None,
                "hf_peak_harmonic_index": int(np.argmax(harmonic_hf)) if hf_out > 0 else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "n_radial": nr,
        "n_harmonics": A.shape[0],
        "native_grid_spacing": 1.0 / (nr - 1),
        "resolution_policy": "native-grid-high-pass-with-radial-length",
        "energy_quadrature": "native-trapezoidal-node-weights",
        "stencil_policy": "all-three-samples-in-same-outside-region",
        "total_energy": total,
        "top2_energy": top2,
        "top2_energy_share": _ratio(top2, total),
        "top2_harmonic_indices": [int(i) for i in top2_indices] if total > 0 else [],
        "valid_continuum_sample_count": int(np.count_nonzero(valid)),
        "unknown_continuum_sample_count": int(np.count_nonzero(~valid)),
        "outside_sample_count": int(np.count_nonzero(side)),
        "records": records,
    }


def assess_continuum_noise(features, thresholds=None):
    """Apply an explicit hypothetical gate; never combine maxima across regions.

    Passing no thresholds disables the gate. Every supported native resolution
    is evaluated: only the high-pass operator remains tied to the grid scale.
    """
    result = {
        "gate_enabled": thresholds is not None and thresholds.enabled,
        "thresholds": asdict(thresholds) if thresholds is not None else None,
        "candidate_found": False,
        "reason": None,
        "qualifying_region_ids": [],
        "witness": None,
    }
    if thresholds is None or not thresholds.enabled:
        return result
    if features["schema_version"] != SCHEMA_VERSION:
        raise ValueError("continuum noise assessment requires v2 radial-length measurements")
    fields = (
        ("hf_out_top2_ratio", thresholds.top2_min),
        ("hf_out_local_fraction", thresholds.local_min),
        ("hf_out_radial_length", thresholds.radial_length_min),
    )
    qualified = [
        row for row in features["records"]
        if all(row[name] is not None and math.isfinite(row[name]) and row[name] >= cut
               for name, cut in fields)
    ]
    if qualified:
        result.update(
            candidate_found=True,
            reason=BAD_EXTENDED_CONTINUUM_NOISE,
            qualifying_region_ids=[row["region_id"] for row in qualified],
            witness=dict(max(qualified, key=lambda row: row["hf_out_top2_ratio"])),
        )
    return result


def empty_continuum_noise_features(config=None):
    """Stable unavailable evidence for a rule input that could not be measured."""
    resolved = config if config is not None else ContinuumNoiseThresholds()
    features = {
        "schema_version": SCHEMA_VERSION,
        "n_radial": None,
        "n_harmonics": None,
        "native_grid_spacing": None,
        "resolution_policy": "native-grid-high-pass-with-radial-length",
        "energy_quadrature": "native-trapezoidal-node-weights",
        "stencil_policy": "all-three-samples-in-same-outside-region",
        "total_energy": None,
        "top2_energy": None,
        "top2_energy_share": None,
        "top2_harmonic_indices": [],
        "valid_continuum_sample_count": None,
        "unknown_continuum_sample_count": None,
        "outside_sample_count": None,
        "records": [],
    }
    features.update(assess_continuum_noise(features, resolved))
    features["candidate_found"] = None
    return features


def extract_continuum_noise_features(mode, omega, low2, high2, *, config=None):
    """Shared production measurement and gate, including evidence when disabled."""
    resolved = config if config is not None else ContinuumNoiseThresholds()
    features = measure_continuum_noise(mode, omega, low2, high2)
    features.update(assess_continuum_noise(features, resolved))
    return features
