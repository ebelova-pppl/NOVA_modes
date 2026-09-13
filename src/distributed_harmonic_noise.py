"""Continuum-independent, simultaneous multi-harmonic grid-scale noise.

All rejection conditions describe one window and the same qualifying centers.
Only a compact joint witness is exported; individual-window maxima are never
combined. The original calibration prototype remains independent audit evidence.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from continuum_noise import native_high_pass_energy


SCHEMA_VERSION = "distributed-harmonic-noise-v1"
BAD_DISTRIBUTED_HARMONIC_NOISE = "BAD_DISTRIBUTED_HARMONIC_NOISE"
DEFAULT_NHF_MIN = 4.0
DEFAULT_TOP2_MIN = 0.005
DEFAULT_LOCAL_MIN = 0.05
DEFAULT_RADIAL_LENGTH_MIN = 0.03
DEFAULT_WINDOW_DR = 0.05


@dataclass(frozen=True)
class DistributedNoiseThresholds:
    """Inclusive participation selection; strict energy and length rejection."""

    nhf_min: float = DEFAULT_NHF_MIN
    top2_min: float | None = DEFAULT_TOP2_MIN
    local_min: float = DEFAULT_LOCAL_MIN
    radial_length_min: float = DEFAULT_RADIAL_LENGTH_MIN
    window_dr: float = DEFAULT_WINDOW_DR

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "top2_min" and value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"distributed noise {name} must be finite and positive")
        if self.nhf_min < 1:
            raise ValueError("distributed noise nhf_min must be >=1")
        if not self.radial_length_min < self.window_dr <= 1:
            raise ValueError("distributed noise requires radial_length_min < window_dr <=1")

    @property
    def enabled(self):
        return self.top2_min is not None


def empty_distributed_noise_features(config=None):
    c = config or DistributedNoiseThresholds()
    return dict(schema_version=SCHEMA_VERSION, gate_enabled=c.enabled,
                thresholds=asdict(c), candidate_found=False, reason=None,
                n_radial=None, n_harmonics=None, native_grid_spacing=None,
                resolution_eligible=None, minimum_n_radial=math.ceil(2 / c.window_dr) + 1,
                actual_window_dr=None, maximum_effective_length=None,
                total_energy=None, top2_energy=None, top2_energy_share=None,
                top2_harmonic_indices=[], n_scanned_windows=0,
                n_firing_windows=0, n_population_windows=0, witness=None,
                status="UNAVAILABLE",
                energy_reference="full-domain-top-two-individual-harmonics",
                energy_quadrature="native-trapezoidal-node-weights",
                stencil_policy="all-three-samples-in-same-window",
                participation_policy="pointwise-N_hf-inclusive; same-selected-centers-for-HF-and-length",
                local_energy_policy="all-raw-nodes-in-window",
                witness_policy="max-min-cut-ratios; earliest-tie; default-top2-cut-when-disabled")


def extract_distributed_noise_features(mode, *, config=None):
    c = config or DistributedNoiseThresholds()
    raw_energy, hp2, weights, top2_indices = native_high_pass_energy(mode)
    nr = raw_energy.shape[1]
    dr = 1.0 / (nr - 1)
    raw = np.sum(raw_energy, axis=0)
    total = float(np.sum(raw))
    top2 = float(np.sum(np.sum(raw_energy, axis=1)[top2_indices]))
    span = min(nr - 1, int(np.floor(c.window_dr / dr + 1e-10)))
    result = empty_distributed_noise_features(c)
    result.update(n_radial=nr, n_harmonics=raw_energy.shape[0], native_grid_spacing=dr,
                  resolution_eligible=span >= 2, actual_window_dr=span * dr,
                  maximum_effective_length=max(0, span - 1) * dr,
                  total_energy=total, top2_energy=top2,
                  top2_energy_share=top2 / total if total > 0 else None,
                  top2_harmonic_indices=top2_indices.tolist() if total > 0 else [])
    if span < 2:
        result["status"] = "WINDOW_UNRESOLVED"
        return result
    if total <= 0:
        result["status"] = "ZERO_ENERGY"
        return result
    power = np.sum(hp2, axis=0)
    probabilities = np.divide(hp2, power[None, :], out=np.zeros_like(hp2),
                              where=power[None, :] > 0)
    concentration = np.sum(probabilities * probabilities, axis=0)
    nh = np.divide(1.0, concentration, out=np.zeros_like(power), where=concentration > 0)
    energies = power * weights
    best = None
    best_score = -1.0
    # Disabled gates still retain a diagnostic witness; their severity is null.
    reference_cut = c.top2_min if c.enabled else DEFAULT_TOP2_MIN
    for start in range(nr - span):
        end = start + span
        result["n_scanned_windows"] += 1
        centers = np.arange(start + 1, end)
        selected = centers[nh[centers] >= c.nhf_min]
        e = energies[selected]
        hf = float(np.sum(e))
        if hf <= 0:
            continue
        result["n_population_windows"] += 1
        length = float(dr / np.sum((e / hf) ** 2))
        raw_window = float(np.sum(raw[start:end + 1]))
        top2_ratio = hf / top2
        local = hf / raw_window
        fires = bool(c.enabled and top2_ratio > c.top2_min
                     and local > c.local_min and length > c.radial_length_min)
        result["n_firing_windows"] += int(fires)
        score = min(top2_ratio / reference_cut, local / c.local_min,
                    length / c.radial_length_min)
        if score > best_score:
            best_score = score
            best = dict(start_index=start, end_index=end, r_start=start * dr, r_end=end * dr,
                        qualifying_center_indices=selected.tolist(),
                        qualifying_nhf=nh[selected].tolist(),
                        qualifying_hf_energy=e.tolist(), n_qualifying=len(selected),
                        hf_energy=hf, raw_window_energy=raw_window,
                        hf_top2_ratio=top2_ratio, hf_total_fraction=hf / total,
                        hf_window_fraction=local, effective_length=length,
                        minimum_cut_ratio=score, fires=fires)
    result.update(witness=best, candidate_found=result["n_firing_windows"] > 0,
                  status="MEASURED" if best is not None else "NO_QUALIFYING_POPULATION")
    if result["candidate_found"]:
        result["reason"] = BAD_DISTRIBUTED_HARMONIC_NOISE
    return result
