# Deterministic TAE rule configurations

The current frozen production preset is `tae_rules_production_v13.yaml`
(configuration schema v13, rejection ruleset v25, grouped features v26,
severity schema v2). Its SHA-256 is
`5d1319910b578d9b684a367d358d5a2304a7319218fe1571b462e9ce9d3b3919`.

V13 adds significant secondary energy peaks to `BAD_EDGE_SPIKE` and the
accepted smoothness/footprint exception to the interior-envelope gate. The
secondary edge branch requires W_peak>=0.5*max(W), r>=0.97, own-FWHM<=10
grid intervals, and peak harmonic amplitude strictly above the maximum at
r<0.9. The original global-edge branch is retained. The footprint exception
requires strict F_spikes<0.50 and Q_local<0.05 in every tested width-0.05 peak
window; unavailable evidence cannot grant the exception. Other gates can
still reject a mode. See the [v13 adoption audit](../../audits/morphology_v13_20260913/README.md)
and [detailed rule reference](../../scripts/README.md#deterministic-rule-sorting-production-and-calibration-interfaces).

## Retained distributed-noise gate from v12

The frozen v12 preset has configuration schema v12, ruleset v23, features v24,
severity schema v2, and SHA-256
`6cec796ae20bac12f2f66bd18ac20a14d9e502aa64453c6f2b5ad10f7b54f925`.

V12 appends `BAD_DISTRIBUTED_HARMONIC_NOISE` after every existing gate. Scan
closed radial windows of width 0.05 without a continuum mask. Compute native
signed second differences divided by four, then select stencil centers with
simultaneous HF harmonic participation `N_hf>=4`. Reject only when that same
selected population in one window meets all three strict cuts:

- HF / full-domain top-two-harmonic energy >0.005;
- HF / raw energy of the whole window >0.05;
- effective radial length >0.03.

The numerator retains every harmonic. The two reference harmonics are ranked
by individually integrated full-domain energy, without an adjacency condition.
Native quadrature weights and the original high-pass calculation are shared
with `BAD_EXTENDED_CONTINUUM_NOISE`; that earlier gate keeps its thresholds
and precedence. No smoothing or resampling is introduced. A grid too coarse
to fit a complete stencil in the requested window emits an explicit resolution
warning, and its enabled severity is unavailable. Empirical calibration used
nr=201. See the [calibration](../../audits/distributed_harmonic_noise_20260913/top2_pilot39/README.md)
and [adoption](../../audits/distributed_harmonic_noise_20260913/adoption/README.md).

The new severity is the minimum of its three normalized energy/length cuts
in the strongest joint window. Participation selects the population before
these ratios are evaluated. Disabled gates are excluded from overall severity.
The new gate's severity joins `overall_rule_severity`, `rule_margin` and the
existing representative ranking, with version/configuration hashes.

## Production and calibration commands

The canonical sorter loads v13 automatically. Current rules require Python
3.10+, NumPy 2.x and SciPy; see [platform setup](../../docs/platforms.md).

```tcsh
python scripts/sort_shot_mixed.py --method rules \
  --shot_dir /path/to/SHOT --out_dir /path/to/output
```

A gate survivor remains `rule_decision=REVIEW` / `NO_GOOD_TEMPLATE`. The
production `accept-as-good-v1` policy separately promotes it to final GOOD,
then applies fingerprinted manual overrides and duplicate handling. V11 and
later presets choose final-GOOD representatives by lowest overall rule severity, with
mode-key tie breaking. No RF checkpoint is needed.

For conservative calibration without survivor promotion:

```tcsh
python scripts/sort_shot_rules.py --shot_dir /path/to/SHOT \
  --out_dir /path/to/audit-output --rule_config tae_rules_production_v13
```

Omit `--rule_config` to experiment with individual gate flags. The five v12
options are `--distributed_noise_nhf_min`, `--distributed_noise_top2_min`,
`--distributed_noise_local_min`, `--distributed_noise_radial_length_min`, and
`--distributed_noise_window_dr`. Use `--disable_distributed_harmonic_noise`
to retain measurements without rejection. A named configuration rejects
config-owned threshold/gate overrides.

## Frozen version history

All earlier configuration files remain byte-for-byte frozen. Supported legacy
adapters explicitly disable gates absent from their schema:

| Preset | Change introduced | Duplicate ranking |
| --- | --- | --- |
| v5 | Continuum-crossing tail | RF p_good |
| v6 | Direct EAE routing for fraction below upper <0.2 | RF p_good |
| v7 | Low-amplitude, low-K crossing-window exception | RF p_good |
| v8 | Strict extremum clearance >0.1% | RF p_good |
| v9 | Axis energy concentration | RF p_good |
| v10 | Extended continuum-side noise | RF p_good |
| v11 | Normalized severities; unchanged v10 morphology | Rule severity |
| v12 | Distributed harmonic noise anywhere in radius | Rule severity |
| v13 | Secondary edge peaks and interior-envelope footprint exception | Rule severity |

Frozen v5-v11 disable the distributed branch; v5-v12 retain the global-only
edge gate and disable the footprint exception. V11 retains its ruleset-v22
identity and original hash
`da5019505f7e7a8025215e78dd41b0ed93dbbb3e41a470a0a0cb22b771bac9da`.
Their morphology decisions remain supported, while new exports use the current
feature/severity schemas. Exact historical export bytes require the matching
source checkout. V1-v4 also require their historical checkout to execute.

Configurations use strict JSON-compatible YAML and load with the standard
library. The loader validates exact keys, numeric constraints, the pinned
ruleset, and frozen hashes. Never edit a frozen preset; create a newly named
version for a gate, threshold, routing, or ruleset change. Detailed inherited
gate definitions remain in [the scripts README](../../scripts/README.md).
