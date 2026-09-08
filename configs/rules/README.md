# Deterministic TAE rule configurations

`tae_rules_production_v5.yaml` is the current frozen deterministic production
preset. It appends `BAD_CONTINUUM_CROSSING_TAIL` after every v4 gate, retaining
all earlier primary reasons. At the same actual lower/upper crossing it
requires strict `K_c > 0.4` and `T_2 > 0.035`, where
`T_2=E_tail/(E_h1+E_h2)` references the two individual harmonics with largest
full-domain integrated squared amplitudes. The numerator includes all tail
harmonics on the side opposite the global radial-energy peak. K uses the
unscaled signed second difference and complete stencil centers within ±4
native intervals of the crossing. It applies only to 201-point radial grids;
other resolutions retain evidence and fail open. Both cuts must hold at one
crossing; separate maxima cannot trigger the gate. Its measurements and
qualifying witness are in `crossing_features.continuum_crossing_tail`.
The frozen v5 configuration SHA-256 is
`982cc0ba3f17aae03a9fc6a4b662104200df0ff2897bda4de21131ce71c5bc9f`.

The preceding v4 preset retains every v3 decision and adds
`BAD_NEAR_AXIS_GRID_OSCILLATION` after the two existing grid-scale gates and
before the continuum gates. For each harmonic, the new gate finds maximal
runs of strictly consecutive nonzero sign changes; a zero or missing flip
ends the run. It rejects only when one run has at least four consecutive sign
flips, that run's largest absolute sample is at strictly `r < 0.1`, its
single-harmonic step norm
`Q_s=sqrt(sum_i (A[i+1]-A[i])^2)` is at least `0.30`, and the independent
mode-level maximum over every harmonic at strictly `r < 0.1` is at least
`0.10`. Runs and `Q_s` are never added across harmonics.
The frozen v4 configuration SHA-256 is
`ddefb105a8faac4d4050eda1636966d28dd6217c9af50305c7ae974c6666985b`.

The inherited `BAD_INTERIOR_HARMONIC_INCOHERENCE` gate rejects
only 201-point modes whose calibrated combined score is strictly greater than
`0.10`:

```text
S_inc = f_core * J_core * N_eff_core * (1 - C_adj)
```

The inclusive core is `r <= 0.5`. `J_core` is the base-2 Jensen--Shannon
divergence between adjacent squared-amplitude harmonic distributions, weighted
by `sqrt(W_i W_(i+1))`; `N_eff_core` is the radial-energy-weighted mean of the
pointwise inverse participation ratio. `C_adj` averages adjacent active
stored-row signed-profile coherences with weight `sqrt(E_h E_(h+1))`; each
pair coherence is maximized over radial lags from `-5` through `+5`. A row is
active when it carries at least `0.005` of integrated core energy. Missing
evidence or a radial sample count other than the calibrated `201` fails open
while retaining the audit measurements.

The current grouped `rule_features` output schema is
`tae-rule-features-grouped-v18`. It adds the crossing-tail energy, roughness,
resolution status, individual crossing records, and qualifying witness.
The complete near-axis amplitude and selected sign-flip-run evidence remain.
The inherited audit summaries for the global
`W`-weighted effective harmonic count, global/core `G_3`, and `B_3,core` at
strict `N_eff(r)>3` remain evidence only and are not decision inputs.

The inherited interior-envelope gate uses an
inclusive `r_peak <= 0.5`, connected total-energy FWHM no greater than two
grid intervals, and a gate-specific continuum-extremum exception requiring
`ext_dr <= 0.02` and `0 <= ext_df_gap <= 0.04`. The exception search covers
candidate centers through `r=0.50` without changing the established RF
extremum-feature definition.

`tae_rules_production_v1.yaml`, `tae_rules_production_v2.yaml`, and
`tae_rules_production_v3.yaml`, and `tae_rules_production_v4.yaml` remain
byte-for-byte as the historical v14, v15, v16, and v17 presets. Use the
corresponding historical checkout to execute a pinned older ruleset.
The canonical sorter loads production-v5 automatically
under its default rules method:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/SHOT \
  --rf_model models/nova_mode_classifier.joblib \
  --out_dir /path/to/output
```

The rule engine still returns REVIEW/`NO_GOOD_TEMPLATE` for a mode that passes
all gates. The production orchestrator records the separate
`accept-as-good-v1` policy that promotes such survivors to final GOOD before
manual overrides. It then uses the supplied RF checkpoint only to select
representatives among close-frequency, structurally matched final-GOOD modes.
This workflow policy and RF ranking are not part of the frozen rule
configuration and do not change its bytes or SHA-256. Omitting `--rf_model`
retains every affected cluster member and is an audit fallback, not the
standard deduplicated production recipe.

For a conservative audit of this exact preset without survivor promotion, run:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/audit-output \
  --rule_config tae_rules_production_v5
```

Use `sort_shot_rules.py` without `--rule_config` for threshold calibration and
feature-only experiments.

Configuration files use strict JSON-compatible YAML so the production sorter
can load them with the Python standard library. The loader validates exact
keys, numerical constraints, the pinned rule-engine version, and the frozen
production SHA-256.

Do not edit a frozen production configuration. Add a newly named version for
any future gate, threshold, routing, or ruleset change. Any interface loading a
named configuration rejects config-owned threshold and gate overrides.
