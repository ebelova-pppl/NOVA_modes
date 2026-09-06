# Deterministic TAE rule configurations

`tae_rules_production_v3.yaml` is the current frozen deterministic production
preset. It retains the v2 `BAD_INTERIOR_UNRESOLVED_ENVELOPE` gate and adds the
last-ordered `BAD_INTERIOR_HARMONIC_INCOHERENCE` gate. The new gate rejects
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
`tae-rule-features-grouped-v16`. It adds only audit summaries for the global
`W`-weighted effective harmonic count, global/core `G_3`, and `B_3,core` at
strict `N_eff(r)>3`; none is a decision input. This schema-only extension does
not change this frozen configuration's bytes, ruleset, or SHA-256. Existing
grouped-v15 outputs remain valid historical results.

The inherited interior-envelope gate uses an
inclusive `r_peak <= 0.5`, connected total-energy FWHM no greater than two
grid intervals, and a gate-specific continuum-extremum exception requiring
`ext_dr <= 0.02` and `0 <= ext_df_gap <= 0.04`. The exception search covers
candidate centers through `r=0.50` without changing the established RF
extremum-feature definition.

`tae_rules_production_v1.yaml` and `tae_rules_production_v2.yaml` remain
byte-for-byte as the historical v14 and v15 presets. Use the corresponding
historical checkout to execute either pinned ruleset. The canonical sorter
loads production-v3 automatically under its default rules method:

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
  --rule_config tae_rules_production_v3
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
