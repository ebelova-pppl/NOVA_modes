# Deterministic TAE rule configurations

`tae_rules_production_v2.yaml` is the current frozen deterministic production
preset. It adds the final `BAD_INTERIOR_UNRESOLVED_ENVELOPE` gate with an
inclusive `r_peak <= 0.5`, connected total-energy FWHM no greater than two
grid intervals, and a gate-specific continuum-extremum exception requiring
`ext_dr <= 0.02` and `0 <= ext_df_gap <= 0.04`. The exception search covers
candidate centers through `r=0.50` without changing the established RF
extremum-feature definition.

`tae_rules_production_v1.yaml` remains byte-for-byte as the historical v14
preset. Use the corresponding historical checkout to execute that pinned
ruleset.
The canonical sorter loads production-v2 automatically under its default rules
method:

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
  --rule_config tae_rules_production_v2
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
