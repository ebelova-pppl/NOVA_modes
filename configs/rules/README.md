# Deterministic TAE rule configurations

`tae_rules_production_v1.yaml` is the frozen production preset for
`scripts/sort_shot_rules.py`. Run it by name:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/output \
  --rule_config tae_rules_production_v1
```

Configuration files use strict JSON-compatible YAML so the production sorter
can load them with the Python standard library. The loader validates exact
keys, numerical constraints, the pinned rule-engine version, and the frozen
production SHA-256.

Do not edit a frozen production configuration. Add a newly named version for
any future gate, threshold, routing, or ruleset change. The CLI rejects
config-owned threshold and gate flags when a named configuration is selected.
