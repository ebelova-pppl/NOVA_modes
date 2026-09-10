# Normalized gate severity and representative ranking, 2026-09-10

Production preset v11 retains the v22 rejection rules and adds grouped feature
schema v23, normalized severities, and simple rules-only duplicate ranking.
The score is a threshold margin, not a calibrated probability or a new label.

## Result

All **4,187** evaluated TAE/mixed classifications and their previous feature
values match v10 exactly: 3,239 BAD and 948 GOOD before deduplication. Every
evaluated mode has complete severity information. The final count remains
**942**, with six duplicates removed. Two pairs choose different representatives:

| Shot | n | Previous RF choice | Severity choice | Previous choice severity | New choice severity |
|---|---:|---|---|---:|---:|
| E204645A16t015 | 7 | 4782 | 4975 | 0.706380 | 0.620590 |
| E204645A16t015 | 8 | 5971 | 5775 | 0.599677 | 0.567800 |

The largest normalized gate in all four cases is `BAD_GRID_SCALE_SPIKE`.
These are relative margins for already accepted, structurally similar modes;
they do not establish an independent probability of physical correctness.
All 191 tests pass, including native-grid behavior, exact threshold semantics,
compound conditions, subthreshold defects, stable ties, legacy RF policy,
missing-severity fallback, and an assertion that v11 never loads an RF model.

Eight recalculated C50/N1 files appeared during the run. They remain INVALID
under the existing user-approved exclusion until their continuum alignment
is reviewed. [Inventory changes](inventory_changes.csv) records them. Current
input count is 19,260; this inventory change is separate from ranking.

Evidence: [selection changes](selection_changes.csv), [shot summaries](shot_summary.csv),
[severity distributions](severity_distribution.csv), and [verification receipt](verification.json).
The existing [226 disagreements](../continuum_noise_20260910/current_disagreements.csv)
are unchanged because classifications are unchanged.

All 27 verified rules outputs have been installed in the existing
`sort_outputs/` root. Previous outputs are preserved under
`sort_outputs/before_rule_severity_v11_20260910/`. Publication verified the
installed trees, backups, and unchanged RF–CNN outputs; see the
[publication receipt](publication.json).

## Output and interpretation

Each valid TAE-side mode has flat CSV fields:

- `gate_severity_BAD_*` for every implemented gate;
- `overall_rule_severity`, the maximum enabled-gate severity;
- `rule_margin = 1 - overall_rule_severity` and `nearest_gate`;
- `severity_complete`, `severity_schema_version`, `severity_config_sha256`;
- named configuration identity/hash in `rule_configuration_name` and
  `rule_configuration_sha256`.

`rule_features.severity_features` contains the same values plus every gate's
original component values, thresholds, comparison operators, normalized
ratios, witness, enabled/status fields, and actual `fired` flag. All gates are
reported even after an earlier gate has already determined the BAD label.
Consequently `nearest_gate` need not equal the first BAD primary reason.

For a large-is-bad component, q=value/threshold. For a small-is-bad width,
q=threshold/max(width,1e-12), using the same grid-interval width as its gate.
AND combines with minimum, OR with maximum; maximize across candidates.
Components for an AND condition belong to the same candidate unless the rule
explicitly requires independent measurements (the near-axis oscillation gate).
No severity is clipped at one.

**One is the threshold surface, not a universal pass/fail convention.**
Inclusive gates can fire at one; strict gates fire above one. Preserve the
existing floating tolerances and consult `fired` for the exact decision.
Zero can mean no applicable candidate or a zero measured defect; `status`
makes this distinct from disabled or unavailable evidence. Disabled gates
are null/excluded from the overall score. An unknown enabled severity makes
overall severity and margin null. Nonpositive normalization thresholds are
undefined, including configurable zero cuts; they cannot silently create a
favorable score. The runtime threshold hash includes all gate configurations.

## Gate definitions

The following expressions use each gate's configured thresholds. Existing
categorical prerequisites and exceptions remain in force.

| Gate | Severity over applicable candidates |
|---|---|
| `BAD_AXIS_SPIKE` | Maximum over local absolute-harmonic axis peaks of min(amplitude/T_amp, T_width/width). Includes subthreshold peaks. |
| `BAD_GRID_SCALE_SPIKE` | Maximum over signed lobes of min(amplitude/T_amp, T_width(r)/width), with the existing high-r width rule. |
| `BAD_GRID_SCALE_PACKET` | Maximum over windows of min(peak amplitude/T_amp, kth strongest turn step/T_step). A turn's strength is the smaller adjacent step at an opposite-sign step pair; k is the required number of turns. |
| `BAD_NEAR_AXIS_GRID_OSCILLATION` | Min(independent mode-level axis amplitude/T_amp, strongest qualifying run's Q_s/T_Q). The existing consecutive-flip and run-peak-radius prerequisites apply. |
| `BAD_CONT_CROSS` | Maximum crossing energy/T_W. Disabled in production v11. |
| `BAD_CONT_CROSS_WINDOW` | Maximum over unexcused crossings of max(window amplitude/T_amp, window energy/T_W). These are window maxima, not point A_cross. |
| `BAD_EDGE_SPIKE` | T_width/global envelope width when the global peak is in the edge window. |
| `BAD_INTERIOR_UNRESOLVED_ENVELOPE` | T_width/global envelope width when the peak is interior and the approved continuum-extremum exception does not qualify. |
| `BAD_INTERIOR_HARMONIC_INCOHERENCE` | Incoherence score/T_score when measurable; absence of an active adjacent pair is inapplicable, and unsupported resolution is unavailable. |
| `BAD_CONTINUUM_CROSSING_TAIL` | Maximum over crossings of min(K/T_K, tail-over-top-two/T_tail); applicable crossings on unsupported grids are unavailable. |
| `BAD_AXIS_ENERGY_CONCENTRATION` | Min(axis amplitude/T_amp, inner energy fraction/T_energy). |
| `BAD_EXTENDED_CONTINUUM_NOISE` | Maximum over connected outside-gap regions of min(HF/top-two/T_top2, local HF fraction/T_local, radial length/T_length). No harmonic-count cut. |

For the new noise gate, the three thresholds remain 0.01, 0.20, 0.04.
The high-pass operator stays native-grid-relative. Full raw diagnostic records
remain in their previous feature groups, including exempted crossings and
ineligible-grid measurements. Candidate collectors reuse the existing peak
and width calculations without changing the original decision candidates.

## Duplicate selection

The v11 policy minimizes `overall_rule_severity`; exact ties use portable
mode key order. It uses the existing greedy frequency/structure resolver with
unchanged pairwise matching conditions. It does not implement a Pareto front,
weighted average, or new morphology thresholds. `duplicate_rank_score` stores
the nonnegative severity with source `rule_severity` (lower is preferred).
The resolver internally sorts the negative of that value because its shared
ranking interface maximizes scores. No RF/CNN checkpoint is loaded.

If enabled-gate severity is unavailable for any member of a close-frequency
cluster, all members of that cluster are retained with an explicit
`SKIPPED_SEVERITY_UNAVAILABLE` fallback. Frozen presets v5-v10 keep their RF
ranking behavior and model-related fallbacks. The explicit RF-CNN backend
retains its own scoring and duplicate policy.

## Reproduction and storage

```tcsh
python scripts/sort_shot_mixed.py --method rules \
  --shot_dir /path/to/SHOT --out_dir /path/to/output
```

`adopt_ranking.py stage` regenerates the fingerprinted 27-shot batch without
RF arguments; `verify` compares every old classification/feature and checks
severity-versus-fired consistency for every gate; `publish` installs verified
outputs with backups. The run and verification driver hashes are recorded
separately so the compact audit export can be updated while all production
source, model-independent feature, registry and baseline hashes stay fixed.

Runtime shot exports, logs, and the combined `mode_severities.csv` stay ignored
under `outputs/review_rule_severity_v11_20260910/`. Only compact summaries,
changed rows, scripts, and receipts are kept here. The distribution table
separates GOOD and BAD; compare shot margins using the same configuration and
with coverage counts, rather than treating them as calibrated shot quality.
