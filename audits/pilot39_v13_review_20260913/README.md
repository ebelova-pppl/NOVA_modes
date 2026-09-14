# Review of the 39 checked shots, production v13

Rebuilt from the installed rules and RF-CNN exports on September 13, 2026.
Both classifier output sets and all labels are unchanged. This review covers
only the 39 checked shots; processing the remaining database is deferred.

## Review lists

The comparison includes **6,004 paired TAE-side modes** (TAE-like plus mixed),
using final GOOD/BAD decisions **before duplicate removal**. There are
**365 rules-versus-RF-CNN disagreements**:

| List | Modes | Purpose |
| --- | ---: | --- |
| [disagreements.csv](disagreements.csv) | 365 | Complete comparison, ordered by shot, n and frequency |
| [disagreements_elena.csv](disagreements_elena.csv) | 365 | Editable review copy; fill `manual_decision` and `manual_reason` for requested changes |
| [rules_bad_ai_good.csv](rules_bad_ai_good.csv) | 218 | Rules reject, AI accepts |
| [rules_good_ai_bad.csv](rules_good_ai_bad.csv) | 147 | Rules accept, AI rejects |
| [rules_vs_rf.csv](rules_vs_rf.csv) | 399 | Separate comparison with RF at p_good>=0.5 |
| [rules_vs_cnn.csv](rules_vs_cnn.csv) | 721 | Separate comparison with CNN at p_good>=0.5 |
| [shot_summary.csv](shot_summary.csv) | 39 | Counts for every shot, including zero-disagreement shots |

These are disagreements, not accuracy measurements. `rf_cnn_decision` is
the saved ensemble's actual final label; it is not a threshold on the mean
probability. The lists retain both scores, the ensemble tier, rule reason,
severity, selected-representative flags and input fingerprint.
`label` and `rules_decision` display the current final rules-workflow decision;
`original_rule_decision` preserves BAD/REVIEW from the deterministic engine.

The list is identical to the immediately preceding verified v13 list.
Relative to v12, [one disagreement is added](added_since_v12.csv),
E203655F01t025 N3/1987, and [five disappear](removed_since_v12.csv).
Existing annotated lists elsewhere in `audits/` are preserved.

[Eight recalculated C50/N1 modes](ai_comparison_excluded.csv) have current
rules results but no current AI GOOD/BAD classification. They are excluded
from all three paired comparisons, not from the rules outputs. EAE-routed
and INVALID modes are outside this morphology comparison.

Per-shot disagreement CSVs, including header-only lists for shots with none,
are in `outputs/review_pilot39_v13_20260913/by_shot/`. The complete paired
comparison is in `outputs/review_pilot39_v13_20260913/all_comparisons.csv`.
These repetitive generated files stay outside version control.

## Inspect and mark corrections

From the repository with the scientific environment active (Flux/tcsh):

```tcsh
python viz/view_modes_csv.py audits/pilot39_v13_review_20260913/disagreements.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

The viewer displays the current rules label and does not write labels.
In `disagreements_elena.csv`, enter `GOOD`, `BAD` or `REVIEW` in
`manual_decision` and a short explanation in `manual_reason` for cases to
override. Leave both blank for no change. Keep the existing `rules_decision`,
mode key and fingerprint as the audit of the starting point. This worksheet
is not a canonical override file: after review, confirmed entries need to
be converted to the existing override schema with reviewer and timestamp.
No review decisions have been prefilled or applied.

The established `--manual_overrides` option applies GOOD/BAD/REVIEW
adjudication after the automatic survivor policy and before deduplication.
It preserves the original rule decision/reason and only applies a unique
override whose fingerprint matches the current mode and datcon contents.
It cannot restore an INVALID input or change EAE routing. For interactive
adjudication including gate-rejected modes, use `label_modes_fast.py` with
`--adjudication all --no-rf`; see the
[manual-adjudication instructions](../../scripts/README.md#manual-adjudication).

After the review, apply the confirmed overrides, regenerate affected rules
shots and assemble the deduplicated accepted list for the group. The current
39-shot baseline contains **1,646 final GOOD modes before deduplication and
1,635 representatives**. That is the starting point for review, not a claim
that every retained mode has been manually approved. No group distribution
has been performed. The remaining database will be handled later.

## Verification and reproduction

The [receipt](verification.json) records source CSV hashes and counts. All
6,012 rule-evaluated modes had their raw mode-plus-datcon fingerprints
checked. Every paired row has matching omega, nr, gap region, signed_delta
and fraction_below_upper2 in the two exports. Source CSV hashes were stable
throughout the refresh, and all 365 disagreements exactly reproduce the
verified v13 decisions and scores. There are no applied manual overrides
in the current 39-shot baseline. Production configuration SHA256 is
`5d1319910b578d9b684a367d358d5a2304a7319218fe1571b462e9ce9d3b3919`.

```text
python audits/pilot39_v13_review_20260913/refresh.py --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai --data-root /path/to/DiTw --out-dir /path/to/new-review --runtime-dir outputs/review_pilot39_v13_new
```

The script refuses to replace an existing review worksheet. Review annotations
may intentionally change the worksheet hash after this initial receipt.
