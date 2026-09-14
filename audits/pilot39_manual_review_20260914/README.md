# Completed 39-shot manual disagreement review

Elena completed both directional disagreement lists and approved **17 label
changes** in `audits/pilot39_v13_review_20260913/disagreements_elena.csv`:
**9 BAD -> GOOD and 8 GOOD -> BAD**, spanning **13 shots**. The remaining
348 disagreements retain their current labels. This is a completed review
of the disagreement lists, not a new blind assessment of every mode.

All 13 affected shots have been regenerated, verified and installed in the
rules output root. Their previous exports are backed up under
`sort_outputs/before_manual_review_20260914/`; the other 26 rules exports
and all 39 AI exports are unchanged. The full 39-shot collection has
**1,647 GOOD modes before deduplication and 1,636 representatives**, with all
17 overrides applied. The input worksheet and its reasons are preserved
unchanged.

## Results and current files

| Count across all 39 shots | Before review | After review |
| --- | ---: | ---: |
| TAE-side candidates, including mixed | 6,012 | 6,012 |
| Final GOOD before deduplication | 1,646 | 1,647 |
| Selected GOOD representatives | 1,635 | 1,636 |
| Final BAD | 4,366 | 4,365 |
| Final-label / RF-CNN disagreements | 365 | 348 |
| Final BAD / AI GOOD | 218 | 209 |
| Final GOOD / AI BAD | 147 | 139 |

The automatic rules still have 365 disagreements with the saved ensemble;
the reduction to 348 comes from applying the approved manual labels. The
348 remaining disagreements were reviewed and kept as labeled. Separate
curated-label comparisons have 384 disagreements with RF and 710 with CNN
using standalone p_good>=0.5. These counts do not estimate model accuracy.
The eight recalculated C50/N1 modes without current AI classifications remain
outside the 6,004 paired comparisons.

- [Accepted TAE modes](accepted_tae_modes.csv): the **1,636 deduplicated
  representatives** for downstream use; all are in the `tae_like` category.
- [The 17 label changes and reasons](label_changes.csv)
- [Canonical manual overrides](manual_overrides.csv)
- [Disposition of all 365 reviewed disagreements](review_dispositions.csv)
- [Current final-label / AI disagreements](disagreements.csv), split into
  [BAD / AI GOOD](rules_bad_ai_good.csv) and [GOOD / AI BAD](rules_good_ai_bad.csv)
- [Resolved disagreements](resolved_disagreements.csv)
- [Per-shot counts](shot_summary.csv), [selection changes](selection_changes.csv)
- [Standalone RF comparison](final_vs_rf.csv), [CNN comparison](final_vs_cnn.csv)
- [AI comparison exclusions](ai_comparison_excluded.csv)
- [Verification receipt](verification.json), [installation receipt](publication.json)
- [Inventory metadata update](inventory_update.json): review notes only,
  preserving checked membership, statuses and training counts.

The full 6,012-row curated classification table and all paired comparisons
are in `outputs/review_pilot39_manual_20260914/`. Original review files remain
in `audits/pilot39_v13_review_20260913/`; they retain the pre-override labels.

Verification checks all 25,967 existing input rows: unchanged routing and
automatic evidence, exactly the 17 approved final-label changes, and exactly
17 corresponding selection-flag changes, with no additional representative
swaps. All 6,012 TAE-side input fingerprints match current files. Stale,
ambiguous, ineligible and unmatched override counts are zero. Final GOOD
lists agree with the selected flags, and there are no resolution warnings
or duplicate-ranking fallbacks. No production code or gate threshold changed.

## Decisions and provenance

- [Manual overrides](manual_overrides.csv) use the existing stable sorter
  schema. Decisions are normalized to uppercase; reasons are copied verbatim.
  `reviewer=Elena`; the timestamp records import of the completed review.
- Every changed mode's stored fingerprint matches its current mode and
  corresponding `datcon#`. The override changes the final decision after
  the automatic survivor policy and before frequency/structure deduplication.
  The automatic rule decision, reason, features and severity remain intact.
- In particular, E204955F02t017 N7/8319 is now approved through a manual
  override. This supersedes its earlier retained-BAD review decision without
  altering the interior-envelope gate.
- Production configuration remains `tae_rules_production_v13`, SHA256
  `5d1319910b578d9b684a367d358d5a2304a7319218fe1571b462e9ce9d3b3919`.
  Training labels, RF-CNN outputs and input-validity exclusions are preserved.

The unmodified original worksheet remains the source record. Its 365 rows
are not a canonical override file. The 17-row `manual_overrides.csv` here
is the reusable import, and each affected installed shot contains its
own matching subset. For future regeneration, pass the existing per-shot
file explicitly:

```text
python scripts/sort_shot_mixed.py --method rules --shot_dir /path/to/SHOT --out_dir /path/to/sort_outputs/SHOT --manual_overrides /path/to/sort_outputs/SHOT/manual_overrides.csv
```

An override with changed raw inputs requires review again. The sorter reports
stale, ambiguous, ineligible and unmatched overrides rather than applying them.

Use `final_decision` or the final GOOD list for downstream selection. The nine
manually rescued modes correctly retain `rule_decision=BAD` in the canonical
audit even though their final decision is GOOD. Their original gate severity
also remains available for diagnostics and representative selection.

## Export conventions

The combined accepted-mode manifest selects final GOOD representatives from
all 39 shots. The eligible pool includes the production TAE-like and mixed
categories; all 1,636 selected modes are TAE-like. It uses
portable `shot/N#/filename` paths; resolve them relative to the DiTw root.
Frequency is copied unchanged from the sorter. The CSV preserves input
fingerprints, decision source, original rule evidence and manual reasons.
It contains no EAE-routed or INVALID rows. Membership in this final list does
not imply that every agreement between rules and AI was individually reviewed.

## Recorded procedure

```text
python audits/pilot39_manual_review_20260914/apply_review.py stage --data-root /path/to/DiTw --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai --runtime-dir outputs/review_manual39_new
```

`stage` prepares the canonical overrides, runs `sort_shot_mixed.py --method
rules` for the affected 13 shots, then verifies the complete 39-shot database.
It checks every unchanged automatic field, all final labels, raw fingerprints,
override outcomes, output lists and duplicate selections before writing a
verification receipt. `publish` requires that receipt, stages and verifies
all replacement trees, preserves previous trees as backups and checks all 39
rules/AI trees after installation. Full regenerated exports, logs and combined
classifications are kept in ignored `outputs/review_pilot39_manual_20260914/`.
No messages or files have been sent to group members. The remaining database
stays deferred.

The command above records the initial import procedure; completed audit
receipts are protected from replacement. Future production reruns should
use the installed per-shot override file with the canonical command above.
