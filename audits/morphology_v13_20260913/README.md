# Production v13: footprint exception and secondary edge peaks

The user approved activation and regeneration on 2026-09-13. **All 39 reviewed
rules shot outputs are regenerated, verified and installed** in the existing
`sort_outputs` root. Previous v12 outputs are preserved under
`sort_outputs/before_morphology_v13_20260913/`. Every installed and backup tree
matches its receipt, and every RF-CNN export tree remains unchanged.

## Adopted behavior

`BAD_INTERIOR_UNRESOLVED_ENVELOPE` retains its original candidate definition
and continuum-extremum exception. It gains an independent exception requiring
both **F_spikes<0.50** and **Q_local<0.05** in peak windows of nominal width
0.05. The shared calculation in `src/envelope_footprint.py` reproduces all
41 [calibration measurements](../envelope_footprint_20260913/measurements.csv).

F_spikes integrates total radial mode energy W over disjoint regions above
half the **global** maximum W, selecting regions <=2 native intervals wide
with peak r<=0.5, divided by full-domain integrated W. Q_local is the largest
signed `diff2(xi)/4` energy/raw-energy ratio in windows centered on the global
W peak and each selected region's peak. Complete stencils must fit inside
the same window. Integrals include interpolated region endpoints and
trapezoidal window endpoint weights; overlapping windows do not double-count
spike energy. Both cuts are strict. Unresolved windows cannot grant the
exception, and other gates can still reject the mode. Energy participation
length is a diagnostic, with no additional threshold.

`BAD_EDGE_SPIKE` preserves the original global-W-maximum branch. It also
checks secondary local W peaks. For the **same peak**, require:

- W_peak>=0.5*max(W);
- peak r>=0.97;
- connected W FWHM<=10 native intervals at half that peak's own height;
- max_h |xi_h(r_peak)| > max_{h,r<0.9} |xi_h(r)|, strictly.

Width is measured on the full grid. The amplitude numerator is at the W peak;
the reference is the maximum over every harmonic at strict r<0.9. There is
no local-median condition. The original global measurements remain intact
for the interior gate. See the [edge refinement](../edge_secondary_peaks_20260913/body_amplitude_comparison.csv).

The canonical `sort_shot_mixed.py --method rules` and conservative
`sort_shot_rules.py` share these changes. The production survivor policy
promotes REVIEW to final GOOD; the calibration CLI keeps REVIEW. No manual
training labels are changed. Per-gate severity honors the new applicability
exception and the compound secondary edge criterion, and still supplies
duplicate ranking without RF.

## Regression and review lists

All **214 repository tests pass**. A fresh recomputation reproduces all 41
original calibration records. The batch verification covers **25,967 inputs**:
6,012 rule-evaluated TAE-side modes, 19,248 routed EAE and 707 invalid inputs.
All 6,012 evaluated modes have nr=201; input fingerprints, validity, routing
and unrelated gate features/severities are identical to the installed v12
baseline. No resolution warnings or ranking fallbacks occur.

There are exactly **six final label changes**, with no primary-reason-only
changes and no additional representative swaps:

| Shot | Mode | Previous | Current | Cause |
| --- | --- | --- | --- | --- |
| E202926A03t025 | N3/1151 | BAD | GOOD | Footprint/smoothness exception |
| E202926A03t025 | N4/3735 | BAD | GOOD | Footprint/smoothness exception |
| G142301L94 | N9/2746 | BAD | GOOD | Footprint/smoothness exception |
| E203655F01t025 | N2/2035 | GOOD | BAD | Secondary edge peak |
| E203655F01t025 | N3/1987 | GOOD | BAD | Secondary edge peak |
| E205057A01t020 | N4/1444 | GOOD | BAD | Secondary edge peak |

Fifteen modes gain the interior exception; twelve still fail another gate.
N7/8319 remains BAD: F_spikes=71.5893% exceeds the exception's <50% cut.
The retained total remains 1,646 before deduplication and 1,635 representatives.
G L94 N9/2746 is accepted by the adopted rule, not a new manual visual label.

Fresh evaluation of all **2,327 active training rows** finds one decision
change: GOOD-labeled E204669M03t025 N4/1691 becomes BAD_EDGE_SPIKE, the known
conflict from the edge audit. The footprint exception changes no training
decisions. The training list itself is unchanged. Current outcomes are
541/575 GOOD labels retained, 34 rejected; among BAD labels, 1,702 rejected,
23 retained, 26 routed EAE and one invalid.

Rules versus saved RF-CNN disagreements fall **369 -> 365 / 6,004 paired
TAE-side modes**. In the latest twelve shots they fall **143 -> 141**.
Five disagreements disappear and one is added: **E203655F01t025 N3/1987**.
The latest-twelve subset loses N3/1151, N4/3735 and N2/2035, and gains
N3/1987. Historical disagreement lists and user annotations are preserved.

- [Six label changes](label_changes.csv)
- [Only newly added latest-12 disagreements](latest12_disagreements_added.csv)
- [Removed latest-12 disagreements](latest12_disagreements_removed.csv)
- [Complete current latest-12 list](latest12_disagreements.csv)
- [Complete current 39-shot list](current_disagreements.csv)
- [Training conflict](training_changes.csv)
- [Shot totals](shot_summary.csv), [selection changes](selection_changes.csv)
- [Tests](tests.json), [calibration reproduction](calibration_reproduction.json)
- [Verification](verification.json), [installation and backups](publication.json)

## Configuration and reproduction

The current configuration is `tae_rules_production_v13`, ruleset v25,
grouped feature schema v26 and severity schema v2. Configuration SHA256:
`5d1319910b578d9b684a367d358d5a2304a7319218fe1571b462e9ce9d3b3919`.
The previously uncommitted v13 candidate was completed before its first
batch installation; frozen v5-v12 retain their settings and decisions.

```text
python audits/morphology_v13_20260913/regenerate.py stage --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai --training-root /path/to/data_mixed --ditw-root /path/to/DiTw --out-root outputs/review_morphology_v13_new
```

`stage` runs the canonical sorter for all 39 checked shots, preserving any
manual overrides. It compares fingerprints, input validity, routing, every
unrelated feature and gate severity, then reevaluates all 2,327 active training
rows against the verified v12 baseline. Only after these checks does it
write `verification.json`. `publish` copies and verifies all new trees before
replacing the existing outputs, with the previous trees retained as backups.
RF-CNN is not rerun; its existing exports are checked for unchanged content
and aligned inputs when forming the new disagreement lists. Corrected C50/N1
has eight TAE-side modes without current AI classifications; they remain
excluded from the paired comparison.

Full regenerated outputs and per-shot logs are under ignored
`outputs/review_morphology_v13_20260913/`. Keep the compact CSV differences
and verification/publication receipts here for review.
