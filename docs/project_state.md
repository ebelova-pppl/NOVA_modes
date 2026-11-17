# NOVA mode classifier: current project state

Updated November 16, 2026. Selection counts below retain their October 10/26/29
snapshot dates; November 16 checks NOVA-C density outputs and implements
the surrogate HDF5 export using the confirmed NOVA density prescription. The complete
prior chronology is preserved locally and in Git history as described in
[research records and recovery](research_archive.md).

## Purpose and current workflow

Select physically useful Alfvén eigenmodes from existing NOVA outputs for
NOVA-C, surrogate modeling, and digital-twin analysis. The repository does not
run NOVA or calculate energetic-particle growth rates.

- Production: `scripts/sort_shot_mixed.py --method rules`, frozen
  `tae_rules_production_v13` (ruleset v25, grouped features v26, severity v2).
- Conservative calibration: `scripts/sort_shot_rules.py`; survivors stay REVIEW.
- Production survivors retain their preliminary REVIEW / `NO_GOOD_TEMPLATE`
  evidence and become final GOOD through `accept-as-good-v1`. Fingerprinted
  manual overrides then apply; severity selects frequency/structure duplicates.
- EAE-like modes are routed separately and are not validated by TAE rejection
  gates. BAE assignments currently come from explicit manual overrides.
- Rules need Python 3.10+, NumPy 2.x and SciPy, with no RF/CNN checkpoint.
  Calibration primarily used nr=201; other grids can have incomplete screening.

See [getting started](getting_started.md), [platform setup](platforms.md),
[rule configurations](../configs/rules/README.md), and the
[script reference](../scripts/README.md) for commands and gate definitions.

## Installed results and shared catalog

The inventory has **184 processed post-training cases plus 14 active training
cases (198 total)**, including equilibrium variants and time slices. R06 is
counted as processed but its whole-shot inputs are invalid. Excluding R06
leaves 197 cases. E205059A01t025 and the empty `nstxu_202806` inventory entry
have no installed results; no released shots await processing.

The [accepted catalog](../catalogs/README.md), dated October 26, contains
**8,943 selected GOOD modes**: 8,366 post-training and 577 training-shot
selections, from 190 nonempty lists among 198 installed outputs. There are
8,941 TAE and two mixed entries; nine selections have applied manual GOOD
overrides. Historical training outputs retain v1/v2 rules and have no saved
severity; post-training outputs use v13.

Review status is separate from selection: **865 reviewed, 6,590 pending,
1,488 unknown**. Pending includes 39 selected training modes with human BAD
labels, documented in the [conflict list](../catalogs/accepted_modes.review_conflicts.csv).
The remaining 6,551 pending modes await review. An accepted-list count does
not establish individual visual review.

The October 29 statistics include all 198 installed outputs: **127,974 input
modes, 32,645 TAE-like/mixed candidates, and 8,943 selected GOOD**. All eight
zero-GOOD cases are G shots, including invalid R06. These are saved-output
counts, not a fresh run with one common rule version.

Retained [shot inventory](../audits/main_dataset_shots/shot_status.csv),
[review evidence](../audits/README.md), and the
[catalog review registry](../configs/catalog_review_sources.csv) support
regeneration. Future reruns must explicitly supply applicable manual overrides,
including BAD and BAE decisions; automatic rules alone do not reproduce all
curated lists.

## Input holds and unresolved scientific work

- **E205059A01t025:** N1/N2 continuum recalculation and correspondence review
  remain required. The October 10 recheck found unchanged mode/continuum inputs.
- **G133964R06:** the entire shot remains excluded pending corrected inputs and
  review. Its completed INVALID outputs do not remove the hold.
- The [known-invalid registry](../configs/known_invalid_inputs.csv) also
  excludes N1 for nstx_135388, G121123B12, G142301W29, and G142301Y93.
  See [active whole-shot holds](../audits/main_dataset_shots/active_shot_holds.csv).
- The October 29 parameter audit found 1,415 saved mode/continuum fingerprint
  mismatches in 25 non-G shots, associated in spot checks with newer continua.
  Poster measurements use saved selections and radial metrics with current
  continua; classifications were not rerun against those updated continua.
  The November 16 export check now finds **4,426 unmatched selected pairs in
  638 n scopes across 80 cases**, after checking live inputs and the training
  snapshot. These targets are withheld in HDF5; the saved catalog is unchanged.
  Available standalone continua remain usable with explicit unpaired status.
  **November 16 follow-up:** all 4,426 current mode files are nonempty/finite
  and exactly reproduce saved omega, gamma_d, radial centroid/RMS width and
  dimensions. Their mtimes/ctimes precede selection; full historical mode-byte
  identity is not independently established. The 638 affected `datconN` files
  have rewrite dates October 25–November 16, and all 80 cases have NOVA-C
  outputs. The inspected NOVA-C launch script runs `gocont_tr`; `datcon` links
  to `datconN` in all 638 directories, so this overwrites the numbered file
  without a backup step. No older full datcon copies were found for these
  80 cases in their directories or the two local training snapshots.
  Of 1,415 October 29 mismatches, 55 are recovered from training copies,
  leaving 1,360 previously mismatched plus 3,066 newly mismatched modes.
  For those 3,066, current versus October 29 frequency boundaries at the
  saved centroid differ by less than 0.01% for 2,801 modes; 32 exceed 1%
  in at least one boundary, maximum 46.98%. This is not a full-profile or
  original-selection-era continuum comparison. HDF5 exclusions reflect
  strict byte matching, not new BAD labels. The October poster had expressly
  retained all saved GOOD membership with current continua. Full diagnostic
  CSVs, dates, copy search and reproducible checks are local under
  `research_archive/nova_surrogate_input_changes_20261116/`.
  The follow-up on a possible 0.01% acceptance tolerance finds no change of
  below/inside/above-gap status at the centroid for the 2,801 small-change
  modes; the largest boundary shift there is 0.061% of local gap width.
  These modes use 378 continuum files: 349 pass at every compared centroid,
  while 29 have mixed results (82 small-change modes share a file with a
  larger-change mode). This supports provisional reuse, not full-profile
  equivalence. Recheck existing continuum-dependent diagnostics across each
  mode's radial extent, using frozen backups and preserving historical labels,
  before asserting unchanged mode/continuum compatibility. The user is
  discussing this policy; no new acceptance or HDF5 changes have been applied.
- Five live nstx_135388/N8 files were empty: 2197, 2861, 3062, 4424, and 7331
  (all E+02). Archived training pairs still match saved fingerprints. Reconcile
  the live files separately; no input or exclusion registry was changed.
- Adjudicate the 39 training-label/catalog conflicts when requested; preserve
  current labels and selections until explicit scientific decisions are made.

## Training and model status

The active training list contains **2,327 rows from 14 shots**, with 575 GOOD
labels. Q62 and confirmed invalid N1 inputs have preserved records outside
that active list. The active RF and raw-CNN checkpoints were fitted to the
older **2,390-row August 28 snapshot**; those provenance counts are distinct
from today's training list and the 577 saved training-shot GOOD selections.

Legacy RF+CNN sorting remains available with `--method rf-cnn`. The model
pipelines are `rf_train_classify.py`, `cnn_raw.py`, `cnn_straightened.py`, and
`cnn_hybrid.py`. Training-shot agreement is not held-out validation; the G-shot
continuum regime remains challenging. See [model provenance](../models/README.md)
and [training-label records](../training_labels/README.md).

On Flux use tcsh commands; on Perlmutter use Bash. `NOVA_DATA` identifies
training data and `NOVA_DITW_ROOT` the live production input database; keep
these roles separate and follow the [platform guide](platforms.md).

## Recent work and next steps

- **November 16 complete-case subset sent to Alvaro:** the user confirmed
  sending `nova_surrogate_v1_complete_cases.h5` to Alvaro. Preserve this
  delivered file as the baseline; publish any expanded dataset under a new
  filename/version. The full HDF5 contains 200
  inventory IDs / 199 populated equilibrium parameter sets, but only **94
  cases** have all parameters, available continua and verified K/mode targets
  for every n=1–10. Extracted these without relabeling or changing numerical
  values to `research_archive/nova_surrogate_export_20261116/`
  `nova_surrogate_v1_complete_cases.h5`: **940 case/n records, 3,584 modes,
  320 valid K=0 records**, 69 non-G and 25 G cases. Original full file remains
  unchanged. Completeness is across n; continuum radial masks remain required
  (49 cases have all n valid throughout nominal r=0.01–0.99). No 2D mode
  structures or lobe counts were added. Companion case-list CSV, HDF5 summary,
  and extraction script retain provenance; schema documentation explains
  missing-radius handling and the distinction between selection and review.
- **November 16 original-continuum evidence recovered:** the 1,360 modes
  already unmatched on October 29 span 23 cases / 179 continua. Of these,
  178 current files have modification dates October 25–29; E203262A04t018/N10
  has a November 4 timestamp but its content hash is identical to October 29,
  so that later timestamp does not date the original mismatch. Older
  `plotcontN` NCAR graphics survive for all 179 scopes, all dated May 29.
  They are readable; all 179 (296,402,400 bytes) were copied and hash-verified
  under `research_archive/old_continuum_evidence_20261116/older_plot_files/`.
  Exact identity with the selection-era datcon arrays is not yet established.
  Extracted original diagnostics for all 1,360 selections and 1,372 original
  crossing witnesses from 645 modes / 135 scopes: boundary, r_cross and omega
  at the crossing. Against frozen current continua, 1,362 points are comparable
  (10 currently unavailable); 1,197 differ by <0.01% and 89 by >1%.
  These are sparse crossing constraints, not full-profile or centroid errors.
  The October 29 centroid-gap measurements already used overwritten continua
  for this group, so cannot serve as original selection-era boundary values.
  CSVs, copied graphics, conversion previews and reproduction scripts are in
  the same archive. This qualifies the earlier statement about missing original
  profiles: full raw datcon arrays remain missing, but old plots and numerical
  witnesses survive. No catalog labels or HDF5 eligibility were changed.
- **November 16 continuum backup completed:** at the user's request, copied
  all **1,990 live datconN files** from the 199 populated inventory cases,
  plus 149 current-training and 160 August-20-training copies (**2,299 files**).
  Independent regular-file copies are read-only under
  `research_archive/continuum_backup_20261116/20261116T211332Z/`, with separate
  `live`, `training`, and `training_2026_08_20` trees. Every copy was checked
  for concurrent source writes and verified by SHA-256; the compressed archive
  was reopened and every member verified. No copy errors or unstable reads.
  `verified_selection_continua.csv` identifies all **1,201 selection-matched
  continuum scopes** from the HDF5 (1,170 live + 31 training), covering its
  4,517 mode targets; none changed before backup. All other available continua
  were also preserved without claiming selection-era identity. Manifest,
  archive, coverage receipt and copy script are retained beside the files.
  Raw NOVA inputs were not modified. Future NOVA-C overwrites cannot propagate
  into these independent copies; this is a local backup, not off-site storage.
- **November 16 HDF5 generated:** user approved both `r` and `psi_N=r²`.
  Added `src/nova_equilibrium.py`, `scripts/export_nova_surrogate.py` and
  [schema/usage documentation](surrogate_hdf5.md). The local product is
  `research_archive/nova_surrogate_export_20261116/nova_surrogate_v1.h5`
  (51,277,296 bytes): all 200 inventory IDs, 199 populated 13-scalar sets,
  1,974 available continua and 4,517 verified selected mode targets.
  Of 2,000 case/n scopes, 1,201 have verified K/mode targets, 638 have
  unmatched selected pairs, 104 have an unverified zero-GOOD continuum,
  31 lack an installed n selection, 16 remain held and 10 belong to the
  empty case. Unknown K/presence is -1, never a false no-TAE label.
  Frequencies stay in native NOVA normalization; mode centroids/RMS widths
  are recomputed separately in both coordinates. No 2D amplitudes or lobe
  counts are generated. Ten focused tests and HDF5 reopening validation
  pass; the density spline agrees with original Fortran for all 199 cases
  to 3.35e-10 absolute normalized density. Companion coverage CSV, summary,
  source-pair preflight and independent checks stay with the artifact.
  Next: reconcile changed inputs before expanding verified K/mode coverage;
  lobe-count definition and future physical-m amplitude export remain open.
- **November 16 density source selected:** the user chose reconstructed
  NOVA `iden=4` axial mass density for all 199 populated surrogate cases;
  `rho_axis_kg_m3` is already saved in the November 15 scalar audit. Use the
  same NOVA reconstruction for the normalized profile and density
  ratio at psi_N=0.5. NOVA-C output remains comparison evidence, and waiting
  for the remaining NOVA-C runs is unnecessary. This follows the inspected
  NOVA code path and confirmed database convention; historical executable
  identity is not independently established for every run. The approved
  13-parameter HDF5 implementation is described in [the schema](surrogate_hdf5.md).
- **November 16 NOVA-C central-density diagnostics:** found the Alfvén-speed/
  central-density pair in 5,338 `N1`–`N10/outb_*` logs covering all 89 NOVA-C
  cases; both values agree exactly within each case. Speed is in cm/s;
  density is in proton-mass units per cm^3, convertible to kg/m^3. Source
  and data reveal a small NOVA/NOVA-C difference: NOVA iden=4 includes
  thermal NH, while the inspected NOVA-C sum uses NMINI_H, zero in all 89
  cases. That recipe matches all logged densities to printed precision.
  NOVA-C values are lower than the earlier NOVA reconstruction by up to
  0.451813% (median 0.282130%). Keep the scales separate pending confirmation
  of NOVA-C's thermal-H treatment; full cross-code profile agreement has not
  been established. Raw/SI values, comparisons, hashes, and validation are
  in `research_archive/outb_density_audit_20261116/`; see the
  [source map](surrogate_input_sources.md). No normalization or selection
  changes were made, and no HDF5 was generated.
- **November 16 saved density profiles:** audited `N1`–`N10` `otaep` files
  following the user's identification of `RHOPRF profile`. The 12:42 EST
  snapshot has 749 complete blocks in 89 cases; all contain 201 positive
  values, matching active `gridparam` and `NNSURF`, with first value 1.0.
  All 88 cases with multiple copies agree exactly at printed precision;
  E204118F01t035 has one copy. Across 1,990 N directories, another 12 files
  are header-only and 1,229 `otaep` files are absent; 110 populated cases have
  no complete profile. The user confirmed that `otaep` is NOVA-C output and
  only 89 cases have been processed with NOVA-C. It saves normalized density
  to three significant digits where available; its physical axial scale was
  subsequently found in `outb_*` as described above. Coverage, extracted
  arrays, hashes, script, and independent
  reparse validation are in `research_archive/otaep_density_audit_20261116/`;
  the [source map](surrogate_input_sources.md) documents the findings.
  No HDF5 or final 13-parameter table was generated. Next: retain run/grid
  correspondence when incorporating saved profiles; use the confirmed
  TRANSP processing where reconstruction is needed. Existing holds remain.
- **November 15 scalar-input availability:** all ten inputs requested for
  Alvaro's continuum-first surrogate are recoverable for all 199 populated
  cases; only the empty `nstxu_202806` inventory entry is missing. The
  [source map](surrogate_input_sources.md) records sources and definitions.
  Midplane geometry agrees with mapped/TRANSP references within 1e-9 m;
  mapped q/pressure copies agree exactly. Species-density processing matches
  original Fortran helpers for all 199 cases. Retain separate axis and peak
  values: sampled pressure/density peaks exceed axis values in 22/77 cases.
  The audit records near-axis q/p proxies and grid/knot extrema explicitly;
  final axis/spline extraction remains loader work. Per-case CSV, hashes,
  script, and validation are in
  `research_archive/surrogate_scalar_availability_20261115/`. No production
  code or selections changed; existing input holds remain. Next: agree the
  export schema and assemble matched scalar/continuum data by `n_tor`.
  Follow-up: the user tentatively prefers axial p_0/rho_0; retain peaks
  separately. Linear near-axis q extrapolation changes q by at most 4.24e-5
  (maximum relative change 0.000880%) across all 199 cases, with small local
  quadratic/cubic differences. TRANSP shaping is also stored, while the
  audit uses shaping derived from the mapped LCFS. None of the 199 NetCDF
  schemas contains mass density; the November 16 audit above additionally
  identifies saved normalized profiles in `otaep` for 89 cases.
  Follow-up measurements and reproduction script are in the same archive.
- **November 13 surrogate example plots:** prepared the five human-GOOD TAEs
  for `nstxuE202855A01t020/N1` as a comparison sheet and individual three-panel
  pages (signed radial harmonics, signed m/r contour, continuum/frequency;
  no m-spectrum). All training mode/continuum fingerprints match retained
  review evidence; matching run metadata verifies physical m=0..13. Figures
  use the reviewed training continuum because the live `datcon1` has changed.
  PDF, PNGs, selected-mode CSV, provenance, and reproduction script are in
  `research_archive/alvaro_tae_examples_20261113/`. No scientific selections
  or shared plotting/processing code changed. The follow-up higher-n package
  in `research_archive/alvaro_tae_high_n_20261113/` includes all six reviewed
  GOOD modes at N8 for the same shot, with verified physical m=0..89 and
  matching training fingerprints. It also uses the reviewed continuum
  (`datcon8` differs in the live database), and preserves the original N1
  package. N8 modes C/D differ by only 1.17% in frequency but have markedly
  different radial location and width.
- **November 13 surrogate conventions confirmed:** the user confirmed across
  the entire database that physical poloidal m can start at zero or a negative
  value, the frequency reference uses the vacuum toroidal field at the
  midplane geometric center and the axial TRANSP species mass sum, and
  `iden=4` builds and normalizes the species-based density profile including
  beam ions and impurities. `qa`, `pa`, and `ga` are the mapped profiles used
  by NOVA and agree with `mapdsk`/`mpout1`. Density prescription and mapped
  profile identity are resolved. The [source map](surrogate_input_sources.md)
  replaces the earlier index-m, electron-density, and global-LCFS-midpoint
  assumptions. Next: implement extraction with physical m labels, the
  confirmed density processing, and the midplane reference geometry.
- **November 10 surrogate input discovery:** the
  [source map](surrogate_input_sources.md) traces the three printed q sections,
  mapped profiles/geometry, TRANSP data, and per-shot frequency normalization.
  All 199 populated inventory cases have the main equilibrium files;
  NetCDF/text q and pressure agree exactly, and binary `mpout1` also matches
  q/pressure/g. E55's `outequ` is truncated despite matching mapped profiles.
  Five mapped cases contain nonzero rotation data. The density recipe and
  mapped-profile interpretation were initially open and were resolved by the
  November 13 confirmation above. Detailed discovery evidence is in the
  ignored research archive. Scientific selections and processing code are
  unchanged; retain per-run settings and file provenance during extraction.
- **October 29 poster preparation:** selected the mixed-routing example
  E204957F05t034/N7/3141 and produced total-count and G/non-G parameter tables.
  The mixed example has a fresh isolated v13 check; a full EAE-gap reconstruction
  is still needed before asserting a double-gap eigenmode. All poster assets,
  per-mode measurements, and provenance remain local archives.
- **November 9 repository cleanup:** generated outputs and detailed experiment
  artifacts are removed from Git tracking while remaining locally intact.
  Audit summaries, essential review/override/input records, and tested audit
  utilities remain tracked. The required 2,327 training fingerprints now live
  in `catalogs/evidence/` rather than depending on an ignored output file.
  The catalog, labels, scientific configurations, and model checkpoints are
  unchanged. Tracking shrank from 1,949 to 392 files; all removed paths remain
  locally intact. All 228 tests pass in a temporary checkout containing only
  retained files. Catalog regeneration exactly reproduces all 8,943 published
  rows, and all 2,871 review mappings are unchanged. Retained Markdown links,
  ignore rules, scientific-file checksums, and the 43-file poster backup were
  verified. The test run used NumPy 2.x and a temporary PyYAML dependency for
  the skill validator; the installed environments were not modified.
- Next: maintain this compact snapshot, register future fingerprinted review
  evidence before rebuilding the catalog, and resolve the input holds and
  scientific conflicts above when authorized. Store detailed new run records
  in the ignored research archive rather than expanding this document.
