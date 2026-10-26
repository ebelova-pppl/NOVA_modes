# Common accepted-mode catalog

[accepted_modes.csv](accepted_modes.csv) is the compact, 12-column catalog
assembled from the direct per-shot `good_tae_final.csv` files in
`$NOVA_REPO/../sort_outputs/`. It preserves the installed final selection;
training labels and review records supply review status, not membership.

The October 26, 2026 snapshot contains **8,943 modes** from 190 nonempty
shot lists. All 198 installed lists were checked, including eight empty ones.
There are 8,941 TAE and two mixed modes, with nine applied manual GOOD
overrides. All rows retain `final_decision=GOOD`. No EAE or BAE modes occur in
these final GOOD lists; routed family lists are not appended automatically.

| Column | Definition |
| --- | --- |
| `path` | Portable `shot/N#/filename`, relative to `$NOVA_DITW_ROOT`, the raw-mode database. It is not relative to the catalog or sorting-output directory. |
| `shot` | Full shot/case directory name, retaining equilibrium variants and time slices. |
| `family` | `TAE`, `mixed`, `EAE`, or `BAE`, taken from the saved family routing. |
| `ntor` | Toroidal mode number; checked against the N directory. |
| `omega` | Original exported NOVA frequency, with source precision preserved. No unit conversion is applied; this column is not asserted to be Hz or kHz. |
| `rad_loc` | Normalized radial energy centroid, not the peak radius. |
| `rad_width` | RMS radial width in normalized radius, not FWHM. |
| `rule_config` | Production preset name from the original run, e.g. `tae_rules_production_v13`. |
| `overall_rule_severity` | Original rule severity, rounded for display. Lower means farther from rejection thresholds, not a probability of physical validity. Blank means unavailable. |
| `manual_override` | `yes` only for an applied manual GOOD override; otherwise `no`. |
| `final_decision` | Final selected classification. Preliminary rule REVIEW/BAD is not substituted here. |
| `review_status` | `reviewed`, `pending`, or `unknown`, using the evidence policy below. |

Rows are ordered by full shot name, numeric `ntor`, frequency and path.
`rad_loc`, `rad_width` and severity use four significant digits. Use the
unrounded original audit for numerical threshold comparisons. Manually
accepted modes can legitimately retain severity above one.

## Review evidence

| Status | Current modes | Meaning |
| --- | ---: | --- |
| `reviewed` | 865 | Recorded GOOD review applies to the same exported mode-plus-continuum fingerprint, or an applied manual GOOD override establishes the decision. |
| `pending` | 6,590 | 6,551 selections await visual review; another 39 conflict with the saved human training label and need adjudication. |
| `unknown` | 1,488 | No matching per-mode review record in the configured evidence sources. This does not mean nobody ever reviewed the mode. |

Training-shot review does count. Of their 577 selected modes, **538 match
human GOOD labels** and are `reviewed`. The remaining **39 have human BAD
labels** but were selected by the older rules outputs. They remain in the
catalog, marked `pending`; [the discrepancy list](accepted_modes.review_conflicts.csv)
identifies them. No training labels or source selections were changed.

The original 39-shot comparison review covered disagreements, not every
selected mode. Only modes explicitly covered by those review dispositions
are marked reviewed on that basis. The 1,488 `unknown` entries are remaining
selections from that cohort. Its accepted-set export alone is not treated as
evidence that each mode was visually inspected. E202806A02t045's separately
recorded result review and the subsequent 14-shot and four-shot GOOD-list
reviews are also included. The October 25 five-mode diagnostic is attributed
to Codex in the review-source registry; it is not described as a human or
blind review.

[configs/catalog_review_sources.csv](../configs/catalog_review_sources.csv)
lists the explicit evidence files, decision columns, shot scopes and reviewers.
The canonical training labels omit fingerprints; the preserved September 13
training audit supplies them. A changed fingerprint cannot inherit an old
review status. Matching is against the saved final-list fingerprint; this
export does not reload or revalidate the current raw mode files.

## Provenance and regeneration

[accepted_modes.metadata.json](accepted_modes.metadata.json) records source
file hashes and counts, full internal rule versions, configuration hashes,
review evidence, conflicts and the catalog checksum. The source outputs use
v13 for 8,366 post-training modes, v1 for 406 training modes and v2 for 171
training modes. All 577 training-shot rows lack saved severity, so those
cells remain blank. No score is invented or recomputed.

From the repository root, using Python 3.10+:

```tcsh
python scripts/build_accepted_catalog.py \
  --sort-root ../sort_outputs \
  --output catalogs/accepted_modes.csv
```

The exporter uses the Python standard library and shared repository I/O and
input-validity helpers. It reads exactly one level of shot directories, checks
inventory coverage, selected GOOD membership, uniqueness, summary counts,
shot/n consistency, configuration identity and current invalid-input scopes.
It refuses inconsistent sources rather than silently dropping rows. It writes
the catalog and metadata after validation and replaces previous exports
atomically per file. CSV and metadata can be paired using the catalog checksum.
It also regenerates the separate conflict CSV. The original per-shot outputs,
raw data, labels and models remain unchanged.

When a new visual review is completed, add its fingerprinted decision file to
the review-source registry. Registry paths are relative to the repository;
explicit shot scopes prevent a cohort manifest from implying review of modes
outside its recorded review scope.
