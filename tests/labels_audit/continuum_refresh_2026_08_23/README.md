# Non-G continuum-refresh suspect modes

This directory contains an explicitly non-blind adjudication shortlist for
the refreshed `nstx_120113` and `nstx_135388` continua.

## Files

- `nonG_suspect_modes.csv`: viewer-ready list containing the relative mode
  path, current adjudicated label, and selection reason.
- `nonG_suspect_mode_details.csv`: companion provenance table containing the
  prior label, review-confidence source and reason, full-precision old/new
  split scalars, absolute scalar changes and ranks, and old/new gap regions.
  Use this table to examine why each mode entered the shortlist.
- `nonG_suspect_label_changes.csv`: the five explicit old-to-new human
  decisions made after inspecting the refreshed continua. All five are
  low-confidence, non-blind decisions with prior labels seen.

This is intentionally a non-blind review aid. Initial labels for 44 modes
came from `tests/labels_audit/labels_human_review_clean.csv`. The newly admitted
`nstx_135388/N4/egn04w.1922E+03` mode was outside the old TAE-like list, so its
preserved BAD label comes from
`training_labels/old_4shots_mixed_labels/all_modes.csv`. The five subsequent
decisions in `nonG_suspect_label_changes.csv` are now reflected in both the
viewer list and the active clean human-review source.

## Selection rule

The universe is the current TAE-like membership in
`/p/hym/ebelova/NOVA/data_mixed`.

1. Include every retained low-confidence/adjudication-sensitive review mode:
   - four `nstx_120113` modes: the two low-confidence sealed rows plus B131
     and B149, which the post-discussion project summary retained as
     low-confidence decisions;
   - all 26 low-confidence rows in the final non-blind
     `nstx_135388_codex_policy_v2_labels.csv` adjudication.
2. For each shot with a nonzero upper-gap scalar change, include the top 10
   current TAE modes by absolute full-precision change in `signed_delta` and
   the top 10 by absolute change in `fraction_below_upper2`.
3. Take the union and retain each mode once, recording every applicable
   reason.

The resulting list has 45 modes: 4 from `nstx_120113` and 41 from
`nstx_135388`. Thirty are confidence-selected and 16 are scalar-selected,
with one mode in both groups. Although all ten numbered `nstx_120113`
continuum files differ byte-for-byte between the old and new trees, their
parsed upper-bound split scalars are exactly unchanged at full precision for
all 174 current TAE modes. Therefore that shot has no scalar-ranked entries.

Before the refreshed-continuum review, the counts were 20 GOOD, 24 BAD, and 1
SKIP. After the five decisions, they are 23 GOOD and 22 BAD.

The older sealed, policy-v2, frozen 2026-08-20, and v2 training artifacts were
not rewritten. These decisions are incorporated in the rebuilt-database
`training_labels/tae_like_v3.csv` list.

## Interactive inspection

```bash
export NOVA_DATA=/p/hym/ebelova/NOVA/data_mixed
PYTHONPATH="$NOVA_REPO/src" python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nonG_suspect_modes.csv" \
  --base_dir "$NOVA_DATA"
```

Signed harmonics are shown by default. Use the interactive `a` key only for a
secondary absolute-amplitude view.

## S31 whole-shot review

`nstxuG133964S31_human_labels.csv` and its deduplicated `_clean.csv` companion
contain the complete refreshed-continuum review of all 76 current S31 TAE-like
modes. Every decision is BAD, with no duplicates, missing paths, extras, or
SKIPs. The old v2 list contained 74 of these modes, also all BAD. The two
newly covered BAD paths are:

- `nstxuG133964S31/N5/egn05w.1581E+02`
- `nstxuG133964S31/N10/egn10w.1678E+02`

The old and new split manifests both contain all 76 paths, so these are
previously omitted training labels rather than continuum-driven membership
additions. The complete 76-row review is included in
`training_labels/tae_like_v3.csv` with current regenerated split scalars.

## H47 whole-shot review comparison

`nstxuG142301H47_human_labels.csv` and its byte-identical `_clean.csv`
companion contain the complete refreshed-continuum review of all 178 current
H47 TAE-like modes: 12 GOOD and 166 BAD, with no duplicate paths or SKIPs.
The old v2 training list contains 169 H47 rows: 9 GOOD and 160 BAD.

The initial comparison exposed five disagreements. During the check,
`N10/egn10w.1403E+02` was identified as a mistaken BAD entry and corrected to
GOOD, matching its old v2 label. The correction is retained separately in
`nstxuG142301H47_post_comparison_changes.csv` as an explicitly non-blind
post-comparison change.

After that correction, 165 of the 169 shared labels agree and four disagree,
for 97.63% agreement and Cohen's kappa of 0.7876. The retained changes are
three BAD-to-GOOD and one GOOD-to-BAD decisions. The user identified all three
retained BAD-to-GOOD modes as extremum-localized types; that rationale is
recorded in `nstxuG142301H47_label_disagreements.csv`. This viewer-ready list
contains the four remaining disagreements. Its `label` column deliberately holds
the old v2 label so that `viz/view_modes_csv.py` displays the prior decision;
`new_label` records the final review decision. The complete finalized H47
review is included in `training_labels/tae_like_v3.csv`; the disagreement
list remains an audit-only inspection artifact.

The other nine current review paths have no old v2 training label. Six were
already members of the old 175-mode TAE split but were omitted from its
169-row training subset:

- `nstxuG142301H47/N2/egn02w.2025E+02` (new review: BAD)
- `nstxuG142301H47/N5/egn05w.1372E+02` (BAD)
- `nstxuG142301H47/N6/egn06w.2005E+02` (GOOD)
- `nstxuG142301H47/N6/egn06w.9086E+01` (BAD)
- `nstxuG142301H47/N7/egn07w.1317E+02` (BAD)
- `nstxuG142301H47/N8/egn08w.1696E+02` (BAD)

Three are continuum-driven additions to the refreshed 178-mode TAE split:

- `nstxuG142301H47/N1/egn01w.3450E+02` (new review: BAD)
- `nstxuG142301H47/N3/egn03w.3059E+02` (BAD)
- `nstxuG142301H47/N8/egn08w.1194E+02` (BAD)

Inspect the four remaining disagreements with:

```bash
export NOVA_DATA=/p/hym/ebelova/NOVA/data_mixed
PYTHONPATH="$NOVA_REPO/src" python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstxuG142301H47_label_disagreements.csv" \
  --base_dir "$NOVA_DATA"
```

The v3 merge uses all 178 final H47 labels and the corresponding current split
scalars. Post-merge validation confirmed exact review/split coverage, 12 GOOD
and 166 BAD rows, portable relative paths, consistent family fields, no errors
or SKIPs, and all H47 mode files resolving under the canonical data root.

## Y93 whole-shot review comparison

`nstxuG142301Y93_human_labels.csv` and its byte-identical `_clean.csv`
companion contain the complete refreshed-continuum review of all 113 current
Y93 TAE-like modes: one GOOD and 112 BAD, with no duplicate paths, SKIPs,
missing split modes, extra review rows, or missing mode files.

The old v2 training list exactly covers the former 106-mode Y93 TAE split: one
GOOD and 105 BAD. The initial comparison found one disagreement at
`N9/egn09w.1539E+02`. During the check, the user retained its old GOOD label.
That explicitly non-blind correction is recorded in
`nstxuG142301Y93_post_comparison_changes.csv`. The final review therefore
agrees with all 106 old labels (100% agreement, Cohen's kappa 1.0), and
`nstxuG142301Y93_label_disagreements.csv` now contains only its header.

The other seven current modes without old v2 labels are exactly the known
continuum-driven additions to the refreshed TAE split, and all are now BAD:

- `nstxuG142301Y93/N1/egn01w.1894E+02`
- `nstxuG142301Y93/N1/egn01w.1937E+02`
- `nstxuG142301Y93/N1/egn01w.2146E+02`
- `nstxuG142301Y93/N2/egn02w.2123E+02`
- `nstxuG142301Y93/N3/egn03w.1635E+02`
- `nstxuG142301Y93/N3/egn03w.2200E+02`
- `nstxuG142301Y93/N3/egn03w.2513E+02`

These seven modes are collected in the viewer-ready
`nstxuG142301Y93_new_tae_modes.csv`. Its `label` column contains the current
review label so the viewer displays BAD while they are reinspected.

Inspect the seven new modes with:

```bash
export NOVA_DATA=/p/hym/ebelova/NOVA/data_mixed
PYTHONPATH="$NOVA_REPO/src" python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstxuG142301Y93_new_tae_modes.csv" \
  --base_dir "$NOVA_DATA"
```

The complete finalized Y93 review is included in
`training_labels/tae_like_v3.csv` using current split scalars. Post-merge
validation confirmed exact review/split coverage, one GOOD and 112 BAD rows,
portable relative paths, consistent family fields, no errors or SKIPs, and all
Y93 mode files resolving under the canonical data root.

## Q62 whole-shot review comparison

`nstxuG121123Q62_human_labels.csv` and its byte-identical `_clean.csv`
companion contain the complete refreshed-continuum review of all 249 current
Q62 TAE-like modes: 16 GOOD and 233 BAD, with no duplicate paths, SKIPs,
missing split modes, extra review rows, or missing mode files.

The old v2 training list exactly covers the former 241-mode Q62 TAE split: 12
GOOD and 229 BAD. The initial comparison had 230 agreements and 11 differences
(95.44%, Cohen's kappa 0.4528): seven GOOD-to-BAD and four BAD-to-GOOD.
`nstxuG121123Q62_initial_label_disagreements.csv` preserves that original
comparison unchanged.

After inspecting the 11 cases together, the user adjudicated all of them GOOD
with low confidence. They share smooth resolved structure, a small-r continuum
crossing, and no resonant-like amplitude spike. The seven precheck BAD labels
were changed to GOOD in the final review and recorded separately in
`nstxuG121123Q62_post_comparison_changes.csv` with `prior_seen=true`.
`N9/egn09w.2152E+02` remains GOOD as originally reviewed.

`nstxuG121123Q62_uniform_low_conf_good_modes.csv` is the viewer-ready complete
11-mode adjudication set, with the final GOOD label, low confidence, shared
rationale, precheck and old-v2 labels, and current split scalars. After the
adjudication, 237 of 241 shared labels agree (98.34%, Cohen's kappa 0.8485).
The four remaining differences are all old-BAD-to-final-GOOD and are retained
in `nstxuG121123Q62_label_disagreements.csv`, whose displayed `label` remains
the old v2 decision.

The other eight current modes without old v2 labels are exactly the known
continuum-driven N1 additions to the refreshed TAE split. All eight are
labeled BAD and classified in the mixed region. They are collected in the
viewer-ready `nstxuG121123Q62_new_tae_modes.csv` with their current review
labels and regenerated split scalars.

Inspect the complete 11-mode low-confidence family with:

```bash
export NOVA_DATA=/p/hym/ebelova/NOVA/data_mixed
PYTHONPATH="$NOVA_REPO/src" python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123Q62_uniform_low_conf_good_modes.csv" \
  --base_dir "$NOVA_DATA"
```

Inspect the eight new modes by substituting
`nstxuG121123Q62_new_tae_modes.csv` in the same command.

The complete finalized Q62 review is included in
`training_labels/tae_like_v3.csv` using current split scalars. Post-merge
validation confirmed exact review/split coverage, 16 GOOD and 233 BAD rows,
all 11 low-confidence GOOD adjudications, portable relative paths, consistent
family fields, no errors or SKIPs, and all Q62 mode files resolving under the
canonical data root.

## nstxu_204202 mode-set refresh transfer

The current `nstxu_204202` TAE-like split has 140 modes and is a strict subset
of the 275 paths labeled for this shot in `tae_like_v2_nonG.csv`. All 140
shared mode files are byte-identical to their copies in the frozen
`data_mixed_2026_08_20` database, so their exact old labels can be transferred
without a new visual review. The staged component
`nstxu_204202_transferred_labels.csv` contains those 140 paths: 62 GOOD and 78
BAD. It preserves current split ordering and uses the regenerated current
split scalars rather than the differently formatted scalar strings in v2.

There are no genuinely new modes in the current TAE-like split. The
label-free `nstxu_204202_new_tae_modes.csv` therefore contains only its header;
no new-mode visual inspection is needed.

The other 135 old labeled paths, comprising two GOOD and 133 BAD labels, are
absent from the current mode tree. They are excluded from the transferred
component and retained with their old label/scalar provenance in
`nstxu_204202_quarantined_old_labels.csv`, with quarantine reason
`absent_from_current_mode_tree`.

The complete 140-row transferred component is included in
`training_labels/tae_like_v3.csv`. Post-merge validation confirmed exact
current split coverage, 62 GOOD and 78 BAD rows, portable relative paths,
current split scalars, consistent family fields, no errors, and all 140 mode
files resolving under the canonical data root. The 135 quarantined paths are
excluded from v3.

## nstx_141711 whole-shot review preparation

The recalculated `nstx_141711` mode payloads require a complete current-shot
review rather than an old-label transfer. The regenerated TAE-like split has
158 unique modes across `N1` through `N10`; all 158 files exist and none of the
split rows has an error.

`nstx_141711_blind_manifest.csv` is the label-free review input generated from
the current split. It contains only portable paths and split-geometry metadata;
it has no human or classifier labels. `nstx_141711_blind_decisions.csv` is the
matching blank structured decision template. The interactive labeler should
write its independent binary review to the new
`nstx_141711_human_labels.csv` path with RF guidance disabled:

```tcsh
setenv NOVA_DATA /p/hym/ebelova/NOVA/data_mixed
setenv NOVA_REPO /p/hym/ebelova/NOVA/NOVA_modes

python "$NOVA_REPO/scripts/label_modes_fast.py" \
  "$NOVA_DATA/nstx_141711" \
  --pattern 'N*/egn*' \
  --data_dir "$NOVA_DATA" \
  --mode-list "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstx_141711_blind_manifest.csv" \
  --csv_out "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstx_141711_human_labels.csv" \
  --no-rf
```

The same command safely resumes after quitting. Do not compare with historical
labels or merge this shot into v3 until all 158 current modes are labeled and
the clean review has been validated against the manifest.

### Completed review and old-label comparison

The completed clean review exactly covers all 158 manifest modes: 75 GOOD and
83 BAD, with no SKIPs or missing files. The raw labeler output contains 159
rows because `N3/egn03w.5246E+02` was recorded GOOD twice; the clean output
deduplicates it to the same GOOD decision. The pre-comparison clean state is
frozen by `nstx_141711_human_labels_clean.csv.sha256` with SHA-256
`0264756d76e83d209390137261511af5baa63e092ebdef2aecc15b1dd13792d3`.
The binary labeler workflow does not capture the confidence and reason fields
required by the formal structured-review sealer, so the checksum records the
exact completed state used for this maintenance comparison.

Of the 158 current paths, 154 have an old v2 label. The refreshed review agrees
on 148 and differs on six: 96.10% agreement with Cohen's kappa 0.9221. The
differences comprise five old-GOOD-to-current-BAD decisions and one
old-BAD-to-current-GOOD decision. Three disagreement modes have changed mode
payloads relative to `data_mixed_2026_08_20`; the other three payloads are
byte-identical. All six pre-adjudication differences are preserved in the
viewer-ready `nstx_141711_initial_label_disagreements.csv`, where `label` is
the old v2 decision and `new_label` is the completed current review decision.
The initial full 154-path audit is retained in
`nstx_141711_initial_shared_label_comparison.csv`.

The four current TAE-like modes without old v2 labels are all N1 modes and all
were reviewed BAD. They are collected in `nstx_141711_new_tae_modes.csv`.
Conversely, 102 old labeled paths are outside the current TAE-like review: 101
files are absent from the current mode tree and one old-BAD file remains in the
tree but is not in the current TAE-like split. These old rows are retained in
`nstx_141711_quarantined_old_labels.csv` and excluded from any proposed merge.

Inspect the six disagreements on Flux (`tcsh`) with:

```tcsh
setenv NOVA_DATA /p/hym/ebelova/NOVA/data_mixed
setenv NOVA_REPO /p/hym/ebelova/NOVA/NOVA_modes

python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstx_141711_initial_label_disagreements.csv" \
  --base_dir "$NOVA_DATA"
```

### Post-comparison adjudication

The user retained the old labels for four of the initial disagreements and
kept the current review decisions only for two modes:

- `N7/egn07w.9318E+02`: final GOOD, old BAD;
- `N8/egn08w.1026E+03`: final BAD, old GOOD.

The four actual post-comparison changes are recorded in
`nstx_141711_post_comparison_changes.csv` with `prior_seen=true`; the complete
six-mode decision record is `nstx_141711_disagreement_adjudication.csv`. The
checksummed pre-comparison clean review remains unchanged.

`nstx_141711_human_labels_final.csv` contains the finalized complete current
shot: 79 GOOD and 79 BAD labels. Its SHA-256 sidecar records digest
`2a82d29d8d9d7905e563dbe47e95bc4f57fe35570731ff58f6f6b6757bffe48a`.
The final comparison agrees with the old label on 152 of 154 shared paths
(98.70%, Cohen's kappa 0.9740). The current
`nstx_141711_label_disagreements.csv` and
`nstx_141711_shared_label_comparison.csv` reflect this final adjudicated state;
only the two explicitly retained current-review decisions remain different.

The finalized 158-row training component is retained in
`nstx_141711_final_training_component.csv` and is included in
`training_labels/tae_like_v3.csv`. Post-merge validation confirmed exact
current split coverage, 79 GOOD and 79 BAD rows, regenerated current scalars,
portable relative paths, consistent family fields, no errors or SKIPs, and all
158 mode files resolving under the canonical data root. All 102 quarantined
old-only paths remain excluded.

## K51 whole-shot review preparation

The recalculated `nstxuG121123K51` mode payloads require a complete
current-shot review rather than an old-label transfer. The regenerated
TAE-like split contains 152 unique modes across `N2` through `N10`; all 152
files exist and none of the split rows has an error.

`nstxuG121123K51_blind_manifest.csv` is the label-free review input generated
from the current split. It contains only portable paths and split-geometry
metadata, with no human or classifier labels.
`nstxuG121123K51_blind_decisions.csv` is the matching blank structured
decision template.

From the repository root on Flux, label the whole current TAE-like set using
`tcsh` with RF guidance disabled:

```tcsh
source configs/paths/nova_paths.flux.csh

python "$NOVA_REPO/scripts/label_modes_fast.py" \
  "$NOVA_DATA/nstxuG121123K51" \
  --pattern 'N*/egn*' \
  --data_dir "$NOVA_DATA" \
  --mode-list "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123K51_blind_manifest.csv" \
  --csv_out "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123K51_human_labels.csv" \
  --no-rf
```

The same command safely resumes after quitting. Do not compare with historical
K51 labels or merge the shot into v3 until all 152 current modes are labeled
and the clean review has been validated against the manifest.

### Completed K51 review and old-label comparison

The completed `nstxuG121123K51_human_labels_clean.csv` exactly covers all 152
manifest modes: 25 GOOD and 127 BAD, with no duplicate paths, SKIPs, or missing
files. Only the requested clean list was used for the comparison. Its frozen
SHA-256 sidecar records digest
`ae59e1386911d2c3d2dbba42e37b932f2bcb1e8eea714fc4bcdde2ba7f13ea1b`.
As with the other binary labeler reviews, the clean file does not contain the
confidence and reason fields required by the formal structured-review sealer,
so the checksum preserves the exact completed pre-comparison state.

Of the 152 current paths, 122 have an old v2 label. The refreshed review agrees
on 112 and differs on ten: 91.80% agreement with Cohen's kappa 0.7408. The
differences comprise six old-GOOD-to-current-BAD and four
old-BAD-to-current-GOOD decisions. All ten disagreement mode payloads changed
relative to `data_mixed_2026_08_20`. The viewer-ready differences are in
`nstxuG121123K51_initial_label_disagreements.csv`, where `label` is the old v2
decision and `new_label` is the clean current review decision. The complete
initial 122-path comparison is retained in
`nstxuG121123K51_initial_shared_label_comparison.csv`.

Thirty current TAE-like modes have no old v2 label: two GOOD and 28 BAD. They
are collected in `nstxuG121123K51_new_tae_modes.csv`. Conversely, all 86
old-only labeled paths are absent from the current mode tree; their old labels
(three GOOD and 83 BAD) are preserved in
`nstxuG121123K51_quarantined_old_labels.csv` and excluded from any proposed
merge.

Inspect the ten disagreements on Flux (`tcsh`) with:

```tcsh
source configs/paths/nova_paths.flux.csh

python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123K51_initial_label_disagreements.csv" \
  --base_dir "$NOVA_DATA"
```

### K51 post-comparison adjudication

The user changed only `N9/egn09w.3938E+02` from the clean-review GOOD decision
to BAD, matching its old v2 label, and retained the clean-review decisions for
the other nine initial differences. The one actual post-comparison change is
recorded in `nstxuG121123K51_post_comparison_changes.csv` with
`prior_seen=true`; `nstxuG121123K51_disagreement_adjudication.csv` records all
ten final decisions. The checksummed clean pre-comparison review remains
unchanged.

`nstxuG121123K51_human_labels_final.csv` contains the finalized complete
current shot: 24 GOOD and 128 BAD labels. Its SHA-256 sidecar records digest
`3d75f3019c8e9bd2f2abd785f3786439a6f5defd8607de587434a83f223ac766`.
Final agreement with old labels is 113/122 shared paths (92.62%, Cohen's kappa
0.7631). The current `nstxuG121123K51_label_disagreements.csv` and
`nstxuG121123K51_shared_label_comparison.csv` reflect this final adjudicated
state and retain nine differences.

The finalized 152-row training component is retained in
`nstxuG121123K51_final_training_component.csv` and is included in
`training_labels/tae_like_v3.csv`. Post-merge validation confirmed exact
current split coverage, 24 GOOD and 128 BAD rows, regenerated current scalars,
portable relative paths, consistent family fields, no errors or SKIPs, and all
152 mode files resolving under the canonical data root. All 86 quarantined
old-only paths remain excluded.
