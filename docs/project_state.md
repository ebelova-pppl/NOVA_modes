# NOVA mode classifier: current project state

Updated November 19, 2026. Selection counts below retain their October 10/26/29
snapshot dates; November 16–17 checks NOVA-C density outputs and implements
the surrogate HDF5 export and continuum restoration using the confirmed NOVA density prescription. The complete
prior chronology is preserved locally and in Git history as described in
[research records and recovery](research_archive.md).

## Purpose and current workflow

November 18: the CSV viewer now uses the shared repository import bootstrap,
so direct execution no longer needs `PYTHONPATH`. Flux setup documentation
distinguishes one-time NumPy/SciPy/Matplotlib installation from per-session
activation; these viewing/rules workflows do not require PyTorch. Startup
passed from another directory with `PYTHONPATH` unset and with PyTorch,
scikit-learn, and joblib imports blocked. Interactive display was not tested.
The lightweight setup command now selects conda-forge exclusively after Flux
reported an Anaconda `defaults` HTTP 429 registration block; no environment
installation or shared-base changes were performed by the assistant.

Select physically useful Alfvén eigenmodes from existing NOVA outputs for
NOVA-C, surrogate modeling, and digital-twin analysis. An optional MPI driver
now runs existing serial NOVA continuum executables in isolated project
directories. Eigenmode and growth-rate calculations remain separate workflows.

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

## Surrogate roadmap and legacy-seed priority

On November 17 the user supplied the October 14 long-term plan:

1. Use the approximately 9K selected old-NOVA modes from the 198 processed
   cases as an up-down-symmetric seed dataset. Alvaro's requested per-case/n
   inputs and targets have already been discussed and the first subset sent.
2. Recalculate the same equilibria with Pablo's superNOVA, which the user
   describes as parallel, more efficient and free of the up-down-symmetry
   restriction, with an improved workflow.
3. Apply the TAE classifier to separate numerical/continuum modes from the
   new output, checking its applicability to the superNOVA representation.
4. Retrain the surrogate on superNOVA frequencies and mode structures.
5. Expand equilibrium coverage through active learning and additional
   superNOVA runs; a connection to Julien's DATAMAK for new NSTX-U cases
   is under consideration, not yet arranged.

For the current seed, prioritize reconstructing the overwritten continua
with their historical settings and checking agreement against available
historical evidence. A comprehensive old-NOVA convergence campaign is not
a prerequisite for this provisional surrogate dataset. Modest documented
convergence limitations are acceptable at this stage; preserve existing
invalid-input holds and review metadata. Distinguish reconstructed continua
from recovered original bytes and from newly converged calculations.
Any expanded export must record that provenance and use a new version,
preserving Alvaro's delivered subset. Convergence and reproducibility checks
should be built into the superNOVA recalculation stage. The user subsequently
authorized continuum reconstruction in the project directory and a reusable
MPI driver. DiTw write access was confirmed on November 17 after the owner's
ACL update. The user then authorized backing up the overwritten continua and
publishing the 638 validated reconstructions there. The user subsequently
requested completion of the scalar-target HDF5, retaining both radial
coordinates and reconstructed NOVA mass density, with 2D structures and lobe
counts explicitly deferred. Additional zero-GOOD continuum reconstructions
for this export remain in the project archive; they have not been published
to DiTw.

### November 19 final legacy-seed release: HDF5 v5 and four DiTw continua

The user authorized releasing E205059A01t025, implementing the previously
approved R06 release, publishing the corrected continua, and updating HDF5.
The v5 archive is `research_archive/nova_surrogate_export_20261119_v5/`.
Its complete-case product contains **199 cases complete for n=1–10, 9,001
selected modes and 579 K=0 records**. All populated inventory cases are
included; only the empty `nstxu_202806` entry remains outside the handoff.
The full inventory product retains that empty entry explicitly (200 IDs).

E205059A01t025 was classified across all 1,346 saved modes using the frozen
production_v13 rules and matched N1/N2 continua. It has no invalid inputs,
no skipped radial-resolution gates and no manual overrides: 275 TAE-side
candidates yield 59 GOOD modes before duplicates and 58 selected. Its K
values for n=1–10 are **0, 1, 2, 4, 5, 7, 9, 10, 10, 10**. The N1/N2
decisions exactly reproduce the earlier paired check. New selections are
marked `review_status=pending`; release approval is not a per-mode GOOD
visual review. Their full-precision outputs and all input bytes are archived.

R06 has K=0 for all ten n records. N1/N2 carry the user's explicit review of
all 95 modes with matched continua. N3–N10 retain the hash-verified historical
rules selection: 33 TAE-side candidates rejected, 482 EAE-like modes routed
separately. The user accepted this zero selection after the direct higher-n
correspondence check. No new higher-n classification, visual review, or
convergence claim is implied.

Four files were backed up and atomically published as numbered `datcon1/2`
in DiTw: R06 m=-2..26/-2..47 and E205059A01t025 m=-5..16/-5..27. Their
unnumbered symlinks and access permissions were preserved. Originals are
verified in the prior calculation archives, `publication/before/` in v5,
and adjacent `N#/continuum_backups/<stamp>/` directories. Publication receipts
and SHA-256 checks are retained. Higher-n profiles were unchanged.

The release copies v4 and changes only the two cases' n records and release
metadata. All **77,695 unaffected original datasets** compare exactly with
v4, including all equilibrium parameters, reconstructed densities, and the
8,943 inherited selected modes. All **78,148 copied subset datasets** match
the full v5 file exactly. Independent checks verify every catalog frequency
and review flag, all 20 released live continua, all 58 new modes' frequencies,
fingerprints and moments in both coordinates, and all four published files
and backup sets. Both v4 products retain their original hashes.
Schema remains `nova-surrogate-v3` with
`data_release=v5`; r and psi_N, radial masks, native frequency normalization,
and deferred structures/lobes are unchanged. The release catalog
`accepted_modes_v5.csv` adds the 58 E-shot selections to the frozen catalog.
The shared historical catalog, installed old outputs and global guards are
preserved; they continue to protect uncorrected copies. The v5 release is
specific to its verified input versions, following the v4 release convention.
See [release evidence](../audits/surrogate_release_20261119_v5/README.md).

### November 19 E205059A01t025 N1/N2 production rules after continuum matching

The user confirmed that N1/N2 resonance alignment now looks good and flagged
N2 omega=3.27793 and 3.21680 as potentially good. A paired local run of the
frozen production rules on all 318 modes confirms that both are final GOOD
with either continuum. Their preliminary REVIEW / NO_GOOD_TEMPLATE evidence
is preserved; accept-as-good-v1 promotes them. No manual labels were added.
No installed historical E-shot production result was found; the original-
continuum run is a fresh control, not a recovered historical classification.

The existing duplicate filter groups `egn02w.1074E+02` (3.277934179885934)
and `egn02w.1035E+02` (3.2168040713397756). Their frequency separation is
1.88245% of the mean and ridge-envelope cosine similarity is 0.996586.
It selects the former, with lower overall rule severity 0.597852 versus
0.925114. The latter remains GOOD but is not the selected representative.

| Scope | Modes | TAE-like/mixed, original -> matched | GOOD before duplicates | Selected GOOD |
| --- | --- | --- | --- | --- |
| N1 | 266 | 49 -> 16 | 6 -> 0 | 1 -> 0 |
| N2 | 52 | 12 -> 10 | 2 -> 2 | 1 -> 1 |

All six original-control N1 GOOD modes, omega=4.684956..4.721413, move to
EAE-like routing with the matched continuum; this is not a new morphology
rejection. All 16 remaining N1 TAE-side modes are BAD; N2 has eight BAD and
the two GOOD candidates. EAE-like modes are not validated for EAE quality.

`research_archive/e205059_t025_rules_20261119/` contains both full sorter
outputs, commands/hashes, paired `comparison.csv`, `comparison_summary.json`,
and `user_candidates.csv` for viewing the two candidates with matched continua.
Both runs use production_v13, ruleset v25, grouped features v26, severity v2,
and datcon-monotonic-tail-v1. Validation confirmed 318 unique paired modes,
unchanged mode metadata, no invalid inputs or resolution-skipped gates, and
unchanged protected registries/catalogs. No RF/CNN or manual overrides were
used. This covers N1/N2 only; N3-N10 processing, publication, full-shot
disposition and HDF5 inclusion remain pending.

### November 19 E205059A01t025 N1/N2 internally matched continuum calculations

The user authorized continuum recalculation for this final populated held
shot, matching the eigenmode internal harmonic ranges. Repeated out_go
headers agree on saved/internal m=-3..14/-5..16 for N1 and
-3..25/-5..27 for N2. All 266/52 mode files and both live continua still
match the October 10 input hashes. There are 249/45 frequency matches with
nonempty singularity lists and 17/7 complete empty lists; the latter are
retained explicitly, not assigned invented resonances.

Both small local continuum executables reproduce the saved datcon files
byte for byte; their emitted namelists establish old continuum ranges
0..14 and 0..25. Their compiled capacities are only 15/26, below the
requested internal sizes 22/33. An initial larger R06 executable failed
reading the E-shot TRANSP format and produced no continua; both failed
receipts are preserved. The E shot's existing N3 executable has capacity 37
and successfully handles both ranges. No source changes or compilation were
needed. It also reproduces both original continua byte for byte at their
original ranges, establishing four successful baseline reproductions.

Flux jobs 2561879/2561880 produced six successful controls/calculations in
total plus the two discarded donor-reader failures. The final comparison
uses the same E N3 executable, equilibrium/TRANSP inputs, gamma=1.66666666,
iden=4, ksound=1 and 201-point grid; only lmin/lmax change. Output qprofile
files are identical. A total of 354 input/source files are backed up and
hash-verified, including all 318 modes; live membership and bytes are unchanged.

For the fixed historical TAE-like/mixed cohort, nearest logged-singularity
offsets above two grid intervals (Delta r=0.005, 0.03<=r<0.75) change:

| Scope | Original | Matched internal basis | Median absolute offset, grid intervals |
| --- | --- | --- | --- |
| N1 | 26/26 crossings | 2/39 crossings | 14.590 -> 0.336 |
| N2 | 6/12 crossings | 0/10 crossings | 1.985 -> 0.742 |

Crossing counts can change; this is a nearest-log comparison, not a
branch-identified resonance match. Raw and viewer-processed summaries agree.
The two remaining flags are both N1 `egn01w.4306E+02`: new crossings at
r=0.73202/0.73745 bracket its visible signed-mode spike near 0.735, whereas
the nearest listed singularity is 0.805. The plot supports local correspondence
despite the missing log counterpart. Other examples show old crossings moving
from smooth mode lobes onto the existing sharp features. No new mode-quality
decisions or harmonic-convergence claims are made.

Artifacts are under `research_archive/e205059_t025_internal_harmonics_20261119/`.
The requested profiles are `matched_internal/cases/nstxuE205059A01t025/`
`N1/datcon1` and `N2/datcon2`. The archive contains receipts, settings,
`README.txt`, `alignment_examples.png/.pdf`, `remaining_n1_example.png/.pdf`,
and absolute-path CSVs `matched_modes.csv` (318), `original_modes.csv` (318),
and `historical_tae_candidates.csv` (61). These work with the existing viewer
without --base_dir; the candidate list is historical routing, not new labels.
Physical m labels in static plots use the verified saved offset -3; the
viewer's contour ordinate remains the stored harmonic index.

No DiTw publication, eigenmode rerun, hold-registry release or HDF5 update
was performed. The subsequent user correspondence review and local N1/N2
sorting are recorded above; the R06/E-shot release and export remain deferred.

### November 19 R06 direct higher-n correspondence check: range difference alone is insufficient

Following this check, the user agreed that R06 can be released and asked to
review the last nonempty held shot, `nstxuE205059A01t025`, before updating the
HDF5 or related release products. R06 is accepted in principle as a zero-good-
TAE case with the existing review/selection provenance; implementation is
deferred until that final-shot check. The empty `nstxu_202806` inventory entry
does not count as an outstanding populated shot. No release files or shared
registries have been changed by this agreement.

The user challenged treating different stored continuum/internal m ranges as
proof of displaced resonances and noted that such differences may occur in
many database cases. The direct follow-up supersedes the earlier recommendation
to require matching all eight higher-n continua before any R06 release.
Do not impose a database-wide recalculation or new hold from range metadata
alone; assess the actual correspondence and the legacy-seed scope.

R06 has no accepted catalog TAEs. Before the September hold, the rules selected
only one mode, at N2; the user's November 19 rejection of all N1/N2 modes
includes it. The 33 historical N3–N10 TAE-like/mixed candidates were all
rule-rejected. This is a selection/review statement, not a new independent
visual classification of all 515 higher-n modes or proof of physical absence.

All 515 higher-n modes were checked against complete frequency-matched logs.
For native continua and 0.03<=r<0.75, 222/228 N3–N9 crossings lie within two
grid intervals of the nearest logged singularity (Delta r=0.005). The six
exceptions are 2/25 at N3, 1/63 at N4, 1/16 at N5, 1/30 at N6, 1/2 at N7,
0/48 at N8 and 0/44 at N9. In the historical TAE/mixed cohort the total is
4/66 exceptions. Nearest-log offsets are not branch-identified spike shifts.

N10 has 65/178 native interior crossings beyond two intervals, but focused
signed-profile inspection shows why interpreting this as a corresponding
spike displacement is unsafe. For N10 `egn10w.1939E+02`, upper/lower crossings
at r=0.71783/0.73085 lie close to visible m=99/100 peaks at 0.720/0.730;
the nearest listed singularity is instead at 0.885. For `egn10w.2146E+02`,
crossings at 0.71928/0.73121 likewise lie near those sampled peaks. The N7
exception has a strong peak near 0.080 and a crossing at 0.07507 despite a
nearest log entry at 0.125. These observed local agreements do not identify
every resonance, certify convergence, or establish the cause of missing log
counterparts. No systematic higher-n spike/crossing shift has been established.

N10 also needs explicit raw/processed distinction: the existing shared viewer
tail treatment removes the steep outer branch rise near r~0.72, leaving only
28 interior crossings, all within two intervals of a logged singularity.
Thus disappearance from that processed comparison is not evidence that an
offset was corrected. N3–N9 raw and processed interior statistics agree.
No preprocessing was changed in this check.

Artifacts: `research_archive/r06_higher_n_crossings_20261119/`, including
`README.txt`, `analysis.json`, raw/processed `summary.csv`, `crossings.csv`,
`focused_correspondence.png/.pdf` and per-n examples. Inspection followed the
visual TAE skill as an explicitly non-blind diagnostic; prior labels were
already known and no new labels were assigned. No NOVA calculation, DiTw
write, registry edit or HDF5 update occurred. Original whole-shot exclusion
metadata remain in place pending the release decision; metadata alone are
not grounds for requiring further higher-n continuum runs.

### November 19 R06 user review and remaining higher-n checks

After viewing all 60 N1 and 35 N2 modes with the internally matched continua,
the user judged all 95 numerical junk and proposed releasing the shot unless
n>2 has additional problems. These decisions are recorded as user-reviewed
BAD modes and K=0 for the two calculated samples. Their complete membership
and hashes match the frozen calculation inputs and viewer copies. This
supersedes unresolved morphology review for these specific N1/N2 inputs;
it is not a statement that no physical TAE could exist in the equilibrium.

The remaining n=3–10 check found the same continuum-basis discrepancy:
N3 uses 0..66 instead of internal -2..68, N4 uses 0..87 instead of -2..89,
and N5–N10 use 0..100 instead of -2..102. All 515 higher-n modes have
frequency-matched complete singularity logs. All 610 live mode/continuum
pairs reproduce their September audit fingerprints exactly, so the higher-n
concerns have not been resolved by replacement inputs.

Historically, n=3–10 had 33 TAE-like/mixed candidates, all rejected by the
rules, and 482 EAE-like modes outside those TAE rejection decisions. These
are historical results with the old continua, not a fresh classification.
Ten modes at n=7–9 have their largest absolute xi_psi amplitude at the highest
stored harmonic m=100; one is the old N8 TAE-like candidate
`egn08w.1084E+02`. This is an observed cutoff-related review flag, not proof
of inadequate internal coverage or an automatically invalid equilibrium.

Evidence is under `research_archive/r06_user_review_20261119/`:
`user_reviewed_bad.csv` pins the 95 user decisions to mode and matched
continuum fingerprints; `review_and_remaining_evidence.json`, `scope_audit.csv`
and `mode_audit.csv` record the higher-n check. A task CSV for the eight
internal ranges and a 33-mode historical candidate list are prepared.
No additional NOVA calculation, new automatic classification, DiTw write,
registry change or HDF5 release was performed. This first metadata check
recommended further higher-n matching/review; the subsequent direct
correspondence check above qualifies that recommendation. N1/N2's reviewed
zeros should be carried into the next scope-specific release. The machine
whole-shot exclusion and delivered v4 file remain unchanged.

### November 19 R06 N1/N2 internal-harmonic comparison

The user requested the same continuum-matching check for nstxuG133964R06.
All 60 N1 and 35 N2 modes match complete frequency-specific singularity logs.
Those logs record internal m=-2..26 (29 harmonics) and -2..47 (50), whereas
the saved continua use the stored-mode ranges 0..24 (25) and 0..45 (46).
The N-directory executables have inferred capacities of 25 and 46; the
existing case-root `gocont_tr` has capacity 125. No recompilation was needed.

Flux job 2561429 completed six isolated calculations: native-executable
baselines, root-executable baselines, and root-executable internal-range
runs. All four baselines reproduce the original datcon files byte for byte.
Within the root-executable comparisons, only lmin/lmax change; all input
hashes, other settings and output q profiles agree. The archive preserves
127 source/input/mode files and verifies that live bytes remain unchanged.

For 0.03<=r<0.75, nearest logged-singularity discrepancies exceeding two
radial grid intervals (Delta r=0.005) decrease from 49/100 to 4/88 crossings
for N1, and from 6/52 to 3/50 for N2. Thus 131/138 new interior crossings
are within two intervals; seven comparisons remain unresolved. These are
nearest-log comparisons, not identified-branch matches; changing continua
can also remove crossings. Raw and viewer-processed interior results agree.
For r<=0.9, maximum boundary-frequency changes are 2.948% (N1) and 0.893%
(N2); much larger raw differences near the edge remain separately tabulated.

Artifacts: `research_archive/r06_internal_harmonics_20261119/`, including
`README.txt`, `r06_alignment_examples.png/.pdf`, `remaining_crossings.csv`,
`analysis_summary.json`, and `r06_matched_modes.csv` for the existing viewer.
The latter uses absolute project paths and requires no `--base_dir`.
New profiles are in `internal_basis/cases/nstxuG133964R06/N1/datcon1` and
`N2/datcon2`. This was a matching experiment, not a convergence campaign.
No DiTw publication, HDF5 update, mode reclassification or whole-shot
structure-hold release was performed during this experiment. The subsequent
user review and remaining higher-n work are recorded above.

### November 19 HDF5 v4: four reviewed N1 scopes released

After comparing convergence results, the user chose to retain the internally
matched continua for consistency with the legacy database and requested
adding the four cases to HDF5. The November 18 user review rejected all 331
raw N1 modes; the four new N1 records therefore use K=0 and presence=0, with
explicit `user_visual_review` zero provenance. This is not the NOVA-filter
empty-directory rule, and does not assert that no physical TAE could exist.

The new handoff is
`research_archive/nova_surrogate_export_20261119_v4/nova_surrogate_v4_complete_cases.h5`:
**197 complete cases, all n=1–10, 8,943 selected modes and 568 K=0 records**.
It adds 135388/B12/W29/Y93 to v3's 193 cases; their n=2–10 records contribute
155 previously selected modes (125/23/7/0 respectively). No new N1 modes
are selected. The full `nova_surrogate_v4.h5` retains 200 inventory IDs,
199 populated parameter sets, 1,978 available continua and 1,970 valid
K/mode records. Only R06 remains incomplete among the 198 processed cases;
the other two incomplete inventory entries are E205059A01t025 and nstxu_202806.

N1 continua use m=-2..22/-2..19/-2..15/-2..17, with hashes matching the
published DiTw profiles and archived producer receipts. Wider-range test
outputs are excluded. Live inputs match the preserved HDF5 equilibrium
hashes; all 331 reviewed mode bytes and membership were rechecked. The
shared exporter supplies the existing coordinates, normalization and radial
mask treatment. All four new continua have 197 valid native samples.

The updater preserves v3 files. **78,254 unaffected datasets** and their
attributes match v3 exactly, including every selected-mode array and all
equilibrium data. The subset copier verified **77,299 datasets** against
the new full file. Independent checks confirm membership is exactly v3 plus
the four approved cases, the 155 added subset modes, unchanged live
continua, and direct agreement with sqrt(native datcon values).

v4 is a data-release version; the structural schema stays `nova-surrogate-v3`
with `data_release=v4`. The four scopes use `status=available_reviewed_zero`,
`continuum_provenance=internal_harmonic_match`, and embedded review/release
evidence retaining their original hold reasons. The release is confined to
these hash-verified HDF5 records; the shared sorter registry still protects
older copies such as data_mixed. Other holds, catalog selections, source
modes and DiTw files remain unchanged. Reproduction uses the archived
`build_v4.py` and `reviewed_n1_release.json`; the archive also contains a
reader, input copies, coverage tables, source snapshots and validation reports.

### November 19 N1 harmonic convergence checks

The user authorized convergence checks for the four N1 cases after confirming
that the matched internal bases align the live continua with existing modes.
Flux jobs 2561408/2561409/2561411/2561413 completed 35 successful continuum
calculations in isolated project directories. The four starting baselines
reproduce the published files byte for byte. Only harmonic limits changed;
equilibrium/TRANSP/executable hashes, gamma, density settings and the radial
grid stayed fixed within each case, and exported q profiles are identical.

All cases tested m=-3..30, -3..40 and -6..40. Further tests used -6..50,
-12..50, -12..40 and -23..40 for 135388; the other three used -6..60,
-12..60, -12..90, -18..90 and -30..90. The separate 135388 40/50 upper-limit
check changes either boundary by at most 0.0149%. The final negative-cutoff
increment is -12..40 to -23..40 for 135388, and -18..90 to -30..90 otherwise.

| Case | Last-tested m range | Max upper-boundary change from matched baseline | Max change in either boundary in final increment | Final increment, r<=0.9 |
| --- | --- | --- | --- | --- |
| 135388 | -23..40 | 5.633% | 0.0105% | 0.00968% |
| B12 | -30..90 | 9.917% | 0.0179% | 0.00690% |
| W29 | -30..90 | 28.968% | 0.3485% | 0.0649% |
| Y93 | -30..90 | 34.653% | 0.8032% | 0.1763% |

These are frequency changes, using sqrt(native datcon columns), over 197
valid native pairs at r=0.01..0.99 (psi_N=0.0001..0.9801). No axis/LCFS
extrapolation or tail repair enters the comparison. The largest final changes
occur at r=0.99. Both cutoffs matter; W29/Y93 retain edge sensitivity and are
not uniformly converged to 0.1%. Successive differences are not error bounds,
and equilibrium/radial resolution was not varied. Upper-cutoff checks precede
the final lower-cutoff extensions; no joint infinite-basis limit is established.

One additional 135388 attempt at -6..60 produced no continuum and was rejected.
Its executable's symbol sizes imply capacity for 64 harmonics; that request
needs 67. The other three executables support 125. The failed receipt is
retained, and subsequent tests stay within those capacities. No executable
was rebuilt or substituted. File/settings preflight alone does not verify
compiled harmonic capacity.

Artifacts are in `research_archive/training_n1_convergence_20261119/`:
`README.txt`, `harmonic_convergence.png/.pdf`, `case_summary.csv`, full and
pointwise metrics, raw arrays in both coordinates, task settings, receipts,
frozen source evidence and `analyze.py`. Total reported batch wall time was
90.4 seconds, excluding queue/MPI startup. All 331 mode hashes and the four
live datcon1 hashes remain unchanged. No publication, hold release,
classification or HDF5 update occurred. These diagnostic profiles remain
separate from the continua matched to the old eigenmode calculations.

The user then requested comparison with the original backups. The reference
is the November 18 `training_n1_continuum_plan_20261118/before/` datcon1 set,
before the internal-range correction; all four backup hashes were verified.
For r<=0.9, latest wider-range frequencies are lower at every compared sample.
Maximum lower/upper changes from those originals are 2.97%/7.00% (135388),
7.50%/7.92% (B12), 4.59%/10.03% (W29), and 4.53%/10.17% (Y93). Median
absolute differences across that region are 1.88–3.20% for the lower boundary
and 2.50–5.81% for the upper. The internal-range correction accounts for a
substantial part of the shift; separate original-to-matched,
original-to-latest and matched-to-latest metrics are retained. Near r=0.99,
upper changes reach 40.03% for W29 and 44.36% for Y93. The original 135388
tail has abrupt jumps; its r=0.99 upper omega-squared=1121.32 is excluded by
the existing raw-value mask (it is not the literal 1000 sentinel). This
excluded native sample is recorded explicitly and does not affect r<=0.9
statistics. The convergence archive now includes
`original_backup_comparison.png/.pdf`, `original_backup_metrics.csv`,
pointwise differences, excluded samples, fingerprints and the reproducible
`compare_original_backups.py`. No additional NOVA runs or live changes.

### November 18 matched-harmonic continuum experiment

**Operational rule:** match the continuum to NOVA's internal harmonic
range, including two extra harmonics on each side for these verified legacy
runs: `lmin=m_min_saved-2`, `lmax=m_max_saved+2` (`nhar_saved+4` total).
This rule is now prominent in the main README and continuum runner guide,
with the four-case table and provenance in the
[input source map](surrogate_input_sources.md#saved-mode-harmonics-versus-internal-surface-field-harmonics-november-18).
Matching the stored `egn*` harmonic count alone is insufficient; verify the
internal limits for other builds. This is distinct from convergence testing.

The user requested matching the existing eigenmode harmonic sets before
convergence tests. Nine isolated continuum calculations completed in Flux
jobs 2561269/2561270, archived under
`research_archive/training_n1_matched_harmonics_20261118/`. The current
135388 continuum is reproduced byte for byte at m=0..12. Matching its
21 saved harmonics (0..20) barely changes the interior discrepancy: 38/41
historical training crossings remain beyond two grid intervals. W29/Y93
at 0..13/0..15 reproduce current continua byte for byte; B12 at 0..17 is
numerically equivalent in the interior (maximum relative difference 1.44e-9).

Source and matching logs expose the relevant distinction: NOVA's internal
surface-field matrices extend two harmonics below/above the saved mode
range. Matching these internal sets gives 135388=-2..22, B12=-2..19,
W29=-2..15 and Y93=-2..17. This is an existing-basis consistency check,
not the proposed -3..30/-3..40 convergence experiment. Input/executable
hashes and non-harmonic settings are identical within each comparison;
exported q profiles are byte-identical.

For the historical training subset and raw continua over 0.03<=r<0.75,
crossings beyond two grid intervals fall from 38/41 to 0/42 (135388),
4/4 to 2/6 (B12), 16/16 to 0/14 (W29), and 21/22 to 1/21 (Y93).
The new median absolute offsets are 0.411/0.825/0.475/0.368 grid intervals.
Across all frequency-matched modes, 105/108 interior crossings are within
two intervals. The three remaining flags lack nearby logged counterparts:
B12 4164/4181 near r=0.053/0.051, and Y93 1937 at r=0.165. Large nearest
distances there must not be interpreted as identified-branch displacements.
The dominant systematic shift is explained, with these residuals retained.

The archive contains four internal-basis datcon1 files, a hash-indexed CSV,
native receipts, frozen source evidence, before/after PNG/PDF plots and
reusable analysis. All 331 mode hashes and 62 original source-file hashes
remain unchanged. The single newer 23-harmonic 135388 mode has no saved
historical reference and is not individually validated by the 250-mode
cohort's range. No DiTw files, holds, labels, catalog or HDF5 were changed.
Convergence remains the next stage after this consistency experiment.

The user subsequently explicitly authorized publishing these four files,
provided the originals were backed up. Publication completed on November 18
(UTC stamp `20261119T000250Z`). Each `DiTw/<case>/N1/datcon1` now matches the
internal-basis result byte for byte, and the existing `datcon -> datcon1`
symlink is preserved. Originals were verified in the initial project backup,
an additional read-only `publication_20261118/before/` backup, and adjacent
`N1/continuum_backups/20261119T000250Z/datcon1` copies before replacement.
Atomic installs retain the original group/access permissions and the prior
owner's access through an explicit ACL. Independent verification confirms
all hashes, backup copies, viewer-loaded arrays and permission preservation.

The experiment archive's `publication_20261118/` contains the frozen plan,
events, completion receipt and independent checks; DiTw has
`.continuum_matching_20261119T000250Z.json`. Publication is explicitly an
approved matching-basis update with three remaining log-counterpart checks,
not an exact historical restoration or convergence certificate. The existing
N1/outcon still describes its previous calculation; the new native outcon,
commands and settings are in the referenced reconstruction receipt. Holds,
labels, catalog and HDF5 products remain unchanged.

After the user reported unchanged viewer offsets, a fresh check confirmed
all four live datcon1 hashes still match the publication. The separate
`data_mixed` training tree retains old continua, and Flux's default
`NOVA_DATA` points there. The user confirmed launching the original
relative-path n1_all_modes.csv without --base_dir, which takes this default.
Added
absolute-DiTw-path `n1_all_modes_ditw.csv` (331) and
`n1_historical_training_modes_ditw.csv` (63) in the prior recheck archive,
leaving portable lists and training inputs unchanged. Both lists resolve
correctly through the viewer even with a data_mixed base directory. A
noninteractive execution of the actual viewer confirms W29/2158's live
crossing at r=0.68493555 versus 0.71838195 in data_mixed; its plotted
arrays match the validated replacement exactly. `nhar=14` in both is
expected because nhar is read from the unchanged eigenmode. The viewer
image and path-verification receipt are in the matching-experiment archive.

The user then confirmed that their viewer command had omitted
`--base_dir $NOVA_DITW_ROOT` and reported that everything aligns well after
selecting DiTw. This confirms the data-root explanation and provides user
visual confirmation of the published profiles. The three missing nearby
log counterparts remain documented as diagnostic limitations; no additional
calculation, input-hold release, or HDF5 change follows from this confirmation.

### November 18 user review of the four held N1 cases

The user reviewed the 331 raw N1 modes in 135388/B12/W29/Y93 and reports all
are bad, many above the TAE gap, with visible spike/continuum-crossing
offsets. They suggested a backed-up m=-3..30 continuum experiment for the
surrogate. No input hold or production/HDF5 status was changed by this
discussion; accepting zero useful modes in this legacy set would not prove
that a wider-basis eigenmode calculation has no TAEs.

`research_archive/training_n1_continuum_plan_20261118/` preserves 62
hash-verified files: current N1 continua/settings/logs and the mapped inputs,
TRANSP files and case-root continuum executables. All 331 current mode
hashes and four datcon1 hashes match the November 17 recheck. Mapped edge q
values are 18.75/14.38/13.54/16.30. Saved N1/outcon records m=0..12 for
135388, 0..13 for W29 and 0..15 for Y93; B12's older format omits these
limits. The original 135388 mode cohort has 21 harmonics, plus one newer
23-harmonic file; B12/W29/Y93 have 18/14/16. Editable harmonic headers
alone cannot establish executable capacities or solver-only harmonics.

Two four-case task CSVs propose m=-3..30 and -3..40 using frozen input copies
and pass the shared runner's file/settings preflight. **No runs launched.**
Recommended work is a fixed-input/binary comparison with a reproduction
baseline, raw full-profile comparison and further range checks (including
the lower cutoff) before claiming convergence. Matching old spikes and
converging the continuum are separate checks; retain that distinction in
surrogate provenance. The archive README and backup_and_preflight.json
record settings, hashes, the user review and the proposed experiment.

### November 17 NOVA-filter zero clarification (v3)

The user confirmed that NOVA's internal filter removes singular/large-amplitude
grid-scale numerical modes before our classifier receives files. Zero `egn*`
files in the 23 identified case/n directories therefore means no modes
surviving that filter. This is sufficient evidence to change K/presence from
-1 to 0 and mark the 13 affected cases complete; the prior suggestion to
check logs for these records is superseded by that confirmation.

`configs/nova_confirmed_empty_scopes_20261117.json` freezes the user's statement,
23 scopes and their continuum hashes. `ConfirmedEmptyRegistry` applies the
same policy in the exporter and `scripts/confirm_nova_empty_scopes.py`, which
versions an existing HDF5 without recomputing scientific arrays. Named
directories must still exist with no mode entries, and continua must match
their hashes. Missing inputs and existing holds remain excluded. New fields
identify the K=0 source as `nova_internal_filter`, with the confirmation saved
per record and globally. `catalog_K=-1` retains the absence of a downstream
classifier summary; usable `K=0` and `mode_targets_valid=True` reflect the
confirmed upstream result.

The new full product is
`research_archive/nova_surrogate_export_20261117_v3/nova_surrogate_v3.h5`:
**193 complete cases**, all **8,943 existing selected modes**, 1,974 available
continua, 1,966 usable K/mode records and **564 K=0 records**. There remain
16 held n records, eight unclassified n records in E205059A01t025 and ten
records in the empty nstxu_202806 entry. Among the 198 processed cases only
five remain incomplete: nstx_135388, G121123B12, G142301W29, G142301Y93 (N1
holds), and G133964R06 (whole-case hold).

The companion `nova_surrogate_v3_complete_cases.h5` contains **193 cases,
1,930 complete n records, 8,788 modes and 545 K=0 records**. Its 75,703 copied
datasets match the new full file exactly. Independent validation confirms
that membership is precisely the prior 180 cases plus the 13 newly complete
cases, with all 23 promoted zeros present. Both v2 file hashes are unchanged.
The new archive includes `README.txt`, the working reader, manifests, code
snapshots, coverage tables and validation summaries. Full SHA-256:
`f46dc55b9ab66cc2c6d0e11f1e591d731c1f6882d3f02bbe3f465fe82fb9965c`.

The update changes six scalar/status fields in each of 23 records and adds
provenance. **78,108 other original datasets** and their attributes match v2
exactly, including all continua, coordinates, mode arrays, parameters and
profiles. Twenty-one focused coordinate/provenance tests pass, including
protection against new mode files, missing directories, changed/missing
continua and accidentally releasing holds. Earlier HDF5 files are preserved.

### November 17 recheck of the four training N1 holds

The user recalled recalculation and approval of the four training cases.
The correspondence does contain Nikolai's October 2 statement that all
identified shots had corrected continua. The explicit recorded acceptances
are C50/N1 on September 11, D46/N1/N2 on October 8, and R42/F62 on October 10
(the latter released because the remaining flagged modes were unsuitable).
No subsequent release of 135388/B12/W29/Y93 was found. The September 30
27-case audit explicitly excluded these four training cases; the later
post-training release lists should not be interpreted as clearing them.

Fresh read-only checks are in
`research_archive/training_n1_hold_recheck_20261117/`. Against the September 10
source hashes, B12/W29/Y93 retain identical N1 continua, all 8/34/38 mode
files respectively, and their available logs. The current crossing/log
diagnostic reproduces the old training-mode flags: 4/4, 16/16 and 21/22
interior comparisons exceed two grid intervals. Large nearest-log distances
can indicate missing correspondence rather than a displacement of an
identified branch.

135388/N1 has a changed continuum and one additional mode, but all 250
original modes are byte-identical. Current logs provide no matching frequency
records. The saved September singularity references remain usable for those
unchanged modes: comparing the current continuum to them still flags 38/41
interior crossings across 30 informative training modes, with median absolute
offset 4.90 grid intervals. Across all matched original modes, 47/52 crossings
are flagged. The current continuum differs from the preserved original by up
to 0.021% (lower) / 0.073% (upper) in frequency over 0.03<=r<0.75; much larger
outer differences do not resolve the original interior mismatch. No holds,
classifications, training labels or HDF5 products were changed by this recheck.

The same archive now includes viewer-ready `n1_all_modes.csv` with **331 raw
N1 files** (135388: 251; B12: 8; W29: 34; Y93: 38), sorted by case and
normalized frequency. `n1_historical_training_modes.csv` contains the **63
pre-hold training entries** (36/2/14/11). Both omit classification labels;
`in_pre_hold_training` records historical membership only. Current file
membership and every mode hash were checked against the recheck metadata,
and both CSVs were verified through the viewer reader. Counts, hashes and
source references are recorded in `n1_lists_summary.json`.

### November 17 HDF5 expansion: v2 verification

Completed products under `research_archive/nova_surrogate_export_20261117/`:

- `nova_surrogate_v2.h5`: **8,943 catalog modes**, 200 inventory IDs,
  199 populated 13-scalar sets, 1,974 available continua, 1,943 usable K/mode
  records including **541 verified K=0 records**; 64,323,636 bytes.
- `nova_surrogate_v2_complete_cases.h5`: **180 cases, all n=1–10**,
  **8,491 modes**, 1,800 paired continua and **470 K=0 records**;
  49,702,520 bytes. The original 94 complete cases are all retained.
- Remaining gaps in the full file: 31 missing selections, 16 existing holds,
  ten n records in one empty inventory case. These remain K/presence=-1.
  All original 8,943 selected modes are nevertheless included in usable n
  records. The 20 incomplete cases are listed in `handoff_validation.json`.
- Independent validation confirms exact catalog membership/frequencies,
  saved full-precision r moments and metadata, and unchanged v1 numerical
  values for all 4,517 old modes, 1,201 paired continua and equilibrium inputs.
  The complete-case copier checks all **70,540 copied datasets** against the
  full file. The example reader runs successfully. The old delivered v1 subset
  retains SHA-256 `72c45e665fd2027aa2ba41fecb96201d3104ed969e428da60fef965fa8e60c40`.
- New full SHA-256:
  `11d92818323cafe0dc85b3f7eca7b2345e6a51a391d186e62bf18572d5a77a44`.
  New subset SHA-256:
  `5773706c0e1de836d2dec24aefddf33edbda90bbb7324f9897d9b196c919236f`.

`README.txt`, `read_surrogate.py`, coverage/case-list CSVs and validation
summaries accompany the files. Native normalized frequencies, both r and
psi_N moments, density convention, radial masks, review flags and two mixed
catalog entries are retained. No 2D structures or lobe counts were added.
Next: share the new handoff; resolve existing missing selections/holds only
through separate work, then follow the superNOVA recalculation roadmap.

The export now accepts explicitly audited reconstructed continua in addition
to exact saved mode/datcon pairs. Reconstruction metadata preserves both the
old selection fingerprint and the fingerprint of the actual exported pair;
mode hashes, continuum hashes, and supporting evidence are checked. Existing
input holds, classifications and review states are unchanged. The new
`scripts/extract_complete_surrogate_cases.py` copies cases complete across
n=1–10 and checks every copied dataset against the source. Fifteen focused
coordinate/provenance tests pass.

The original 638 restoration tasks covered scopes containing selected modes.
The 104 zero-GOOD scopes also had continuum files, but their selection-era
versions had not been verified. A further 104 project-only tasks completed
under Flux job 2557625. Complete file membership and saved frequency, damping,
radial moments and grid size match for all 3,869 classified modes. Ninety-nine
scopes reproduce their historical crossings with case-root settings.

Five G-shot N1 scopes require historical ranges beginning at m=0:
G121123K34: 0–24; G121123K70: 0–17; G142301E72: 0–17;
G142301L94: 0–21; G142301U84: 0–23. A surviving K34 graphic explicitly
records 0–24; these range sizes also match their mode harmonic counts.
Job 2557629 reproduces all 119 saved crossings for those five scopes to
7.64e-15 maximum relative error. The attempted common -3–108 and 0–108
reconstructions did not match and remain archived as diagnostics, excluded
from the export. The current N1 files/logs date to October 2 and use other
ranges; the root outcon alone is insufficient for these five records.

Together, all 1,766 saved crossing witnesses in 93 of the 104 zero-GOOD
scopes reproduce within 3.46e-12 relative error. The other 11 scopes have
settings and mode-scalar evidence without saved crossing witnesses. The
registry covers 742 reconstructed scopes and distinguishes 459 centroid,
228 crossing and 55 settings-only validations. This restores legacy
selection compatibility, not a convergence certificate. Evidence and the
builder are in `research_archive/nova_surrogate_export_20261117/`.

### November 17 MPI reconstruction

`scripts/run_nova_continuum_mpi.py` reuses the scheduling pattern of
`FRC_AI/scripts/build_frcin_atlas_mpi.py`: one coordinator and independent
serial workers, checkpointed results, per-task errors, and verified resume.
The shared worker `src/nova_continuum_run.py` copies and hashes each case's
`mapdsk`, `mpout1`, `transp.dat`, and `gocont_tr`; retains every attempt; checks
runtime settings and structural outputs; and preserves native values.

The target manifest contains 638 affected case/n scopes across 80 cases,
covering the 4,426 selected modes whose continua were overwritten. Every
case-root `outcon` retains `lmin=-3`, `lmax=108`, `iden=4`, `ksound=1`,
`gamma=1.66666666`, `ns=3,201,1`. These settings and their source hashes are
recorded in `research_archive/continuum_reconstruction_20261117/`.

Flux pilot job 2557343 completed all ten n values for E204645A16t015 in
26.6 seconds with ten workers. All 84 saved October mode-centroid boundary
pairs agree within 4.784e-11 relative error; N1/N2/N5/N10 are byte-identical
to the earlier controlled reconstructions. A verified resume skipped all ten
without rerunning the producer. Twelve focused execution and continuum-loader
tests passed. Full Flux job 2557348 completed all 638 tasks with 32 workers,
zero failures, and exit code 0 in 557 scheduler seconds (555.7 seconds in
the driver), writing to
`research_archive/continuum_reconstruction_20261117/restored/cases/<shot>/N<n>/`.

The full historical comparison verified every published output hash. Both
boundaries match all 3,066 original October centroid measurements within
4.784e-11 relative error. For the 1,360 modes already mismatched in October,
all 1,372 original crossing witnesses (645 modes, 135 of 179 scopes) are
available again and agree within 5.851e-12 relative error. The other 44
scopes have no such crossing witnesses; matching recorded settings is not
a pointwise numerical check there. None of the 459 available original
full-file hashes matches the reconstructed bytes. Preserve the distinction
between numerical agreement at historical samples and byte-exact recovery;
do not reuse original input fingerprints for these files.

`restored/validation_summary.json` and the three `historical_*_comparison.csv`
files retain these results; batch/attempt receipts retain settings, copied
inputs, executable hashes, and outputs. The already delivered complete-case
HDF5 still has SHA-256
`72c45e665fd2027aa2ba41fecb96201d3104ed969e428da60fef965fa8e60c40`.
Next: use explicit reconstructed-continuum provenance in a new export version
when requested. Existing invalid-input holds and selection/review decisions
remain in force. The user has now authorized this batch's DiTw publication
after preserving the existing files; see the publication record below.

Following the owner-run ACL instructions, a November 17 host permission
recheck confirmed effective `ebelova` write access to DiTw, the sampled
E204645A16t015 case and N1 directory, and its existing `datcon1` file.
The inspected directories have both access and default `user:ebelova:rwx`
entries; the file has effective `user:ebelova:rw-`. All ancestors remain
traversable. This was an ACL/access check, not a write test or an exhaustive
audit of all descendants. That permission check made no DiTw changes.

### November 17 continuum publication

The user authorized two steps: back up the overwritten numbered continua,
then install the validated historical-setting reconstructions. The scope is
638 files across 80 cases, not all approximately 90 NOVA-C cases. A fresh
host preflight verified that every target still matches its November 16
audited hash, is a regular single-link file owned by ngorelen, and has the
expected unnumbered `datcon` symlink and a writable destination directory.

`scripts/publish_nova_continua.py` prepares independently copied project
backups and a frozen plan, then creates and verifies a second backup set
beside the live files before any replacement. Atomic replacements preserve
the unnumbered symlinks and existing ACL access, with an explicit ACL entry
retaining the previous owner's permissions. Six publication tests passed,
including changed-input/live-file rejection, backup verification, interrupted
replacement recovery, and previous-owner access. All 638 files were installed
and verified on November 17 at 18:11 EST (23:11 UTC), with no skips or errors.

The frozen plan, original metadata/ACLs, project backups, per-file event log,
and completed report are under
`research_archive/continuum_publication_20261117/`. The two independent backup
sets preserve the NOVA-C-era contents immediately before publication:

- Project: `before/<shot>/N<n>/datcon<n>` below that archive.
- DiTw: `<shot>/N<n>/continuum_backups/20261117T230824Z/datcon<n>`.

Every active numbered file now matches its validated reconstructed SHA-256;
both backup sets match the corresponding pre-publication SHA-256. A separate
host check verified all 638 ACLs, original groups/modes, and independent
backup inodes. Atomic replacement makes `ebelova` the file owner; explicit
`user:ngorelen:rw-` ACL entries preserve Nikolai's effective write access.
The existing unnumbered `datcon` symlinks remain intact. DiTw's
`.continuum_restoration_20261117T230824Z.json` records the complete plan and
publication result. Existing NOVA-C `outcon` logs remain historical records;
the restoration manifest records the settings of the active reconstructed
continua. Later NOVA-C runs can still overwrite those active filenames, so
the workflow still needs separate output names to prevent recurrence.

Mode files, selections, holds, and the delivered HDF5 were not changed.
The next expanded HDF5 must use the new hashes and explicit reconstructed
provenance; this publication does not make the historical combined mode/input
fingerprints match again.

### November 17 full-profile continuum comparison

Compared all 638 frozen pre-publication NOVA-C-workflow files with their
validated historical-setting reconstructions. Both input hashes are checked
against the publication plan. Frequency is the square root of native datcon
columns; no physical-unit conversion is applied. Raw comparisons use the
shared sentinel mask without tail repair; processed comparisons use the
unchanged shared loader. Metrics use common finite, nonnegative, ordered
boundary pairs on the native r grid; unavailable samples are not filled.

Across the full available radial domain, 620/638 profiles exceed 1% in at
least one boundary, and 554 raw / 552 processed exceed 10%. For r<=0.9
(psi_N<=0.81), both representations have 104 profiles above 1% and 33 above
10%; thus 516 profiles' >1% differences occur only outside r=0.9. This
radial restriction is a diagnostic, not a convergence criterion. 186
NOVA-C-workflow profiles lose usable samples relative to the reference
under the existing validity mask (606 raw / 609 processed grid points).

Representative maximum upper-frequency differences, relative to the restored
profile, are:

| Case/n | Raw maximum | r at raw maximum | Maximum after shared loader |
| --- | ---: | ---: | ---: |
| E204645A16t015/N10 | +559.755% | 0.865 | +47.842% at r=0.845 |
| E204645A16t015/N9 | +530.247% | 0.900 | +65.956% at r=0.885 |
| E204645A16t015/N2 | +657.324% | 0.985 | +657.324% at r=0.985 |
| E204669M03t030/N9 | +911.075% | 0.990 | +17.023% at r=0.990 |

The last row is the largest raw upper-boundary percentage change among all
638 pairs: Omega changes from 2.95698 to 29.89725. Lower-boundary percentage
maxima can be much larger near vanishing reference frequencies; the report
also measures absolute shifts relative to the reference gap midpoint and
radial RMS. These are native-grid maxima, not interpolated extrema.

The plotted rises track nq exceeding the NOVA-C launcher m_max: for example,
the original N10 case crosses 100 at r=0.845, while the largest raw upper
case has nq=98.63 at r=0.99 with m_max=90. This supports the established
harmonic-range explanation. Historical high-n curves can also rise near
their own m_max=108 limit; neither raw spikes nor the loader's tail clamps
are convergence evidence. The unchanged tail repair suppresses some large
differences and leaves others, including the N2 example, intact.

Reproducible scripts, per-profile metrics for both representations/domains,
compressed arrays with r and psi_N, input hashes, copied settings evidence,
seven diagnostic PDF pages, and overview/distribution PNG/PDF figures are in
`research_archive/continuum_full_profile_comparison_20261117/`. This comparison
does not change DiTw data, the loader, classifications, or the HDF5 export.

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

- **E205059A01t025:** N1/N2 continua were recalculated locally on November 19
  with m=-5..16/-5..27 and published after the user confirmed correspondence.
  Full-n production rules select 58 modes. All ten records are released in
  HDF5 v5; their per-mode visual review flags remain pending.
- **G133964R06:** released in HDF5 v5 with K=0 for all n, using the 95 reviewed
  N1/N2 rejections and the approved historical higher-n zero selection.
  Matched N1/N2 continua are published. The old shared input registry and
  historical INVALID outputs remain guards for uncorrected input copies;
  they are not the status of the v5 release. Higher-n basis differences alone
  do not require recalculation under the approved seed-database scope.
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

- **Continuum parallel-execution reference:** the user proposed reusing
  `FRC_AI/scripts/build_frcin_atlas_mpi.py` for embarrassingly parallel
  continuum recalculation. GitHub could not serve the supplied link through
  the available connector/web access; inspected the clean local copy at
  `/u/ebelova/FRC_AI/scripts/build_frcin_atlas_mpi.py` (checkout c89b2e9,
  file SHA-256 9be65b1847b17d3affc2a4ace441931f09ce75d3235f68d786fdd8a8208d088e).
  It uses mpi4py.futures.MPICommExecutor with rank 0 coordinating independent
  serial workers, root-only ordered CSV/checkpoint writes, resume validation
  and per-task failure reporting. The NOVA adaptation uses one (case, n_tor)
  per independent working directory; FRC-specific extraction/retry logic is
  not applicable. Eleven ranks provide ten workers for a full n=1..10 shot.
  The pilot measured 26.6 seconds for all ten, versus roughly 15 seconds per
  n in earlier serial diagnostics. A project-local environment at
  `research_archive/.nova_mpi_env` contains mpi4py 4.1.2 and uses Flux module
  `openmpi/4.1.6-gcc-13.2.0`. See the reconstruction section above for
  execution and validation status.
- **November 17 Tommaso workflow reference received and inspected:** the user
  supplied `/u/tbarberi/NOVA_runs/NOVA_workflow`. Read the current (non-OLD)
  `transp_dat/README`, `bash_scripts/READMEsh`, `python_scripts/READMEpy` and
  principal setup/run wrappers. They document TRANSP extraction -> q-solver
  -> mapping -> continuum -> NOVA eigenmodes -> NOVAK -> RBQ. `makeNOVA.sh`
  prepares/rebuilds the five stages using Nikolai's sources; `runst.scr`
  supports broad, targeted and detailed frequency scans with `goin`/`gost`.
  `run_gotae3.py` links an eigenfunction, opens `NOVAK_param` for editing,
  runs `gotae3_dev` and organizes output. `run_boost_pipeline.py` checks
  `pk=0.5` and prepares the RBQ inputs/database; the README notes that NOVAK
  and RBQ share object filenames and require separate/clean builds.
  `prepare_clichlmax.sh` keeps compiled minm_1=-3 and sets maxm_1 to rounded
  n_max*TRANSP-q_edge+5; this is an initial basis choice, not a convergence
  certificate or a check of the final mapped q. `prepare_gridparam.sh`
  takes ndat from TRANSP and fixes nmbrgrd=201, mnths0=256. Equilibrium beta
  matching still involves manual p_0 adjustment according to the README.
  These wrappers are interactive and site-specific; use as references for
  future isolated validation, not an already validated unattended pipeline.
  Current continuum wrapper leaves unnumbered `datcon` in the working
  directory; the NOVAK ntor mismatch warning occurs after execution, so
  future automation must verify inputs and preserve outputs explicitly.
  No external scripts were run or changed. This does not expand the current
  execution status beyond the validated continuum diagnostics.
- **November 17 continuum discrepancy traced and reproduced:** isolated
  continuum-only runs identify changed physical poloidal-harmonic limits as
  the cause of the reported >1% changes (32 modes / 18 case-n scopes). The
  older continuum used m=-3..108; the NOVA-C launcher regenerates `datcon`
  with m_min=0 and smaller per-n m_max, capped at 100. Using a common retained
  executable per case and copied equilibrium/TRANSP inputs, old settings
  recover October 29 boundaries for all 61 selected modes in these scopes
  within 4.79e-11 relative; new settings reproduce all 18 November 16 datcon
  backups byte-for-byte. Both use the NOVA thermal-NH continuum density;
  NOVA-C's later NMINI_H prescription is a separate discrepancy.
  In E204645A16t015/N10, changing only m_max=100 to 108 recovers the 46.98%
  upper-boundary shift at r=0.8417084, where nq=99.65. The 46.98% uses
  existing tail repair; the raw shift there is 67.23%. Low-n tests also show
  a significant effect from dropping negative m. Increasing m_max from 108
  to 166 changes this example's upper boundary by another 1.24%, so recovery
  of historical settings is not proof of convergence. Diagnostic scripts,
  independent input copies, source snapshots, receipts, comparisons and a
  plot are in `research_archive/continuum_cause_20261117/`; see the
  [source map](surrogate_input_sources.md). Production inputs, selections and
  delivered HDF5 files remain unchanged. Follow the legacy-seed restoration
  priority in the surrogate roadmap above; reserve comprehensive convergence
  work for the superNOVA recalculation. This audit does not validate all
  other scopes or recover historical hashes.
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
- **November 17 density provenance clarification:** rechecked the source
  mass sums and all 199 populated TRANSP inputs. Every case supplies constant
  `AIMPJ=12`, `XZIMPJ=6`; the reconstruction uses `AIMPJ*smooth(NIMP)` with
  proton-mass units, then extrapolates the complete mass sum to the axis.
  The old impurity-mass-13 comment is obsolete; that term is inactive in
  both sources. All TRANSP hashes and the NOVA source hashes are unchanged.
  NOVA-C's current source differs from the audited backup only by correcting
  the XZIMPJ comment from A to Z. The thermal `NH` versus `NMINI_H` evidence
  and prior 89-case diagnostic agreement remain applicable; historical
  executable identity is still not established. See the
  [source map](surrogate_input_sources.md) for source locations and details.
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
