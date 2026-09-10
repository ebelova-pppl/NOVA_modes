# Remaining DiTw N1 continuum / eigenmode-log audit

2026-09-10. Further production sorting is paused at the user's request while
continuum/eigenmode input consistency is investigated and affected inputs are
corrected. This audit adds no rejection rule, validity exclusion, or training
label change. NOVA computes eigenfrequencies and eigenmode structures.

Scanned the **159 remaining shots**, beyond the 14 training shots and 27 pilots:
134 E shots, 24 G shots, and one legacy entry. All 318 requested shot/N groups
completed without a changing-input or processing error. There are **11,716 raw
mode files: 6,404 N1 and 5,312 N2, all nr=201**. Empty N directories remain
explicitly inconclusive.

## Review results

Start with [priority_review.csv](priority_review.csv): 21 shots, comprising
19 G shots and two E shots. These are **candidates for input review**, not
newly confirmed invalid shots. The counts below use TAE-like/mixed modes and
the interior window `0.03 <= r < 0.75`.

| N1 shot | Informative modes | Crossings beyond two intervals | Finding |
|---|---:|---:|---|
| E202806A02t045 | 2 | 2 / 2 | Large disagreement; N2 also lacks correspondence. |
| E205059A01t025 | 51 | 35 / 51 | Smaller repeated upper-branch offset. |
| G121123Q62 | 18 | 25 / 28 | Repeated; training was already suspended separately. |
| G121123R42 | 30 | 54 / 54 | Repeated; N2 also warrants review. |
| G142301B37 | 25 | 20 / 27 | Repeated. |
| G142301B85 | 10 | 21 / 21 | Repeated. |
| G142301D46 | 10 | 16 / 19 | Repeated. |
| G142301E55 | 5 | 4 / 6 | Repeated, with branch-dependent offsets. |
| G142301E77 | 8 | 14 / 22 | Repeated, with branch-dependent offsets. |
| G142301F62 | 10 | 10 / 10 | Repeated; clear displaced sharp structure in examples. |
| G142301F83 | 25 | 29 / 31 | Repeated. |
| G142301K79 | 9 | 12 / 12 | Repeated. |
| G142301L89 | 7 | 10 / 10 | Repeated; clear displaced sharp structure in examples. |
| G142301M32 | 6 | 11 / 12 | Repeated. |
| G142301U85 | 21 | 30 / 41 | Repeated, with branch-dependent offsets. |
| G121123N22 | 3 | 7 / 7 | Sparse; offsets and missing branch counterparts. |
| G121123N75 | 1 | 1 / 1 | Sparse N1 evidence; N2 has repeated disagreement. |
| G133964U37 | 4 | 6 / 10 | Sparse; lower-branch correspondence differs, upper can align. |
| G142301E34 | 2 | 4 / 4 | Inner lower crossings lack nearby logged counterparts. |
| G142301F66 | 3 | 6 / 6 | Upper offsets 5.81–6.72 intervals; lower counterparts missing. |
| G142301S94 | 1 | 1 / 1 | Only one raw N1 mode; 4.12-interval offset. |

The 13 G entries marked repeated each have at least five informative modes.
The six sparse G entries require particular care: a high disagreement fraction
from one or two modes cannot establish a whole-N1 conclusion. Their large
nearest distances can include missing counterparts rather than displaced
versions of the same resonance.

The two E cases deserve different interpretations:

- **E202806A02t045/N1/3772,4129:** datcon's upper crossing is at r=0.1815
  or 0.2195, whereas sharp mode structure follows logged radii around
  r=0.43–0.63. This is an evident input-consistency problem. Its cause, and
  whether continuum files, eigenmode files, or run provenance need correction,
  have not been established. N2 also has missing correspondence, so an N1-only
  fix should not be assumed sufficient.
- **E205059A01t025:** all 48 interior upper crossings lie outward of their
  nearest logged radii by 1.45–3.44 grid intervals; 35 exceed two intervals.
  Three lower crossings align. This is a smaller repeated pattern than the
  large G discrepancies and needs adjudication before declaring inputs invalid.

Two additional G shots remain in secondary review: **G133964R48/U27**.
Only 2/6 and 1/4 TAE-side comparisons exceed two intervals, respectively.
Including EAE-side raw modes gives 27/32 and 3/6. EAE-side plots are included;
the TAE/EAE routing depends on the continuum under investigation and cannot
be the only basis for clearing a shot.

Across the new E shots, 112 shots supply 1,103 informative TAE-side N1 modes
and 1,208 interior comparisons; 84 exceed two intervals. Many isolated large
distances come from an extra crossing pair around a shallow upper-continuum
dip, with no nearby logged counterpart. Sampled profiles often have structure
at that dip. These are local correspondence questions, not evidence that a
known resonance shifted by tens of intervals. A few other E shots have modest
offsets in a minority of modes; E204707X02t027/N1/3240 is an isolated larger
offset (6.21 intervals, with the other 22 comparisons within two).

The new 159-shot review priorities are:

- **1: 21 priority shots**, as listed above.
- **2: 24 secondary/coverage investigations**, including local extra crossings,
  sparse or minority offsets, missing logs, and R48/U27's EAE-side evidence.
- **3: 24 without a usable core conclusion**: 21 lack usable TAE-side interior
  comparisons; Q91, S39 and legacy `nstxu_202806` have no N1 mode files.
- **4: 90 within the review tolerance** for their available TAE-side interior
  comparisons. This is not a certification of eigenmode quality or other branches.

Two particularly limited log cases are E203655F01t017 (seven of eight
TAE-side N1 modes have incomplete blocks) and G142301M21 (97/99 raw N1 modes
lack exact-frequency records, and all 242 N2 matches have incomplete blocks).
M21 supplies no usable N1 comparison. Modes with empty matched singularity
lists and datcon crossings are recorded separately rather than silently passing.

N2 is a diagnostic control, not assumed valid. For example, N75 has 41/41,
R42 48/55, and F62 37/50 interior TAE-side N2 comparisons beyond two intervals.
The [N2 summary](database_n2_control_summary.csv) records all 159 shots, including
coverage failures. These observations justify checking affected N2 inputs too;
they do not constitute automatic N2 exclusions.

## Method, scope, and evidence

The driver imports the existing
[training measurement code](../n1_training_alignment_20260910/check_alignment.py).
It loads raw signed harmonics and continua through the shared repository
loaders; RF/CNN predictions and rule labels do not enter the audit. Every raw
N1/N2 file is measured, including EAE-side modes. Compact primary statistics
use geometric TAE-like/mixed routing, while `n_raw_*` and `raw_fraction_abs_gt_2`
preserve wider coverage. No new shots are marked `post_training_checked` in the
main sorting inventory: an input-consistency scan is not production sorting.

Binary-header omega squared must match a record in `out_go` or `out_go_prev`
within relative 1e-12. Incomplete or conflicting singularity blocks are excluded.
For each datcon crossing, the signed offset is `r_cross - nearest_logged_r`,
expressed in native radial intervals. **Two intervals (0.01 here) are a review
tolerance, not a calibrated validity gate.** Nearest matching does not identify
branches; extra crossings, absent counterparts and opposite inner/outer shifts
need explicit interpretation. Full-radius crossings remain available for review.

Raw versus shared-loader continuum checks found **zero changed samples in the
primary region across all 313 populated N1/N2 groups**. Thus the interior
discrepancies are not introduced by the edge-continuum repair. All 12,376 scan
source hashes were reverified after evidence generation. Twelve focused
log-pairing, input-validity and snapshot/cache tests pass. New cache tests cover
changed file bytes, added logs/modes, failed-group retries and files appearing
during a scan.

Versioned evidence:

- [All 200 N1 statuses](all_200_n1_status.csv): one row per inventory shot,
  current confirmed registry scope, audit cohort, input root and snapshot time.
- [New N1 summary](database_n1_summary.csv), [N2 controls](database_n2_control_summary.csv)
  and [priority shots](priority_review.csv).
- [Review modes](review_modes.csv): 87 selected examples across 45 shots,
  ordered by review priority, including N2 controls and EAE-side follow-ups.
  The first 50 rows cover priority-1 shots: 42 N1 examples and eight N2 controls.
- [Review crossing measurements](review_crossings.csv) and [receipt](receipt.json).

The all-200 table combines the **earlier 14 training and 27 pilot snapshots**
with this new 159-shot scan. It is not a fresh live remeasurement of all 200.
Training primary statistics refer to the training list before the 63 confirmed
N1 exclusions, using canonical training mode files; pilot/new scans use live
DiTw files. Five confirmed N1 exclusions (including C50) and the whole-shot
R06 exclusion remain unchanged. C50's earlier result remains inconclusive
during recalculation; recheck corrected inputs after that run is stable.

Full mode/crossing tables, hashes, resumable group JSON, 45 PNG comparison pages,
and the one-off table assembly script stay ignored under
`outputs/review_n1_database_alignment_20260910/`. The compact CSVs and scan
driver are the durable record. Rendering uses the existing shared audit helper.

```tcsh
# Run from the repository with the scientific Python environment active.
# NOVA_DITW_ROOT points to the live DiTw shot root.
python audits/n1_database_alignment_20260910/check_database.py \
  --data-root "$NOVA_DITW_ROOT" \
  --out-dir outputs/review_n1_database_alignment_20260910

python audits/n1_training_alignment_20260910/render_evidence.py \
  --audit-dir outputs/review_n1_database_alignment_20260910 \
  --mode-list audits/n1_database_alignment_20260910/review_modes.csv

python viz/view_modes_csv.py \
  audits/n1_database_alignment_20260910/review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

The PNG pages show signed harmonics, datcon crossings and matched logged radii
together. The interactive viewer uses its usual continuum markers; it does not
overlay NOVA log radii. Reruns reuse group snapshots only after checking code,
source hashes and file inventories. Changing main-inventory cohort membership
changes which shots are selected; the original selection is retained in the
runtime `selected_shots.csv` and fingerprinted main-inventory snapshot.

Next: adjudicate the candidates, investigate paired upstream inputs and complete
logs, then recheck corrected data before resuming production sorting. Do not
compensate for mismatched inputs by relaxing morphology gates or shifting
continuum curves to match selected mode features.
