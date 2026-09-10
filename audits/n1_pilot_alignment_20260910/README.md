# N1 continuum / eigenmode-log screen of the 27 pilot shots

2026-09-10. Applied the same frequency-matched NOVA log comparison as the
[training audit](../n1_training_alignment_20260910/README.md), using the frozen
27-shot membership from `../continuum_noise_20260910/adopted_shot_summary.csv`.
No new pilot exclusions or sorting-label changes have been applied.

The scan covered 2,541 stable N1/N2 mode files: 1,451 N1 and 1,090 N2, all
nr=201. C50/N1 changed during both attempts and was explicitly excluded from
the numerical comparison; the other 53 shot/N groups completed. Successful
group results are saved individually so a live recalculation cannot erase
the evidence for other shots. C50's existing N1 exclusion and R06's whole-shot
exclusion remain in force.

## Findings

Primary comparisons use `0.03 <= r < 0.75`, the shared continuum loader,
binary-header omega squared matched to logs at relative tolerance 1e-12,
and distance to the nearest logged singularity. Two native grid intervals
mean 0.01 in normalized radius. Summaries below use TAE-like/mixed modes
selected by shared geometric routing, without consulting rule or AI labels.

| N1 shot | Informative modes | Crossings beyond two intervals | Interpretation |
|---|---:|---:|---|
| G121123K34 | 16 | 24 / 24 | Strong repeated mismatch; the 15 main upper crossings differ by 5.79–10.29 intervals. Extra lower crossings can lack a nearby log counterpart. |
| G142301U84 | 11 | 20 / 22 | Strong repeated mismatch. Inner crossings differ by 7.80–11.24 intervals; outer crossings by 1.54–4.90, in the opposite direction. |
| G142301E72 | 2 | 4 / 4 | Suspicious in both available modes, but sparse. The two outer offsets are 3.65 and 3.15 intervals. |
| G142301L94 | 2 | 2 / 2 | Suspicious in both available modes: 7.95 and 8.74 intervals. |
| G121123K70, outer region only | 11 | 8 / 11 | Additional edge review. All crossings lie at r=0.888–0.965, outside the primary window; offsets are 1.18–6.18 intervals. |

Signed-profile spot checks support K34/U84: `K34/N1/6996,7510` and
`U84/N1/2769,3274` have structure near logged radii displaced from datcon
crossings. The E72/L94 examples also show structure near log radii; their
small mode counts limit shot-wide conclusions. K70's `2987,3507` examples
have weak outer structure near the log radii, offset from the datcon markers.
For the four interior candidates the continuum repair changes zero samples
in the primary region. For K70 it changes zero samples over the entire datcon
range, so the extra outer-offset finding is not created by the repair.

The E shots have **167/169** usable interior N1 crossings within two intervals
(154 modes across 12 E shots). The exceptions are E204707A04t030 N1/4123,
at 2.045 intervals, and E204944A01t017 N1/1659, with one nearest-log distance
of 12.22 intervals near r=0.746. Neither establishes a repeated shot-wide
pattern. E shots without usable interior comparisons are not counted as passes.

Follow-up inspection of the two E exceptions:

- E204707A04t030 N1/4123: datcon crossing r=0.575223 versus logged r=0.565,
  or 2.045 grid intervals, just beyond the two-interval review tolerance.
  The strongest signed second difference in 0.54<=r<=0.59 is at r=0.570,
  about one interval from the datcon crossing. Neighbors 4112/4133 have
  log offsets of 1.10/1.95 intervals. This is a marginal flag rather than
  the repeated large displacement seen in the confirmed G cases.
- E204944A01t017 N1/1659: the main upper crossing at r=0.690451 aligns
  with logged r=0.685 to 1.09 intervals. A shallow upper-continuum dip
  around r=0.750 creates an extra pair at r=0.746120 and 0.755657, absent
  from the log. The signed mode has a sharp peak/second difference near
  r=0.755, consistent with structure in that region. The reported 12.22
  intervals compare the extra inner crossing to the main logged singularity;
  they do not establish a displacement of the same resonance. The second
  extra crossing lies outside the primary r<0.75 window. Raw and loaded
  continuum samples agree around the dip. This is a local log/continuum
  correspondence issue; its cause is unresolved, and it does not establish
  a systematic N1 shift. The neighboring 1644 mode's main crossing also aligns.

The numerical table retains the original nearest-distance measurements.
These follow-up plots are under the runtime `figures/` directory with the
two E-shot names; their five-mode manifest is `e_exception_modes.csv` there.

Additional limitations:

- **H56/N1:** all 384 raw mode records, including 78 TAE-side modes, have
  incompatible declared and printed singularity counts. For example, one
  block declares 57 points but supplies 24 radii. The partial lists are
  excluded. Three plotted signed-profile examples are noisy; this does not
  establish the specific systematic shift being tested. The original run/log
  needs investigation before a reliable log-based alignment result is possible.
- **N80/N1:** none of its eight TAE-side modes has an exact-frequency log
  match. Two plots are supplied for inspection, with no inferred log radii.
- **C50/N1:** active recalculation prevents a stable comparison. The failing
  group is explicitly recorded in the metadata and summary, not represented
  as a successful zero-crossing scan. Recheck after the upstream run finishes.
- **R06:** already INVALID as a whole shot. Its raw diagnostic comparisons
  do not reopen that decision.
- N2 is a control, not assumed valid. K34 has 28/40 interior TAE-side N2
  comparisons beyond two intervals; K70 8/8 and V21 5/5. Other N2 results
  are in the control summary. Thus some issues may extend beyond N1.

Nearest-log matching is not a branch identification. A very distant nearest
radius indicates absent correspondence; it is not a measured translation of
a known resonance. Opposite shifts on different branches can cancel in a
signed mean, so inspect the per-crossing records and absolute distances.

## Evidence and reproduction

- [N1 summary](pilot_n1_summary.csv): all 27 shots, coverage and review notes.
- [N2 control summary](pilot_n2_control_summary.csv).
- [Review modes](review_modes.csv): 44 N1 modes, grouped as repeated, sparse,
  outer-only or isolated findings; no GOOD/BAD decisions.
- [Review crossings](review_crossings.csv): 66 relevant comparisons.
- [Figure selections](figure_modes.csv) and [receipt](receipt.json).

Full coverage, all crossings, source hashes, per-group snapshots and seven
PNG pages are ignored under `outputs/review_n1_pilot_alignment_20260910/`.
The shared renderer now shows the full radius and labels missing/incomplete
log evidence, omitting unverified partial radii. The earlier training plots
and their original renderer hash remain historical; its renderer source is
preserved in that runtime directory as `render_evidence_before_pilot.py`.

```tcsh
python audits/n1_pilot_alignment_20260910/check_pilots.py \
  --data-root "$NOVA_DITW_ROOT" \
  --out-dir outputs/review_n1_pilot_alignment_20260910

python audits/n1_training_alignment_20260910/render_evidence.py \
  --audit-dir outputs/review_n1_pilot_alignment_20260910 \
  --mode-list audits/n1_pilot_alignment_20260910/figure_modes.csv

python viz/view_modes_csv.py audits/n1_pilot_alignment_20260910/review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

Eight focused registry, preprocessing and log-comparison tests pass. The
user-confirmed training N1 exclusions were applied separately: 135388, W29,
Y93 and B12, with 63 BAD-labeled rows archived out of active training. No
models or external sorted outputs were regenerated in this diagnostic task.
