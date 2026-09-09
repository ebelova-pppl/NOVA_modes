# E204645 seven-mode grid-scale diagnostic, 2026-09-09

Non-blind explanation of the seven modes queried by the user, with current
rules labels already visible. This is diagnostic evidence, not adjudication
or a change to production rules, saved classifications, or training labels.

`measurements.csv` records the strongest width-qualified signed lobe in each
mode. All seven raw mode/datcon fingerprints and their complete recomputed
rule feature dictionaries match the current saved
`sort_outputs/nstxuE204645A16t015/all_modes_rules.csv`. The configuration is
`configs/rules/tae_rules_production_v6.yaml`; the shared continuum repair is
`datcon-monotonic-tail-v1`. All modes have nr=201, hence delta_r=0.005, and
remain TAE-like with fraction_below_upper2=1.

All seven first fail `BAD_GRID_SCALE_SPIKE`: a signed local extremum has
absolute normalized amplitude >=0.3 and interpolated signed-lobe FWHM <=1
grid interval. Their flagged radii are 0.495–0.635, below the r>0.7 region
where the width cutoff becomes 0.75. The stored harmonic index is zero-based
and is not a physical poloidal m.

Diagnostic re-evaluation disables only the grid-spike decision by setting
`GridScaleSpikeConfig(amplitude_min=None)`, with the exact-point continuum
gate disabled as in production. Six modes then return REVIEW/NO_GOOD_TEMPLATE
(all BAD gates passed; the production survivor policy would accept them as
GOOD before deduplication). N10/3812 instead returns BAD_GRID_SCALE_PACKET:
stored h=51 has samples -0.147575, +0.691975, -0.147724, +0.644460, +0.175074
at r=0.490–0.510, meeting the three-large-turn condition. Its selected peak
at r=0.495 is inside the packet gate's r<=0.5 limit.

The plots show narrow, radially shifted peaks in neighboring harmonics as
well as the flagged lobes. The gate does not distinguish such coupled edge
structure from an isolated numerical defect. Its rejection is reproducible;
whether these cases warrant a morphology-aware exemption remains open.
N10/5190 is particularly close to the width cutoff (0.989720 grid intervals),
whereas N10/3812 has both the narrow-lobe and repeated-turn signatures.

Local figures and the diagnostic generator are ignored under
`outputs/review_grid_scale_e204645_20260909/`: `new_list.png` contains the
four new-list modes and `previous_list.png` the three earlier modes. Each
page shows all signed harmonics, a grid-point zoom with the measured
half-height interval and neighboring harmonics, and the repaired absolute
continuum with W/max(W). Only this compact explanation and the seven-row
measurements are intended for version control.
