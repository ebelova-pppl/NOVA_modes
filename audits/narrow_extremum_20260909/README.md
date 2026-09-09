# Five narrow rule survivors, 2026-09-09

All five modes questioned by the user are detected as narrow by the interior
total-energy gate, then exempted by its continuum-extremum exception. This is
a non-blind diagnostic; production decisions and thresholds are unchanged.
All have nr=201, so one radial interval is 0.005. Raw mode/datcon fingerprints,
every grouped v19 feature, and preliminary REVIEW decisions reproduce the
saved canonical rules outputs exactly. Production promotes survivors to GOOD.

| Shot / mode | W peak r | Connected W FWHM, intervals | Narrowest signed lobe with amplitude >=0.3, intervals |
| --- | ---: | ---: | ---: |
| E205040A01t016 N6/2836 | 0.265 | 1.474565 | 1.243353 |
| E204708F03t017 N8/6836 | 0.290 | 1.064122 | 1.013592 |
| E204708F03t017 N8/8950 | 0.120 | 1.111013 | 1.103147 |
| E204708F03t017 N10/7424 | 0.240 | 1.023774 | 1.107698 |
| E204645A16t015 N6/4765 | 0.135 | 1.129785 | 1.199712 |

The interior gate detects `r_peak<=0.5 AND W_FWHM<=2`, but exempts a nearby
upper minimum/lower maximum with `ext_dr<=0.02` and
`0<=ext_df_gap<=0.04`. All five satisfy both geometry cuts: radial mismatches
are zero or 0.005 and signed frequency clearances divided by mode frequency
range from 0.000609 to 0.005884. The exception has no additional minimum
width or smoothness requirement. It therefore protects even these visibly
sharp one/few-sample peaks. E205040 N6/2836 has a more extended body around
the sharp central signed reversal; its connected tallest W component is
still below the two-interval cutoff.

The separate signed-spike gate requires amplitude >=0.3 AND lobe FWHM <=1
at r<=0.7 (<=0.75 at larger r). All appreciable lobes are slightly wider
than their cutoffs. The strongest width-qualified amplitudes are respectively
0.22156, 0.26020, 0.04026, 0.13571, and 0.00671, all below 0.3. There are no
qualifying repeated-turn packets. Boundary, incoherence, and crossing-tail
gates also do not reject. No crossing-window violation requires the recently
adopted smooth-crossing exception: that exception is unused in all five.

`measurements.csv` retains full precision and fingerprints. `verification.json`
records the recomputation checks and relevant source/config hashes.
Native signed-profile/energy/continuum inspection plots and the local
diagnostic script are ignored under `outputs/review_narrow_extremum_20260909/`.
The diagnostics call the shared NOVA/continuum loaders, `evaluate_mode`, and
the engine's signed-extremum/half-maximum routines without smoothing profiles.

The calibration issue to revisit is minimum radial resolution within the
extremum exception. Any tightening needs a training/regression audit, since
the original exception was deliberately introduced to protect narrow labeled
GOOD extremum modes. No new cutoff has been adopted from these five cases.
