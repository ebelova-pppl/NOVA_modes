# Interior harmonic-incoherence calibration audit

This compact audit records the non-blind, post-hoc evidence used to freeze
`BAD_INTERIOR_HARMONIC_INCOHERENCE` in production v3. It is not an independent
validation set.

The calibrated score is

```text
S_inc = f_core * J_core * N_eff_core * (1 - C_adj)
```

with inclusive `r <= 0.5`, base-2 Jensen--Shannon divergence, an integrated
core-energy active-harmonic cutoff of `>=0.005`, signed adjacent stored-row
coherence maximized over lags `-5..+5`, and a strict decision boundary
`S_inc > 0.10`. The threshold is eligible only for arrays with 201 radial
samples.

## Audit-only participation summaries

The grouped rule-feature schema also retains scalar harmonic-participation
summaries for future calibration. For samples with `W_i > 0`, define
`N_eff(i)=1/sum_h p_hi^2` and use the fixed strict reference `N_eff(i)>3`:

```text
N_eff_global = sum_i W_i N_eff(i) / sum_i W_i
G_3_global   = sum_i W_i I[N_eff(i) > 3] / sum_i W_i
G_3_core     = sum_{r_i<=0.5} W_i I[N_eff(i) > 3]
               / sum_{r_i<=0.5} W_i
B_3_core     = sum_{r_i<=0.5} W_i I[N_eff(i) > 3] / sum_i W_i
```

These are audit evidence, not rejection gates. In particular, a large
effective count alone does not establish incoherence: weak outer common-mode
tails can contain many simultaneous, strongly correlated stored harmonics.
The code therefore stores no unweighted pointwise maximum and introduces no
hard `W/W_max` cutoff or configurable decision threshold.

On the active labels, `B_3_core>0.10` is an illustrative audit slice that
contains 72 BAD and zero GOOD modes. It selects exactly the two H56 targets
among the 281 pilot-v2 rule survivors. Whole-radius `G_3` is not comparably
selective: `G_3_global>0.10` includes 12 labeled GOOD edge modes. The global
weighted effective count is promising but remains morphology-ambiguous; its
four pilot survivors include the two H56 targets plus E203609 N6/6239 and
N7/9477, whose large values are driven by outer common-mode tails.

`calibration_summary.csv` gives aggregate score-candidate counts.
`participation_summary.csv` records compact illustrative slices for the three
audit measures. `pilot_global_neff_candidates.csv` identifies the four
pilot-v2 survivors in the illustrative `N_eff_global>3` slice and records why
that slice is not a single morphology class. `h56_targets.csv` preserves the
exact score and participation components for the two visually identified H56
pilot modes that motivated the rule. Full exploratory per-mode tables remain
outside the version-controlled production contract.
