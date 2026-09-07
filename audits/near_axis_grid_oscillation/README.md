# Near-axis grid-oscillation calibration

This is a non-blind, post-hoc deterministic-rule calibration based on visual
review of grid-scale oscillations near `r=0`. It is not an independent
validation set.

The accepted production-v4 gate uses no missing-flip allowance. On each
stored harmonic, split the radial profile into maximal runs of strictly
consecutive nonzero sign changes. Keep runs with `N_s >= 4` only when the
run's largest absolute sample is at strict `r < 0.1`, and define

```text
Q_s = sqrt(sum_i (A[i+1] - A[i])^2).
```

Take the largest `Q_s` from one run on one harmonic; do not add across runs or
harmonics. Independently define the mode-level amplitude
`A_peak=max_{h,r<0.1}|A_h(r)|`. Reject as
`BAD_NEAR_AXIS_GRID_OSCILLATION` when both `A_peak >= 0.10` and
`Q_s >= 0.30`. The amplitude and `Q_s` winners may be different stored
harmonics and that relationship is retained in the audit features.

The earlier exploratory screen used `N_s >= 4`, run-associated amplitude at
least `0.08`, and `Q_s >= 0.25`. It selected 14 of the 281 production-v2
pilot rule survivors and six of 572 accessible active-training GOOD labels.
The accepted definition selects 10/281 pilot survivors and 5/572 accessible
training GOOD labels, with no additions outside the original 20 rows. Four
active-training GOOD inputs are unavailable in the live DiTw tree: three
zero-byte `nstx_135388/N8` files and one missing file.

The five exploratory rows released by the accepted thresholds are:

- pilot E203963 N6/4648: `A_peak=0.084451`, `Q_s=0.336443`;
- pilot E205035 N7/2822: `A_peak=0.093062`, `Q_s=0.360539`;
- pilot E205035 N7/2973: `A_peak=0.084768`, `Q_s=0.296677`;
- pilot E205057 N8/1767: `A_peak=0.134690`, `Q_s=0.272688`;
- training 204202 N7/8631: `A_peak=0.189177`, `Q_s=0.257899`.

The 20-row source table and accepted decision are recorded in
[`outputs/near_axis_sign_flip_pilot14_plus_training6.csv`](../../outputs/near_axis_sign_flip_pilot14_plus_training6.csv),
SHA-256
`d5432c9207d643a7f655294fba2fca0bab9c5cfe0d0f6b450245e0d15987f438`.
