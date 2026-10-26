# nstxu_204202: five-mode diagnostic review, October 25, 2026

Four of the five queried modes are already included in
`$NOVA_REPO/../sort_outputs/nstxu_204202/good_tae_final.csv` and in the live
shot's GOOD export. Only `N10/egn10w.1026E+02` is rejected there, with
`BAD_AXIS_SPIKE`. A fresh five-mode check with frozen production v13 agrees
with all five installed decisions. The canonical training CSV labels all five
GOOD; the N10 training/rules disagreement remains unresolved.

This is a **non-blind diagnostic review**: the user's reported decisions,
training labels, saved rules, legacy RF outputs and earlier review records
were inspected. No labels, overrides, rules, models or installed shot outputs
were changed. Classifier scores were not used to adjudicate morphology.

| Mode | Installed selection / current v13 | Raw-data assessment |
| --- | --- | --- |
| N4/egn04w.8263E+01 | GOOD / GOOD | Compact, coherent dominant pair near a lower-continuum maximum. Energy FWHM is 5.21 native grid intervals. Supports retaining the mode. |
| N6/egn06w.2987E+02 | GOOD / GOOD | Broad edge-localized envelope, energy FWHM 37.00 intervals. Only weak remote crossings. |
| N6/egn06w.1273E+02 | GOOD / GOOD | Resolved main envelope near r=0.655, energy FWHM 34.15 intervals, with coupled outer structure. No true TAE-boundary crossings. |
| N7/egn07w.9477E+01 | GOOD / GOOD | Resolved multi-lobed structure, main energy FWHM 26.81 intervals. Its single remote crossing has W/Wmax=0.00382. |
| N10/egn10w.1026E+02 | BAD / BAD | Otherwise plausible main structure, but an isolated near-axis spike supports rejection under the current policy. |

N4's global energy peak is at r=0.505. The nearby sampled lower-continuum
maximum is at r=0.510, with frequency 2.86659724 versus mode frequency
2.87445781 in the shared loader's NOVA units. The mode lies on the gap side
of this maximum; there is no crossing through the compact peak. Its only
true crossing is at r=0.275758, with W/Wmax=0.002322. The dominant harmonics
have a resolved rise and fall on the native grid. An older sealed review
called this mode BAD for a thin envelope, but that was not the adopted
training label. The compact peak alone does not justify rejecting it under
the current morphology policy. N6/2987 has a different, broad edge morphology;
both are retained.

N10's decisive feature is in **zero-based stored harmonic 15**, not an inferred
physical poloidal m. At r=0.015 its signed amplitude is -0.284064923; its
neighboring values at r=0.010 and 0.020 are -0.020092449 and +0.016409511.
The absolute-profile FWHM is 1.068712 native grid intervals. This meets the
axis gate (r<=0.03, amplitude>=0.2, width<=10). The small integrated energy
of this isolated feature does not exempt an individual-harmonic axis spike.
The main energy peak near r=0.575 and negligible continuum-crossing energies
do not explain its rejection; the spike does. The same training/rules
conflict was documented on August 16 and is reproduced here. A label change
would require a separate explicit adjudication.

The suspected installed directory was checked after Elena identified it as
the probable source. Its files use the older internal v14 ruleset string,
whereas the current production-v13 preset pins internal v25. Nevertheless,
all five input fingerprints and decisions match the fresh check. In the
installed `rule_results.csv`, the four survivors have preliminary REVIEW
decisions; in `final_classifications.csv` and `good_tae_final.csv`, they have
`final_decision=GOOD` and `selected_final=True`. REVIEW means no rejection
gate fired before the production survivor policy. It does not mean BAD.
This distinction does not establish which file the colleague actually used.

## Evidence and validation

- [Mode summary](mode_summary.csv): current training label, selection membership,
  rules verdict and measured geometry for each mode.
- [Rule evidence](rule_evidence.json): complete grouped feature payloads from
  the fresh v13 check.
- [Provenance](provenance.json): hashes of inputs, configuration, source files,
  training list, exports and installed output evidence.
- [N4 peak and N10 axis detail](../../outputs/nstxu_204202_five_mode_review_20261025/figures/n4_peak_n10_axis_detail.png)
  ([PDF](../../outputs/nstxu_204202_five_mode_review_20261025/figures/n4_peak_n10_axis_detail.pdf)).
- [Full diagnostic figures](../../outputs/nstxu_204202_five_mode_review_20261025/figures):
  all 54 signed harmonics, radial energy, absolute-frequency continuum and
  stored-harmonic spectra for all five modes. The renderer's inner-extremum
  search is restricted; use the explicit N4 local-maximum check above for
  its r=0.51 feature, rather than the generic inner-extremum annotation.

All five mode files and the four associated continua are byte-identical
between the live DiTw root and the training root. All five mode-plus-continuum
fingerprints match the September 13 v13 training audit and the installed
sort output. Every mode has 54 stored harmonics and 201 radial samples. The
fresh check reports five TAE-like modes, four GOOD, one BAD, zero INVALID,
and no radial-resolution warnings. It operates on an isolated five-mode
symlink subset, not a full-shot reprocessing or replacement of installed
selections. Existing full-shot lists independently confirm membership.

The run and manifest are under
`outputs/nstxu_204202_five_mode_review_20261025/`. To reproduce the detailed
plot with the project's NumPy 2.x environment, from the repository root:

```tcsh
python audits/nstxu_204202_five_mode_review_20261025/plot_details.py \
  --shot-dir "$NOVA_DITW_ROOT/nstxu_204202" \
  --out-dir outputs/nstxu_204202_five_mode_review_20261025/figures
```

Next steps: identify the colleague's exact rejection file if a discrepancy
persists, and explicitly adjudicate the N10 training label if requested.
