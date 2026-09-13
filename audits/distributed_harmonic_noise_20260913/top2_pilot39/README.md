# Distributed harmonic noise: top-two reference and 39-shot pilot

2026-09-13. The user corrected the first audit's global denominator: use
the full-domain energy of the **two strongest individual harmonics** to limit
edge-mode dilution. This is the same reference as the existing continuum-tail
and extended-continuum-noise gates. Retain every harmonic in the HF numerator;
do not restrict it to the two reference harmonics.

This was the pre-adoption calibration of an additional OR rejection branch,
`BAD_DISTRIBUTED_HARMONIC_NOISE`. The user subsequently approved adoption;
see the [production-v12 adoption record](../adoption/README.md). The tables
below retain the v11 baseline and proposed changes from that calibration.

## Results

**No GOOD training label is flagged, and N10/3470 is the only newly flagged
pilot survivor.** The global-reference correction also catches one existing
training false acceptance near the edge.

| Population | Measured | Flagged by proposed branch | Newly rejected rule survivors |
| --- | ---: | ---: | ---: |
| GOOD training labels | 575 | 0 | 0 |
| BAD training labels | 1,751 | 35 | 1 |
| Current pilot GOOD modes | 1,647 | 1 | 1 |
| Current pilot BAD modes | 4,365 | 91 | 0 |

Of the 35 flagged BAD training labels, 33 are already rule-BAD, one is routed
EAE-like, and one is currently accepted: **G121123J38/N8/2222**. Relative to the
incorrect total-energy trial, the top-two reference flags 16 additional
BAD-labeled training modes, including that false acceptance. None of the 575
GOOD labels is lost.

| Newly rejected mode | Witness window | HF/top-two | HF/total (diagnostic) | HF/window | L_eff |
| --- | --- | ---: | ---: | ---: | ---: |
| Training: G121123J38 N8/2222, labeled BAD | 0.930–0.980 | 1.02795% | 0.14792% | 17.4445% | 0.033542 |
| Pilot: E205042A01t022 N10/3470 | 0.065–0.115 | 1.08988% | 0.94883% | 7.4911% | 0.038820 |

For J38, the top two harmonics contain only **14.39%** of total mode energy.
The reference correction therefore matters as intended: its edge window
passes the 0.5% top-two cut, while total-energy normalization misses it.
N10/3470 satisfies both denominator versions. **There are no further newly
rejected pilot modes** beyond the case already inspected by the user. Of
the 92 total pilot flags, 27 additional flags arise from changing the reference;
all 27 are already rule-BAD. The separate N5/2352 example does not trigger
this branch and keeps its existing BAD_CONT_CROSS_WINDOW rejection.

The closest GOOD training witness by the minimum of the three normalized
cut ratios is **NSTX 135388 N3/4934**. Its edge window r=0.95–1 has
HF/top-two=0.99547%, HF/window=16.414%, but L_eff=**0.02817**, below 0.03.
Its joint cut ratio is 0.93888, so the present length threshold is material;
zero training losses does not establish safety for looser cuts.

All **8,338 measured inputs have nr=201**. The remaining active training row
is pre-existing INVALID BAD K51 N4/8769 (nonfinite metadata). From the pilot
baseline's 25,967 rows, 19,248 EAE-routed and 707 INVALID inputs are explicitly
excluded: 610 registry-excluded R06 modes and 97 invalid-metadata modes.
There are no unexpected input errors or fingerprint mismatches.

Compact evidence:

- [New pilot rejections](newly_rejected_pilot.csv): one mode for the viewer.
- [New training rejection](newly_rejected_training.csv): J38 N8/2222.
- [All flagged training modes](flagged_training.csv): 35 BAD labels.
- [All flagged pilot modes](flagged_pilot.csv): 92, including 91 already BAD.
- [Per-shot coverage and counts](pilot_shot_summary.csv): all 39 shots.
- [Both discussed pilot examples](pilot_examples.csv).
- [Summary and verification receipt](summary.json).

## Exact candidate

| Parameter | Setting |
| --- | --- |
| Closed radial-window width | 0.05 |
| Simultaneous high-pass harmonic participation | N_hf >=4 |
| Qualifying HF / full-domain top-two-harmonic energy | >0.005 |
| Qualifying HF / raw energy in the whole window | >0.05 |
| Effective radial length of qualifying HF | >0.03 |

The top-two denominator integrates each individual harmonic over the full
native radial domain, then sums the two largest energies. There is no adjacency
requirement. Ties prefer the lower stored index; these are stored array indices,
not inferred physical m numbers. This ratio is a reference-energy ratio and
can exceed one. Total-energy fractions remain diagnostic columns only.

All other measurement conventions match the initial experiment. Form the
native signed high pass `h[m,i]=diff(xi[m,:],2)[i]/4` before any selection.
Pointwise `N_hf=(sum_m h_m^2)^2/sum_m h_m^4` selects simultaneous activity;
zero-HF points cannot qualify. The three stencil samples must lie inside the
same window. There is no continuum mask, smoothing or resampling.

Only centers with N_hf>=4 contribute to the HF numerator and its effective
length `L_eff=dr*(sum e_i)^2/sum(e_i^2)`. The local denominator includes raw
energy at **all** nodes of that window. Energies use native full-domain
trapezoidal node weights. All three rejection cuts must hold in the same
window and on the same qualifying HF population. Strict equality passes the
energy/length cuts; equality at N_hf=4 qualifies a center.

At nr=201, width 0.05 gives 11 nodes, nine eligible centers, and maximum
effective length 0.045. The witness maximizes the minimum energy/length cut
ratio, preferring a firing window and then the earliest exact tie.

## Scope and provenance

Training uses the active 2,327-row list and its fingerprinted v10 morphology
baseline. V11 changed ranking, preserving those rule decisions. Pilot
membership is exactly the 39 `post_training_checked=yes` shots in
`audits/main_dataset_shots/shot_status.csv`. Baselines come from the current
installed production-v11 `all_modes_rules.csv` tables, including accepted
recalculated C50/N1 inputs. Each measured mode-plus-continuum fingerprint must
match its baseline before and after measurement. Source, training, inventory
and baseline CSV hashes are checked again after the scan.

All 6,012 pilot top-two energy shares and stored-index pairs match the saved
shared production measurements. Recomputing the old total-reference condition
reproduces all 2,326 measured training flags from the first run exactly. Pair
witness reproduction, amplitude invariance and synthetic simultaneous versus
sequential packets also pass; details are recorded in the summary receipt.

The pilot comparison covers every TAE-side mode, including mixed routing,
before duplicate removal. EAE-routed and INVALID rows are retained in an
explicit excluded-input table; they cannot become new morphology rejections.
This audit does not sort unprocessed shots or inspect excluded raw inputs.
Training's existing EAE-routed labels are measured for continuity with the
first experiment but are never counted as new TAE-side rejections.

This is calibration against existing labels and previously sorted pilot
decisions, not an independent accuracy estimate. New pilot rejections need
visual review. Simultaneous HF participation measures multiplicity, not
statistical independence of harmonics.

## Reproduction and inspection

With the configured NOVA environment, from the repository root (Flux/tcsh):

```tcsh
python audits/distributed_harmonic_noise_20260913/scan_proposal.py \
  --training-root "$NOVA_DATA" --ditw-root "$NOVA_DITW_ROOT" \
  --global-reference top2 \
  --include-pilots --out-dir outputs/review_distributed_noise_top2_repeat
```

The five numerical defaults are the candidate above. A fresh output directory
is required. `--rules-root` defaults to the checkout's sibling `sort_outputs`
directory; pass an explicit path when relocating the saved baselines.
Full measurements and excluded-input rows remain ignored under
`outputs/review_distributed_harmonic_noise_top2_pilot39_20260913/`.

Inspect the compact proposed-rejection list with:

```tcsh
python viz/view_modes_csv.py \
  audits/distributed_harmonic_noise_20260913/top2_pilot39/newly_rejected_pilot.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

The viewer shows the saved baseline rule label; a proposed diagnostic flag is
not a relabeling. The complete proposed metrics are columns in the CSV.
