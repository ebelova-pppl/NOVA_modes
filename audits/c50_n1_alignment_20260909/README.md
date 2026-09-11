# C50 N1 continuum / eigenmode-run inconsistency, 2026-09-09

**Historical status: user-approved INVALID for the original 73 C50 n=1 modes.**
On 2026-09-11 the user accepted the recalculated 18-mode N1 set and the
registry exclusion was removed. See the [correction review](../c50_n1_recalculated_20260911/README.md).
The measurements and invalidation receipts below describe the original inputs.

Terminology correction: NOVA calculates eigenfrequencies and eigenmode
structure. The earlier description of its run as a stability calculation
was incorrect.

The user subsequently instructed that the complete N1 set be invalidated
because eigenmode structure does not correspond to the continuum. This
covers 14 previously TAE-side modes and 59 previously EAE-routed modes.
The cause remains unresolved; no continuum shift or morphology-rule change
has been introduced. Historical measurements below document the diagnosis.
The user visually inspected every TAE-side mode in `nstxuG142301C50` and
reported displaced resonance structure for n=1, with other n appearing
consistent. This follow-up is explicitly non-blind.

All **14 N1 TAE-side modes** have an inner TAE-gap crossing at 0.03<=r<0.75
lying **2.42–9.73 native grid intervals farther out** than the nearest
singularity listed by the original NOVA eigenmode run at that exact mode
frequency. Median separation is 4.63 intervals; one interval is 0.005.
The first two modes use the lower boundary and the remaining twelve the
upper boundary. This comparison does not assign a continuum-branch identity
to every logged singularity or assess the crowded outer-edge spectrum.

| N1 mode | Eigenmode-log singularity r | Datcon upper crossing r | Separation in grid intervals |
| --- | ---: | ---: | ---: |
| 8889 | 0.470 | 0.517203 | 9.44 |
| 9040 | 0.465 | 0.511017 | 9.20 |
| 9225 | 0.455 | 0.503648 | 9.73 |

Raw signed-mode inspection independently shows sharp structure near the
logged locations. In these three examples, the strongest signed second
difference within 0.4<r<0.55 occurs at 0.475, 0.465, and 0.455, respectively.
The largest harmonic amplitudes there are approximately 0.499, 0.898, and
0.726. Thus the structure is appreciable and the displacement is large
compared with the +/-2-grid crossing window. Before input invalidation,
N1/8889 and N1/9040 survived the rules, while N1/9225 failed
BAD_CONT_CROSS_WINDOW. All three are now INVALID on input-validity grounds.

## Checks and limits

- Raw mode/datcon fingerprints match the current saved output for all 14
  modes. Recomputed crossing records match the saved records exactly.
- Every inspected mode has 22 stored harmonics, nr=201, and ntor=1.
  `datcon1` covers the expected 1-based radial indices 3 through 199.
  Its raw lower/upper arrays are exactly equal to the shared loader's output:
  **the recently adopted continuum cleanup changes zero points** here.
- The matching `out_go` frequency is omega squared, and agrees with each
  mode's squared binary-header frequency to floating-point precision.
  The logged singularity radius therefore supplies evidence independent of
  the Python viewer and sorter. This is not just a plotting-marker offset.
- Current qprofile, bprofile, rgrid, gridparam, mapdsk, mpout1, and transp.dat
  bytes are identical across N1 through N10. N1's saved continuum input and
  compiled harmonic-range configuration both specify n=1 and harmonics 0–21.
  No obvious current file-selection, point-count, or harmonic-count mismatch
  was found. Array plots retain stored harmonic indices.
- The N1 mode/log timestamps are March 25; the current continuum timestamp
  is June 20. Dates alone do not establish which inputs were actually used.
  The original `equout` and `equou1` targets under `/local/` are absent.
  Consequently this audit cannot prove the historical eigenmode and
  continuum calculations used the same equilibrium or identify which is
  wrong. The earlier wrong-q-profile incident in project history is a
  possible diagnostic lead, not an established cause here.
- The user reports N2–N10 look consistent. This audit does not independently
  certify all their modes or infer an automatic mismatch threshold from
  nearest-log-singularity distance; logged singularities span other branches
  and need not exhaust every boundary crossing.

The persistent exclusion is recorded in
`configs/known_invalid_inputs.csv`, with issue `CONTINUUM_MODE_MISMATCH`,
the user's decision, date, and this evidence. Both production sorting
methods and `make_tae_like_list.py` apply it before TAE/EAE routing. The
reason is `KNOWN_INVALID_INPUT`; its diagnostic includes the issue and
registry SHA-256. Rule outputs use `final_decision=INVALID`; RF-CNN outputs
use `status=rejected, final_label=invalid`. Excluded modes are retained in
`rejected_modes.csv` and the complete audit CSV, and removed from all usable
TAE/EAE lists. Summaries report `n_known_invalid_inputs=73`.

The registry matches the exact shot basename and toroidal mode number,
independently of root path or filename. It stays active for reruns and new
files in that scope until corrected inputs have been reviewed and the entry
is explicitly removed. GOOD/BAD/REVIEW morphology overrides cannot revive
INVALID data. Raw loaders and diagnostic viewers remain usable to inspect
or investigate the excluded files. Frozen rule presets and model weights
are unchanged; this is input validation, not a new morphology gate.

## Evidence and reproduction

`measurements.csv` records all 14 comparisons, exact input fingerprints,
frequencies, and pre-invalidation rule outcomes. `summary.json` records input hashes,
current input equality checks, and missing historical cache targets.
`check_alignment.py` recomputes these checks without modifying raw data;
use the pre-invalidation output backup as `--rules-dir`. The original
`summary.json` remains historical, including its original script hash.

```text
python audits/c50_n1_alignment_20260909/check_alignment.py --shot-dir /path/to/nstxuG142301C50 --rules-dir /path/to/sort_outputs/before_c50_n1_invalid_20260909/nstxuG142301C50 --out-dir outputs/review_c50_n1_alignment
```

Diagnostic figures remain local under
`outputs/review_c50_n1_alignment_20260909/`: `n1_all.png` shows all 14 signed
profiles with true crossing markers; `continuum_raw_loaded.png` compares
raw and loaded continua for n=1,2,3. Large plots are excluded from Git.

## Invalidation verification and saved outputs

All **163 tests pass**, including both production paths, exclusion before
TAE/EAE routing and inference, exact scope, malformed-registry errors, and
protection against a GOOD override reviving INVALID data.

Both C50 output sets were regenerated. All 73 N1 rows become INVALID;
**every field in all 538 N2–N10 rows is exactly unchanged** in each method,
including rule features and RF/CNN probabilities. C50 now has 194 TAE-side,
344 EAE-side, and 73 INVALID modes. Selected GOOD counts are zero for rules
(previously two) and one for RF-CNN (unchanged). No active training row
matches the excluded scope, so the training list is unchanged.

`invalidated_modes.csv` contains all 73 fingerprints and prior routing and
method labels. Run inputs, verification, and publication receipts are
`invalidation_run_inputs.json`, `invalidation_verification.json`, and
`invalidation_publication.json`. Both canonical output roots preserve the
previous C50 directory under `before_c50_n1_invalid_20260909/`.
`publish_invalid.py` verifies outputs and installs them with checked backups;
its docstring gives the reproduction commands. After publication, point
the verification phase at the pre-invalidation backup roots to reproduce
the old/new comparison. Do not publish into a backup root.

The current 27-shot comparison is `current_disagreements.csv`: **232 rows**,
from 234. `disagreements_removed.csv` lists N1/8889 and N1/9040, removed
because they are INVALID. No disagreement is added. The user's working
question list is unchanged by this update.
