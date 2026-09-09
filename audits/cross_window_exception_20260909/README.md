# Crossing-window exception audit and adoption, 2026-09-09

The user inspected and accepted all 13 additional candidates, then requested
adoption. Their fingerprinted review is recorded in `user_review.csv`.
The shared engine now implements the audited exception in production v7
(ruleset/features v19). V5/v6 presets remain frozen and disable the exception.
The results below describe the pre-adoption impact audit; the final section
records completed integration, regeneration, and publication verification.

The user selected **A_cross < 0.2 AND K_cross < 0.1** for an impact audit.
During that initial audit, no production gate, classification, training label,
or canonical output was changed. The existing grid-scale width thresholds
remain intact.

Across the 27 processed shots, the exception recovers **19 of the 1,247
BAD_CONT_CROSS_WINDOW modes** (1.52%). All 19 pass the remaining BAD gates;
production GOOD would increase from 940 to 959 before deduplication (2.02%).

| Shot | Recovered |
| --- | ---: |
| nstxuE204186A01t020 | 3 |
| nstxuE204621A03t030 | 3 |
| nstxuE205045A01t022 | 8 |
| nstxuE205057A01t020 | 5 |

The first six are the modes previously discussed with the user. The other **13**
are in `additional_to_review.csv`, with their subsequent approval in
`user_review.csv`. `recovered.csv` includes all 19 shot modes plus the two training
recoveries, identified by `cohort`. The blank `label` column keeps these
files suitable for diagnostic viewing without pre-filling an adjudication.

The complete current 2,390-row training list was reevaluated with current
shared preprocessing and routing. Two labeled GOOD modes are recovered:

- nstxuE204669M03t025 N8/9673: A_cross=0.18996, K_cross=0.03483.
- nstxuE205052A01t022 N9/1061: A_cross=0.19578, K_cross=0.05860.

**No labeled BAD training mode is newly accepted.** The baseline training
survivors comprise 541 labeled GOOD and 25 labeled BAD; the projection is
543 GOOD and the same 25 BAD. The 26 EAE-routed BAD training rows and one
documented invalid input are preserved. These are calibration results using
existing labels, not an independent validation of physical correctness.

## Exact exception definition

1. At every true lower/upper continuum crossing, measure the existing
   +/-2-grid-interval window using the shared extractor. A crossing violates
   the current gate when window A>=0.25 OR window W/max(W)>=0.05.
2. For each violating crossing, linearly interpolate every **signed**
   harmonic to r_cross, then set A_cross=max_h(abs(interpolated xi_h)).
   This is a point amplitude, not the window maximum or sqrt(W/max(W)).
3. Use the existing K_cross from the shared tail-feature extractor: the
   norm of unscaled signed second differences divided by the local amplitude
   norm, using all harmonics and complete centers within +/-4 grid intervals.
4. Excuse a violating crossing only when **both strict cuts hold at that
   same crossing**, K is defined, and nr=201. Equality does not qualify.
   Every violating crossing must qualify; any remaining offending crossing
   retains BAD_CONT_CROSS_WINDOW. Non-violating crossings need no exception.
5. Only after every offending window is excused, evaluate all subsequent
   BAD gates unchanged. Earlier rejection reasons cannot be rescued.

All inspected valid inputs have nr=201. Recovered-mode CSVs retain every
violating crossing's amplitude, K, pointwise W, window maxima, and inner
integrated-energy fraction, plus the raw mode/datcon fingerprint. The two
`required_*_max` columns are maxima across all violating crossings and are
equivalent to requiring both cuts separately at every such crossing.

## Threshold sensitivity

The complete 4x4 sweeps for both cohorts are in `threshold_sweep.csv`.
Illustrative results after applying the later gates:

| A_cross limit | K_cross limit | Shot recoveries | Training GOOD recovered | Training BAD recovered |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | 0.1 | 11 | 0 | 0 |
| **0.2** | **0.1** | **19** | **2** | **0** |
| 0.25 | 0.1 | 31 | 3 | 0 |
| 0.2 | 0.2 | 32 | 2 | 0 |
| 0.2 | 0.4 | 37 | 2 | 2 |

At looser settings, some modes clear the window gate but fail a later BAD
gate; the sweep distinguishes these from actual recoveries. The selected
thresholds have no such cases in either cohort.

## Reproduction and verification

Run with the repository's scientific Python environment, from the repo root:

```text
python audits/cross_window_exception_20260909/audit.py --kind shots --input-root /path/to/sort_outputs --out-dir outputs/review_cross_window_exception_20260909/shots
python audits/cross_window_exception_20260909/audit.py --kind training --input-root /path/to/training/data --out-dir outputs/review_cross_window_exception_20260909/training
```

The historical audit script uses the frozen 27-shot membership, production-v6
configuration, and datcon-monotonic-tail-v1. Its recorded calculation uses
the pre-v19 engine (checkout `e3d2e4e`); keep that source when reproducing the
original impact report. The historical script verifies all 1,247 raw
mode/datcon fingerprints against the saved shot exports, recomputes and
matches their crossing records and K features, and matches the complete
baseline feature dictionaries for all possible recoveries in the sweep.
Training modes use fresh shared loading, routing, and rule evaluation.
All source files are hashed before and after the audit to detect changes.
`summary.json` retains input/code hashes, baseline counts, and sweep results.
Strict-cut equality, undefined K, the known invalid training input, and the
previously discussed N2/8520 were checked directly before the full audit.

Full per-rejection measurements remain local and ignored under
`outputs/review_cross_window_exception_20260909/`. The audit performs no RF
ranking, CNN inference, manual overrides, or frequency/structure
deduplication. The integrated adoption uses the canonical sorter and its
active RF ranker as described below. The 13 additional shot candidates have
now been visually accepted by the user.

## Integrated v7 verification

`adopt.py stage` runs all 27 canonical rules workflows into a local output
root. Both `stage` and `verify` then compare every saved decision and prior
feature with the v6 baseline, freshly recompute complete features from raw
inputs for all 4,267 TAE-side modes, and evaluate all 2,390 training rows
with and without the exception. `verify` reuses completed staged exports;
it still performs every raw-data comparison before recording source hashes.
RF duplicate selection must complete without fallback. `publish` requires
these verified hashes, copies and checks all new exports before replacing
any output, and preserves and verifies every old shot directory as a backup.

```text
python audits/cross_window_exception_20260909/adopt.py stage --data-root /path/to/DiTw --training-root /path/to/data --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai --rf-model models/nova_mode_classifier.joblib --out-root outputs/review_cross_window_v7
python audits/cross_window_exception_20260909/adopt.py publish --rules-root /path/to/sort_outputs --out-root outputs/review_cross_window_v7
```

Reproduction requires the pre-exception v6 exports as `--rules-root`. After
publication, these are under `before_cross_window_exception_20260909/`.
RF-CNN inference is unchanged; disagreement comparisons use the existing
fingerprinted RF-CNN outputs. The original `summary.json` remains the
historical impact audit, separate from integrated verification receipts.

All **160 tests pass**. Full raw-data verification matches every current v19
feature record for all 4,267 TAE-side modes, and all pre-existing feature values
match the v6 baseline. The 27 regenerated runs recover exactly 19 modes,
producing **959 GOOD before clustering and 953 selected**, from 940 and 934.
The remaining labels and routing match across all 19,325 inputs. No duplicate
ranking fallback or enabled rejection-gate resolution exclusion occurred.
Training verification recovers exactly two labeled GOOD and zero labeled BAD.

All 27 outputs are published in the established rules root. Each previous
shot export is retained in `before_cross_window_exception_20260909/` there.
`adoption_verification.json` records configuration/source/input comparisons;
`publication.json` verifies every published and backup tree by content hash.
`adopted_changes.csv` records the 19 production label changes and
`regenerated_shot_summary.csv` records final representative counts.

`regenerated_disagreements.csv` is the current **234-row** comparison with
unchanged RF-CNN results. Against the preceding 229-row list, **222 entries
are identical, seven disappear, and twelve appear**. All twelve additions
have fingerprint-matched user GOOD approval in `user_review.csv`, so there
are **zero new cases needing review**. `disagreement_changes.csv` lists all
19 additions/removals; `disagreement_delta.json` records counts and hashes.
Recreate that comparison with `compare_disagreements.py`. Earlier lists and
the user's edited question list are preserved.
