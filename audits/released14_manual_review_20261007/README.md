# Completed GOOD-list review of the released 14 shots, October 7, 2026

Elena reviewed the 76 selected GOOD modes from the October 5 batch. Her
corrections are installed in the B37 and E55 outputs; the other 12 shot
directories are unchanged. The current collection contains **70 accepted
TAE representatives**, one newly BAD mode and five separately classified BAEs.

- [Accepted TAE list](accepted_tae_modes.csv): 70 selected modes.
- [BAE list](bae_like.csv): the five manually assigned E55/N10 modes.
- [Six corrections](label_changes.csv) and [reusable overrides](manual_overrides.csv).
- [Disposition of the original 76 selections](review_dispositions.csv).
- [Per-shot counts](shot_summary.csv), [verification](verification.json),
  [installation receipt](publication.json) and [inventory update](inventory_update.json).

## Corrections

`nstxuG142301B37/N5/egn05w.9275E+02` is now final BAD, as requested by Elena.
The user did not specify a narrower morphology reason; the override records
her visual rejection without inventing one.

The five E55/N10 modes are `egn10w.1292E+02`, `egn10w.1375E+02`,
`egn10w.1478E+02`, `egn10w.1751E+02` and `egn10w.2149E+02`. Elena explicitly
confirmed this five-mode scope. Their mode-energy fractions below the lower
TAE boundary are 98.26%, 98.69%, 99.94%, 99.999% and 99.99995%, respectively.
The [measurements](bae_evidence.csv) use the shared loader/continuum repair
and the same radial sample weights W=sum_h |xi_h|^2 as the current gap split,
on the common finite, ordered, nonnegative boundary support.

The current automatic TAE/EAE split uses the upper boundary, so a mode below
the lower boundary can still be automatically routed TAE-like. BAE here is
the user's manual family assignment. No automatic BAE threshold or new
rejection gate was introduced.

## Reusable manual BAE decision

The existing override CSV schema now accepts `manual_decision=BAE` in
addition to GOOD/BAD/REVIEW. It applies only to eligible rule-evaluated modes
with a unique mode key and matching mode/continuum fingerprint. Invalid or
EAE-routed inputs remain ineligible. Stale/ambiguous overrides retain the
existing REVIEW behavior for otherwise accepted survivors.

For an applied BAE override, `final_decision=BAE`, `gap_region=bae_like`, and
`decision_source=manual_override`. The mode goes to `bae_like.csv` and stays
in the complete/final audit, but leaves all TAE, GOOD and BAD lists. Original
routing remains in `rule_results.csv`, with rule verdicts, feature values
and severities unchanged. Summaries include `n_bae_like`; the production
configuration and gate thresholds remain frozen v13. The interactive
labeler's g/b/r keys are unchanged; BAE is supplied explicitly in the CSV.

Both installed shots contain their reusable `manual_overrides.csv`. Future
reruns must supply that file, for example:

```tcsh
python scripts/sort_shot_mixed.py --method rules \
  --shot_dir "$NOVA_DITW_ROOT/nstxuG142301E55" \
  --out_dir /path/to/sort_outputs/nstxuG142301E55 \
  --manual_overrides /path/to/sort_outputs/nstxuG142301E55/manual_overrides.csv
```

## Counts and validation

Across these 14 shots: 12,250 inputs; 1,822 final TAE-like, 167 mixed,
10,256 EAE-like and five BAE-like. There are 1,918 final BAD modes and 71
GOOD before deduplication, yielding 70 selected GOOD; zero INVALID. B37 has
two selected GOOD (previously three); E55 has seven (previously twelve).

Verified all 1,272 input fingerprints in the two regenerated shots, exactly
six final-label/selection changes, byte-identical preliminary rule tables,
all other mode rows unchanged, consistent output lists, and zero stale,
ambiguous, unmatched or ineligible overrides. All 12 other shot trees match
their saved hashes. Both previous output directories are preserved under
`sort_outputs/before_released14_manual_review_20261007/`.

104 core rule-workflow tests and three rules-without-AI tests passed. The
full rule test suite's separate skill-validator check could not run because
the existing environment lacks PyYAML; no skill files were modified.

Both live inventories now mark the 14 shots `sorted_rules_good_reviewed`:
this records review of the selected GOOD list, not every rejected mode.
Membership remains 177 processed plus 14 training. Installed processed
outputs now contain 8,318 selected GOOD modes; 6,522 selections from the 123
remaining pending-review shots await GOOD-list review. The earlier reviewed
40-shot manifest, training labels, known-invalid scopes and the independent
F83/K79 release are preserved. Potential EAE/mixed flags remain recorded.

The [October 5 76-mode export](../released14_rules_20261005/good_tae_final_batch.csv)
and its receipts remain the historical automatic snapshot; use the 70-mode
accepted list above for this cohort's current selections.
