# Completed GOOD-list review of N75/B85/F83/K79, October 7, 2026

Elena reviewed the 20 selected GOOD modes and rejected
`nstxuG121123N75/N4/egn04w.1299E+02` for **many sharp, large-amplitude spikes
near the edge**. The remaining 19 selections are approved.

- [Current accepted TAE list](accepted_tae_modes.csv): nine N75 and ten F83
  modes; B85 and K79 have no selected GOOD modes.
- [Manual BAD override](manual_overrides.csv) and [label change](label_changes.csv).
- [Disposition of all 20 reviewed selections](review_dispositions.csv).
- [Current per-shot counts](shot_summary.csv), [verification](verification.json),
  [installation receipt](publication.json) and [inventory update](inventory_update.json).

N75 was regenerated with the existing production rules and fingerprinted
manual override. Its installed `good_tae_final.csv` now has nine modes;
`bad_tae_like.csv` includes N4/1299. The original automatic verdict
`REVIEW / NO_GOOD_TEMPLATE` remains preserved, while the final verdict is
`BAD` with `decision_source=manual_override` and Elena's reason.

Only this mode's final classification/selection changed. All 768 N75 input
fingerprints match; preliminary `rule_results.csv` is byte-identical and
every other mode row is unchanged. The other three shot output trees are
unchanged. The original N75 directory is backed up under
`sort_outputs/before_n75_manual_review_20261007/`. All 20 reviewed mode
fingerprints and all 19 accepted-list paths were verified.

Across the four shots: 3,007 inputs, 654 TAE-like, 63 mixed, 2,290 EAE-like,
**698 BAD and 19 selected GOOD**, zero INVALID or final REVIEW. All 44 earlier
crossing-review modes remain automatically BAD. No gate, configuration,
training-label or known-invalid-registry changes were made.

The reusable override is also stored in the installed N75 output. Future
reruns must explicitly supply it:

```tcsh
python scripts/sort_shot_mixed.py --method rules \
  --shot_dir "$NOVA_DITW_ROOT/nstxuG121123N75" \
  --out_dir /path/to/sort_outputs/nstxuG121123N75 \
  --manual_overrides /path/to/sort_outputs/nstxuG121123N75/manual_overrides.csv
```

Both inventories mark the four shots `sorted_rules_good_reviewed`, referring
to the selected GOOD list rather than all rejected modes. Membership remains
181 processed plus 14 training; total installed selections are now **8,337**.
The 123 other pending-review shots contain 6,522 selected GOOD modes. The
four held shots and empty entry are unchanged.

The original [20-mode automatic export](../released4_rules_20261007/good_tae_final_batch.csv)
and its receipts are preserved as historical evidence; use the accepted
19-mode list above for the current reviewed selections.
