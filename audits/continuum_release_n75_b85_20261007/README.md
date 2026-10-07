# N75/B85 released; E205059A01t025 requires recalculation

**Later processing:** N75/B85 and the earlier F83/K79 releases are now
[sorted and installed](../released4_rules_20261007/README.md). The release
snapshot below is preserved; four held shots plus one empty entry remain.

Elena completed the [42-mode TAE crossing review](../continuum_review_n75_b85_e205059_20261007/README.md)
on October 7, 2026.

| Shot (omit `nstxu`) | Review outcome | Processing status |
|---|---|---|
| G121123N75 | N2 offsets confirmed; all 16 listed modes look junky | Ready for rules |
| G142301B85 | N2 offsets confirmed; all nine listed modes look junky | Ready for rules |
| E205059A01t025 | N1 and N2 offsets confirmed; some affected modes otherwise look acceptable | Hold for N1/N2 recalculation |

N75/B85 are released because the affected TAE-like modes are unsuitable,
not because the continuum correspondence has been corrected. Elena judged
continuum recalculation unnecessary for these two shots. The
[25-mode assessment record](reviewed_modes.csv) preserves their fingerprints;
the [released-shot list](released_shots.csv) retains mixed and potential EAE
findings separately. No individual sorter overrides or training labels
were changed.

E205059A01t025 remains unprocessed with
`status=continuum_recalculation_pending`. The [hold record](held_shots.csv)
specifies both N1 and N2. The user did not identify individual acceptable
modes, so no per-mode GOOD labels are inferred from that assessment.
Recalculated inputs need a renewed correspondence check before release.
Potential EAE issues and log-coverage limits remain recorded.

All 591 mode, continuum and available log sources in the six reviewed N
groups still match the October 5 snapshot, with unchanged mode inventories.
All 42 review fingerprints were checked again. The [receipt](receipt.json)
records source checks, protected-file hashes and the five before/after
inventory-row updates (three in the main list, two in the G subset).

## Remaining processing

[Nine entries remain unprocessed](remaining_unprocessed.csv):

- **Four ready:** G121123N75, G142301B85, G142301F83, G142301K79.
- **Four held:** E205059A01t025 (N1/N2 recalculation), G121123R42 and
  G142301F62 (continuum review), G142301D46 (changing/missing inputs and
  incomplete continuum review).
- **One empty:** `nstxu_202806`.

All listed G/E names have the `nstxu` prefix. Completed membership remains
177 post-training plus 14 active training shots. No sorting was run in this
release step; production outputs and known-invalid registry entries are unchanged.
