# F83 and K79 released for rules processing, October 7, 2026

**Later processing:** F83/K79 and N75/B85 are now
[sorted and installed](../released4_rules_20261007/README.md). The release
snapshot below is preserved; four held shots plus one empty entry remain.

Elena reviewed all 19 flagged TAE-like modes in F83 (eight) and K79 (eleven).
They are grid-scale numerical modes or have other morphology problems,
mostly near axis. She released both shots for production rules processing.
Both live inventories now use `ready_for_rules`, with processing still
pending. The [released list](released_shots.csv) preserves potential EAE
issues, mixed findings and log-coverage gaps separately.

The [reviewed-mode manifest](reviewed_modes.csv) records the user assessment
and input fingerprints. All 211 source files in F83/K79 N1--N2 and the mode
inventories match the October 5 snapshot. No individual sorter override or
training-label change was made, and no sorting was run in this release step.

## D46 remains on hold

D46's active files are changing. At the release check, N1 contained one file,
`egn01w.3009E+02`, and N2 contained none, compared with 57 and 45 on October 5.
The user could not complete its review. Its inventory status is now
`input_update_pending`; input availability and continuum correspondence must
be checked before processing. The earlier NaN issue remains resolved.
The cause of the current file changes has not been established.

The old D46 rows in the October 7 review CSV are historical and their files
are currently unavailable. The F83/K79 review is complete.

## Current inventory

[Nine entries remain unprocessed](remaining_unprocessed.csv): **two ready,
six held and one empty**. The held shots are E205059A01t025, G121123N75,
G121123R42, G142301B85, G142301F62 and G142301D46 (all prefixed `nstxu`).
Completed membership remains 177 post-training shots plus 14 training shots.

[Receipt](receipt.json) records the input checks, D46 availability observation,
file hashes and exact before/after changes in both live inventories.
