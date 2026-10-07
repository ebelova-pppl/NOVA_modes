# Fourteen shots released for rules processing, October 5, 2026

**Subsequently completed:** all 14 released shots were sorted and installed;
the [production batch](../released14_rules_20261005/README.md) selected 76 GOOD
modes. Eight continuum holds and one empty entry remain. The release-time
status and inventory counts below are preserved as a historical snapshot.

Elena reviewed the strict TAE-like crossing shortlist and released all 14
shots for production rules processing. Almost all reviewed modes look
numerical/grid-scale or have large axis amplitudes or spikes; recalculating
their continua is not warranted. The exception is
`nstxuG142301E34/N2/egn02w.2204E+02`, which looks presentable and for which
Elena sees no continuum-crossing issue. This records her visual assessment;
the original diagnostic measurements remain available.

The [released list](released_shots.csv) contains the 14 shots, including U37,
which had zero flagged TAE-like comparisons. The reviewed list has 39 modes
in the other 13 shots. Both live inventories now mark the released shots
`ready_for_rules`; `post_training_checked` remains `no` until sorting completes.
The production workflow is `sort_shot_mixed.py --method rules` with the
existing frozen configuration and ordinary input checks. No sorting was run
as part of recording this release, and no mode-level override was created
from the aggregate review statement.

Potential EAE crossing issues, EAE log-coverage gaps and separate mixed-mode
findings remain recorded in the release CSV and the
[original review package](../continuum_small_review_20261005/README.md).
Release for TAE processing does not establish EAE/mixed continuum validity.
Q62 remains excluded from active training; the release changes its processing
readiness, not the status of its historical training labels.

## Eight remaining continuum-review shots

These are TAE-like comparisons with absolute nearest-log distance >2 grid
intervals in `0.03 <= r < 0.75`, summed over N1+N2. They are diagnostic flags,
not automatically confirmed physical resonance offsets.

| Shot (omit `nstxu`) | N1 | N2 | Total flagged TAE comparisons |
|---|---:|---:|---:|
| G142301D46 | 1 | 11 | 12 |
| G142301F83 | 1 | 12 | 13 |
| G142301K79 | 0 | 13 | 13 |
| G121123N75 | 0 | 16 | 16 |
| G142301B85 | 0 | 16 | 16 |
| E205059A01t025 | 13 | 6 | 19 |
| G142301F62 | 0 | 37 | 37 |
| G121123R42 | 2 | 44 | 46 |

The [eight-shot hold list](remaining_continuum_holds.csv) retains separate
TAE, mixed and EAE evidence. There is also the empty `nstxu_202806` entry.
Thus the [23 unprocessed entries](unprocessed_shots.csv) now comprise
**14 ready + eight continuum holds + one empty entry**. Completed membership
remains 163 post-training shots plus 14 active training shots.

## Verification

[Receipt](receipt.json) records the user's review, E34's mode/continuum
fingerprint, source hashes, and before/after rows for both inventories.
All 281 exported raw mode, continuum and available log sources associated
with the released shots still match the October 5 review snapshot. The
main and G inventories agree. Known-invalid exclusions and active training
labels are unchanged; processed membership has not increased.
