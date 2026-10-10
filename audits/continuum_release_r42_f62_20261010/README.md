# R42 and F62 released after visual review, October 10, 2026

Elena reviewed all **21 changed-input TAE-like/mixed modes** from the
[recalculation check](../last3_recalculated_20261010/README.md): 16 in R42 and
five in F62. All look grid-scale numerical or have large axis spikes. She
can see continuum-crossing offsets but judges further accuracy unnecessary
for these unsuitable modes, and explicitly releases both shots for production
rules processing. Both inventories now mark them `ready_for_rules`.

The [reviewed-mode record](reviewed_modes.csv) retains all 21 fingerprints,
types and the shared user assessment. The exact pathology is not assigned
individually. The release does not assert corrected continuum correspondence
or establish that the rules have already rejected these modes. Their eventual
sorter outcomes should be compared with this review. No sorter overrides,
training labels or production outputs were changed in this step.

**E205059A01t025 remains on hold:** Elena reconfirms that the old N1/N2 cases
still require recalculation. R42's earlier acceptance of n>2 is retained.
Diagnostic offsets, F62's inner-boundary observation and potential EAE
findings remain recorded in the source audit.

All recorded N1/N2 sources and all six inventories still match the October 10
snapshot; all 21 reviewed fingerprints were checked. The
[release record](release.json) preserves the user's statement, source-audit
hashes, verification and before/after inventory rows.

[Five entries remain unprocessed](remaining_unprocessed.csv):

- **Ready:** `nstxuG121123R42`, `nstxuG142301F62`, `nstxuG142301D46`.
- **Held for N1/N2 recalculation:** `nstxuE205059A01t025`.
- **Empty:** `nstxu_202806`.

No sorting was run. Membership remains 181 processed post-training shots plus
14 active training shots, with 8,337 selected GOOD modes.
