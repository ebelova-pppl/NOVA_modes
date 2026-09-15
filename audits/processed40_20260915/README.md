# Accepted modes from 40 processed post-training shots

On 2026-09-15 Elena checked the recalculated E202806A02t045 rules results and
approved adding the shot to the processed list. There are now **40 processed
post-training cases plus 14 disjoint active training shots: 54 cases total**.
The added shot was processed with rules and reviewed by Elena. The other
39 cases retain their rules/RF-CNN comparison history.

- [Accepted TAE modes](accepted_tae_modes.csv): **1,726 selected representatives**.
- [Per-shot counts and methods](shot_summary.csv): all 40 processed cases.
- [Verification and inventory update receipt](verification.json).

| Count across the 40 processed cases | Modes |
| --- | ---: |
| TAE-side candidates, including mixed | 6,256 |
| Final GOOD before deduplication | 1,737 |
| Selected GOOD representatives | 1,726 |
| Final BAD | 4,519 |
| Routed EAE-like | 19,468 |
| INVALID | 707 |

All selected representatives are TAE-like. The manifest preserves the previous
1,636 accepted rows exactly and adds 90 from E202806A02t045. Paths are portable
`shot/N#/filename` keys relative to the DiTw root. Labels, frequencies,
fingerprints, automatic rule evidence and existing manual reasons are retained.
The added shot needed no manual label overrides. Training modes are not part
of this export; its counts describe the 40 post-training cases only. R06
remains a processed but wholly invalid case, with no accepted modes.

The new shot's installed file hashes match its
[verified rerun](../recalculated_e202806a02t045_20260915/README.md). The earlier
manifest matches its [manual-review receipt](../pilot39_manual_review_20260914/verification.json).
All 1,726 mode keys are unique; each selected row is final GOOD. Only
E202806A02t045's row in the main inventory changed, to
`post_training_checked=yes`, `status=checked_post_training`,
`checked_methods=rules`, with dated review notes. The G-shot inventory and
training membership are unchanged.

The original 39-shot disagreement review and its 17 overrides remain intact.
Its 348 retained AI disagreements are still the latest comparison results;
the new shot has no RF-CNN comparison. The existing manual overrides must
still be supplied explicitly when rerunning their affected shots. The
remaining full-database rollout stays deferred.
