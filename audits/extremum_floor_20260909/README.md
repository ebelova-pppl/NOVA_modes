# Extremum-exception clearance and width floors, 2026-09-09

**Adopted decision:** the user subsequently selected clearance **>0.1% only**,
with no additional minimum-width requirement. Production v8 implements
`0.001 < ext_df_gap <= 0.04` in the existing interior extremum exception.
The comparison below is the pre-adoption audit. Its six shot and one training
changes are the accepted clearance-only projection; the 1.1 option was not
adopted. Training labels remain unchanged, including H47 N7/2530.

All 27 production rules exports have been regenerated and published as v8.
The six changes exactly match the earlier clearance-only projection:
`clearance_adopted_changes.csv`. The 4,187 evaluated modes retain every
measured feature; only the lower-cut metadata and its exception flag change.
GOOD totals are 950 before duplicate removal and 944 selected (six fewer
each). `clearance_shot_summary.csv` records every shot.

All 2,390 training inputs were recomputed. The sole new decision conflict is
H47 N7/2530, documented in `clearance_training_change.csv`: 542 labeled GOOD
survive, 33 are rejected; the 25 labeled BAD survivors are unchanged. No
training truth label was edited. All 167 tests passed, including strict
boundary equality, the next representable value above it, old-preset
tangency, and canonical workflow/CLI checks.

The refreshed `current_disagreements.csv` has **228 rows**, compared with
226 in the preceding R06-invalid comparison. Only `disagreements_added.csv`
(four rows) needs new review; `disagreements_removed.csv` contains two rows.
The user's `disagreements_elena.csv` is preserved.

During regeneration, eight formerly INVALID C50 N1 inputs were absent from
the raw directory. `absent_invalid_inputs.csv` records their previous
fingerprints. Rules now inventory 19,317 inputs, including 772 INVALID,
with 65 remaining C50 N1 entries. RF-CNN exports remain byte-for-byte
unchanged and still include the eight historical INVALID rows. No valid
input disappeared. C50 exclusion diagnostics also update their registry
provenance from v1 to v2; their exclusion reason is unchanged.

`clearance_verification.json` records these counts and source/export hashes;
`clearance_publication.json` records the 27 published and backup directories.
The existing outputs are backed up under
`before_extremum_clearance_v8_20260909/` in the rules root. Staged exports and
runtime logs remain ignored under `outputs/review_extremum_clearance_v8_20260909/`.

The user proposed relative frequency clearance of 0.1% and a minimum width
of 1 or 1.1 grid intervals. This audit interprets those as **strict lower
requirements for the existing interior-envelope extremum exception**:

```text
ext_df_gap > 0.001
AND connected_total_energy_FWHM_grid > width_floor
```

The existing upper clearance limit 0.04, radial alignment limit 0.02, and
interior-gate applicability (`r_peak<=0.5`, connected W FWHM<=2) remain.
This is a projection only; no production code, thresholds, labels, raw data,
or canonical shot outputs have changed. Wider/core-external modes do not
need this exception and are unaffected by this proposal.

`ext_df_gap` is signed frequency clearance divided by mode frequency, positive
on the gap side of the matched upper minimum/lower maximum. Width is the
connected FWHM of `W=sum_h A_h^2` in native radial intervals, not harmonic
amplitude width or the global radial standard deviation. All evaluable
audited modes have nr=201 (one interval is 0.005).

| Width must exceed | Newly rejected current shot GOOD | Newly rejected training GOOD | Newly rejected training BAD |
| ---: | ---: | ---: | ---: |
| 1.0 | 6 | 1 | 0 |
| 1.1 | 11 | 4 | 1 |

These are paired with clearance >0.001 in both rows. The 1.0 floor alone
rejects no current survivor in either cohort: an interior peak of a
nonnegative sampled W profile has interpolated FWHM at least one interval.
The six shot/one training rejections therefore come entirely from clearance.
Increasing the floor to 1.1 adds five shot rejections and four training
rejections (three labeled GOOD and one labeled BAD).

## Training conflicts and existing reviews

| Mode | Training label | W FWHM, intervals | ext_df_gap | Additional condition that fails |
| --- | --- | ---: | ---: | --- |
| E202855A01t020 N8/9221 | GOOD | 1.024460 | 0.01757128 | width >1.1 |
| G142301W29 N8/2304 | BAD | 1.056210 | 0.00202066 | width >1.1 |
| G142301W29 N9/1982 | GOOD | 1.094519 | 0.00183281 | width >1.1 |
| G142301H47 N7/2530 | GOOD | 1.232528 | 0.00006176 | clearance >0.001 |
| G142301H47 N9/1813 | GOOD | 1.074347 | 0.00123481 | width >1.1 |

The 1.1 proposal also rejects E203262A04t018 N10/6623 (W width 1.070820,
clearance 0.00651966), which has fingerprint-matched user GOOD approval in
`../continuum_monotonic_tail_20260908/user_review.csv`. Its clearance passes;
only the new width floor causes the conflict. These are existing labels and
reviews, not independent correctness determinations for this audit.

Of the five modes that prompted this discussion, the 1.0 option rejects
only F03t017 N8/8950. The 1.1 option additionally rejects F03t017 N8/6836
and N10/7424. E205040A01t016 N6/2836 and E204645A16t015 N6/4765 survive both.

`shots_to_review.csv` contains all 11 possible shot changes, including flags
showing which fail each width/clearance requirement and the earlier user
review. `training_to_review.csv` contains the five affected training modes.
The two `newly_rejected_width_*.csv` files give exact changes for each option.
Each row retains its raw input fingerprint and original decision.

## Verification and reproduction

The current 27-shot exports contain 19,325 inputs: 780 INVALID, 14,358 EAE,
and 4,187 rule-evaluated. Of 956 automatic GOOD before deduplication (950
selected), 58 rely on the interior extremum exception. Every one of these
58 was freshly loaded and fingerprint checked; all complete v19 feature
records and preliminary decisions exactly match the saved exports. All
possible shot changes were selected representatives. No deduplication rerun
was performed; projected changes refer to decisions before deduplication.

All 2,390 active training rows were reloaded through shared validation,
continuum preprocessing, routing and rule evaluation. The baseline exactly
matches the prior v7 audit: 543 labeled GOOD and 25 labeled BAD survive;
32 GOOD and 1,763 BAD are rejected; 26 BAD route to EAE; one BAD input is
INVALID. Tightening this exception cannot recover an already-rejected mode.

Raw recomputation uses the unchanged v19 engine defaults with exact-point
crossing gate disabled, matching production v7. Source/config/training and
all 27 input-manifest hashes are checked before and after the audit and
stored in `summary.json`. Full per-training-row measurements stay ignored
under `outputs/review_extremum_floor_20260909/`.

```text
python audits/extremum_floor_20260909/audit.py --rules-root /path/to/sort_outputs --training-root /path/to/training/data --out-dir outputs/review_extremum_floor_20260909
```

The 1.1 option remains unadopted. The user selected the clearance-only policy
after seeing the H47 N7/2530 conflict and the alternative width impacts.
The original impact script/receipts are historical; reproduce that pre-v8
audit from checkout `f018a25`. Current default engine behavior includes the
adopted clearance floor.

`adopt_clearance.py` stages all 27 canonical rules runs with the active RF
representative ranker, compares every prior feature/decision, and rechecks
all training inputs against the saved pre-adoption measurements. `verify`
reuses completed staged exports; `publish` requires matching source/tree
hashes and keeps previous directories under
`before_extremum_clearance_v8_20260909/`. RF-CNN outputs are hash checked and
preserved. The verification script's docstring gives the portable commands.
