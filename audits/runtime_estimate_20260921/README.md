# Remaining DiTw rules-sorting runtime estimate

The user subsequently authorized this 122-shot subset. The
[completed batch](../remaining122_rules_20260921/README.md) installed 118 shots
and held four newly discovered NaN-input cases. The estimate below records
the pre-run inventory and known flags.

For planning, allow **1–2 hours with four shot processes**, or **2–3 hours
sequentially**, including automated result checks and output handling. These
are estimates for current rules on Flux, excluding human visual review,
RF-CNN inference, NOVA recalculation and queue delays.

The 200-shot inventory contains 40 processed cases and 14 disjoint active
training shots, leaving 146. Excluding the 20 outstanding shots from the
21-shot priority N1 list, NaN-flagged D46/M21, and the empty legacy entry
leaves **124 candidates and 73,729 current mode files**. D46 is already in
the N1 list, so the union excludes 22 shots. Recalculated E202806A02t045 is
already among the 40 processed cases. Holding the two secondary N1-review
cases R48/U27 as well leaves **122 shots and 71,656 modes**.

Counts use direct N1–N10 directories in the DiTw root. All candidate populated
directories have matching continuum files. This was not a fresh all-mode NaN
or continuum-alignment audit: the exclusion list uses recorded flags, and
several candidates still have limited or inconclusive N1 evidence.

Two previously processed shots were timed sequentially with the current v13
rules, using the existing compatible environment and one numerical thread
per process. Results went into ignored benchmark directories; production
exports and inventory statuses were not changed.

| Shot | Raw modes | TAE-side modes | Elapsed |
| --- | ---: | ---: | ---: |
| E202806A02t045 | 464 | 244 | 44.3 s |
| E205055A01t022 | 1,017 | 155 | 64.2 s |

Scaling the pooled rate to 73,729 modes gives about **90 minutes sequentially**;
the individual sample rates give **78–117 minutes**. Four processes would
ideally reduce that to about 20–30 minutes, but shared filesystem contention,
shot composition, validation and output handling justify the wider planning
budget above. Parallel scaling has not been measured in this estimate.

- [Remaining-shot count and recorded holds](remaining_shots.csv)
- [Timing measurements, calculation and source hashes](estimate.json)

This estimate does not authorize or start the remaining production batch.
Flagged recalculations need review before being included in a later run.
