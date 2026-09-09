# Incremental disagreement review, 2026-09-09

Start with **`to_review.csv`**, containing just 13 newly appearing cases that
have no recorded matching approval. It includes the current method labels,
rule reason, model scores, and input fingerprint. The explicit empty `label`
column prevents the viewer from treating a method's decision as a human label.

The latest list covers 27 shots. The comparison therefore combines the
previous fresh twelve-shot `disagreements.csv` (132 rows) and fifteen-shot
`rf_cnn_disagreements.csv` (93 rows, using its v5 decisions). Comparing with
the twelve-shot list alone would incorrectly present older cases as new.

| Change | Count |
|---|---:|
| Previous disagreements across 27 shots | 225 |
| Current disagreements | 229 |
| Retained, with unchanged decisions, rule reasons, and input fingerprints | 212 |
| Newly appearing | 17 |
| Of those, already explicitly approved as GOOD after the repair | 4 |
| New cases remaining to review | 13 |
| No longer disagreements | 13 |

All changes occur in the fresh twelve-shot cohort; its disagreement count
changes from 132 to 136. The older fifteen-shot list remains unchanged.
The 13 new review cases remain rules BAD, while RF-CNN changes BAD to GOOD:
five in E203262A04t018, seven in E204645A16t015, and one in E205040A01t016.
Only matching input fingerprints and explicit user GOOD decisions from
`../user_review.csv` are used to exclude the four approved modes.

`changes.csv` contains all 30 additions/removals with before/after labels,
rule reasons, routing, fingerprints, and review status. Removed disagreements
can represent method agreement or routing to EAE-like; the latter is not a
GOOD/BAD adjudication. Changes in probabilities alone do not make a retained
case a new classification disagreement.

The user's current `disagreements_elena.csv` is preserved. Of its 33 question
rows, 31 still appear in the current list. E204645 N7/6025 and N8/5775 no
longer disagree. These carried-over questions remain in that original file;
the new 13-row list does not duplicate them or infer answers to them.

Example viewer command from the repository root (Flux/tcsh):

```tcsh
python viz/view_modes_csv.py \
  audits/continuum_monotonic_tail_20260908/disagreement_delta_20260909/to_review.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

Recreate with `../compare_disagreements.py --out-dir <directory>`. The script
uses the saved publication receipt for old/new export locations; `summary.json`
records input hashes, including the user question-list snapshot it read.
