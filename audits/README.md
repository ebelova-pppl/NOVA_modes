# Retained scientific evidence

This directory keeps compact dated summaries and selected evidence needed for
current results. Detailed runs, intermediate data, and most one-off scripts
are local archives; see [research records and recovery](../docs/research_archive.md).
The explicit allowlist is in `.gitignore`.

| Record | Purpose |
| --- | --- |
| [Shot inventory](main_dataset_shots/shot_status.csv) and [active holds](main_dataset_shots/active_shot_holds.csv) | Catalog coverage and outstanding input issues |
| [Input issues](main_dataset_input_issues/README.md) | Recorded live-input problems |
| [39-shot review](pilot39_manual_review_20260914/README.md) | Fingerprinted decisions and reusable manual overrides |
| [40-shot collection](processed40_20260915/README.md) | Accepted list supplying scoped E202806A02t045 review evidence |
| [14-shot review](released14_manual_review_20261007/README.md) | Accepted TAEs, BAE assignments, and manual corrections |
| [Four-shot review](released4_manual_review_20261007/README.md) | Accepted TAEs and the N75 manual correction |
| [Five-mode diagnostic](nstxu_204202_five_mode_review_20261025/README.md) | Explicit non-blind diagnostic evidence used by the catalog |
| [N1 input alignment](n1_training_alignment_20260910/README.md) | Exclusion evidence, suspended labels, and tested alignment utility |
| [Database alignment](n1_database_alignment_20260910/README.md) | Tested database audit utility |
| [Production v13 adoption](morphology_v13_20260913/README.md) | Compact validation, changes, and per-shot totals |

Other retained summaries document the scientific development history. Archived
paths and procedures in those summaries require the historical artifacts; they
are not all executable from a fresh checkout. Current commands are in
[getting started](../docs/getting_started.md) and the
[script reference](../scripts/README.md).

The [review-source registry](../configs/catalog_review_sources.csv) is the
authority for catalog review attribution. A batch summary or accepted-list
count alone does not establish individual visual review. To reproduce curated
shot selections, explicitly pass the applicable `manual_overrides.csv` to the
canonical sorter; do not silently omit manual BAD or BAE decisions.
