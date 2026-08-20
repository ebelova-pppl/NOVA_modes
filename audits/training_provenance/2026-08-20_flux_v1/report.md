# NOVA training provenance audit: 2026-08-20_flux_training_vs_ditw_v1

- Schema: `nova-training-provenance-v1`
- Generated: `2026-08-20T20:58:36+00:00`
- Training root: `/p/hym/ebelova/NOVA/data_mixed`
- Reference root: `/p/nstxdigtwin/energetic_particles/nova/DiTw`
- Training CSV: `/p/hym/ebelova/NOVA/NOVA_modes/training_labels/tae_like_v2_nonG.csv`
- Training CSV SHA-256: `8587aef8876c575c27f4404d44a4a45f9e46ffa210b7efc53ec67e2de149f0ad`
- Audit script SHA-256: `118e3257fef63076fc7dc492d753182db279ba855c4da6dfc49fb04112dc14a8`
- Shots: 15
- Canonical rows: 2900

## Scope

This is a read-only, byte-level comparison of the training-relevant payload for the shots named by the training CSV: all `egn*` mode files, active `datconN` files, preserved `datconN_old` backups, and `datcon_gf.txt`.
Run executables, plots, logs, and other NOVA working-directory artifacts are intentionally outside the manifest.

## Shot summary

| Shot | Canonical rows (same/different/missing reference) | All modes (same/different/training-only/reference-only) | Active datcon (same/different/training-only/reference-only) | Status |
| --- | ---: | ---: | ---: | --- |
| `nstx_120113` | 53/0/120 (+0 missing both) | 341/0/200/0 (+0 missing both) | 0/10/0/0 | `mode_and_continuum_mismatch` |
| `nstx_135388` | 344/0/0 (+0 missing both) | 1169/0/0/0 (+0 missing both) | 1/9/0/0 | `continuum_mismatch` |
| `nstx_141711` | 113/42/101 (+0 missing both) | 476/59/144/0 (+0 missing both) | 0/10/0/0 | `mode_and_continuum_mismatch` |
| `nstxuE202855A01t020` | 79/0/0 (+0 missing both) | 125/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuE204669M03t025` | 217/0/0 (+0 missing both) | 973/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuE205052A01t022` | 291/0/0 (+0 missing both) | 1486/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuG121123B12` | 135/0/0 (+0 missing both) | 637/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuG121123J38` | 174/0/0 (+0 missing both) | 620/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuG121123K51` | 16/106/86 (+0 missing both) | 633/113/91/20 (+0 missing both) | 0/10/0/0 | `mode_and_continuum_mismatch` |
| `nstxuG121123Q62` | 241/0/0 (+0 missing both) | 1186/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuG133964S31` | 74/0/0 (+0 missing both) | 830/0/0/0 (+0 missing both) | 0/10/0/0 | `continuum_mismatch` |
| `nstxuG142301H47` | 169/0/0 (+0 missing both) | 1016/0/0/0 (+0 missing both) | 0/10/0/0 | `continuum_mismatch` |
| `nstxuG142301W29` | 158/0/0 (+0 missing both) | 1060/0/0/0 (+0 missing both) | 10/0/0/0 | `active_payload_aligned` |
| `nstxuG142301Y93` | 106/0/0 (+0 missing both) | 651/0/0/0 (+0 missing both) | 0/10/0/0 | `continuum_mismatch` |
| `nstxu_204202` | 140/0/135 (+0 missing both) | 532/0/193/1 (+0 missing both) | 10/0/0/0 | `mode_mismatch` |

## Most important current differences

Canonical same-name mode files differ in: `nstx_141711` (42), `nstxuG121123K51` (106).
Canonical training modes absent from the reference tree: `nstx_120113` (120), `nstx_141711` (101), `nstxuG121123K51` (86), `nstxu_204202` (135).
Active `datconN` mismatches: `nstx_120113` (10), `nstx_135388` (9), `nstx_141711` (10), `nstxuG121123K51` (10), `nstxuG133964S31` (10), `nstxuG142301H47` (10), `nstxuG142301Y93` (10).
Preserved `datconN_old` files: `nstxuG121123Q62` (10 total, 10 different from active reference).

## Changed canonical mode diagnostics

| Shot | Changed modes | Structure unequal | Shape changed | Median `abs(delta omega / omega)` | Maximum | Parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nstx_141711` | 42 | 42 | 9 | 5.74973e-07 | 0.000113839 | 0 |
| `nstxuG121123K51` | 106 | 106 | 0 | 5.13397e-07 | 7.45523e-05 | 0 |

## Interpretation

- `identical` means equal SHA-256 content, not merely equal filename, size, or timestamp.
- `training_only` means the training snapshot contains the file but the current reference tree does not.
- `reference_only` means the current reference tree contains the file but the training snapshot does not.
- Changed same-name modes include parsed `omega`, damping, array-shape, and classifier-used mode-structure diagnostics in `file_manifest.csv`.
- `datconN_old` is paired with the corresponding reference `datconN` so a refreshed active file does not erase the prior continuum provenance.

## Artifacts

- `file_manifest.csv`: complete scoped file inventory and hashes.
- `differences.csv`: non-identical or missing subset of the manifest.
- `shot_summary.csv`: per-shot counts used in the table above.
- `run_metadata.json`: exact roots, options, schema, and artifact hashes.
- `SHA256SUMS`: integrity hashes for the generated audit artifacts.
