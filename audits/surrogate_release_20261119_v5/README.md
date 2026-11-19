# Final populated-case surrogate release, November 19, 2026

The user authorized releasing E205059A01t025, completing the previously
approved R06 release, publishing corrected continua, and updating HDF5.
The handoff is
`research_archive/nova_surrogate_export_20261119_v5/nova_surrogate_v5_complete_cases.h5`:
**199 complete cases, 9,001 selected modes, 579 K=0 case/n records**.
Every populated inventory case has all 13 parameters and all ten continuum
and mode-count records. The empty `nstxu_202806` inventory entry is excluded.
Native radial masks remain necessary.

| Case | n=1–10 K values | Selection provenance |
| --- | --- | --- |
| nstxuE205059A01t025 | 0, 1, 2, 4, 5, 7, 9, 10, 10, 10 | Current frozen production rules; review status pending |
| nstxuG133964R06 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | User rejection of all 95 N1/N2 modes; approved unchanged historical N3–N10 zero selection |

E-shot rules processed all 1,346 modes with no invalid inputs, radial-resolution
gate omissions, manual overrides or model inference. The 275 TAE-side candidates
yield 59 GOOD modes before duplicate filtering and 58 selected. N2 omega=3.27793
is selected; omega=3.21680 remains GOOD but is grouped as its duplicate.
Release authorization does not constitute individual visual GOOD labels.

R06 higher-n evidence covers 515 unchanged mode/continuum pairs: all 33
historical TAE-side candidates were rule-rejected; 482 EAE-like modes were
routed separately. The user's release followed direct correspondence checks
that did not establish a systematic higher-n spike/crossing displacement.
No new higher-n morphology labels or convergence certification are claimed.

Four numbered continua were published with atomic replacements:

| Case | File | Internal m range |
| --- | --- | --- |
| nstxuG133964R06 | N1/datcon1 | -2..26 |
| nstxuG133964R06 | N2/datcon2 | -2..47 |
| nstxuE205059A01t025 | N1/datcon1 | -5..16 |
| nstxuE205059A01t025 | N2/datcon2 | -5..27 |

The original files are preserved in the prior calculation archives, the v5
`publication/before/` tree, and adjacent DiTw `continuum_backups/<stamp>/`
directories. Replacement hashes, originals, aliases and input hashes are
verified; prior permissions are retained. Higher-n continua are unchanged.

The version-specific release preserves the old global registry, installed
historical outputs and shared 8,943-mode catalog. Those continue to protect
uncorrected copies; they are not the status of v5. The v5 archive includes
`accepted_modes_v5.csv`, full rule outputs and source inputs, release and
publication manifests, build/validation scripts and the self-contained HDF5.
`released_scopes.csv` pins the 20 released continuum hashes and K values.

The build uses the shared exporter and complete-case copier, retains v4
products and compares all 77,695 unaffected original datasets exactly.
All prior equilibrium quantities, density conventions, 8,943 selected modes,
review flags, r/psi_N coordinates and masks are preserved. No two-dimensional
mode structures or lobe counts were added. See accompanying validation JSONs
for the complete-file, subset-copy and independent live-input checks.
