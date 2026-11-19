# NOVA surrogate HDF5

`scripts/export_nova_surrogate.py` exports the fixed shot inventory, mapped
equilibrium scalars, normalized TAE continuum boundaries, and existing selected
GOOD mode scalars. It does not rerun classification or change the accepted
catalog. Requires NumPy, SciPy and h5py; no RF/CNN model or NOVA executable.

The user approved both `r` and `psi_N=r**2`, and reconstructed **NOVA** `iden=4`
ion mass density for all populated cases. NOVA-C `otaep`/`outb_*` densities are
not substituted. Two-dimensional mode amplitudes and lobe counts remain deferred.

Version 2 adds an explicit, audited reconstruction path for overwritten
continua. It preserves the saved selection fingerprint separately from the
fingerprint of the mode and reconstructed continuum actually exported. Exact
original pairs remain the first choice. A reconstruction requires a listed
scope, pinned continuum and mode hashes, unchanged supporting evidence, and
a matching saved selection identity. It does not rerun classification or
certify harmonic convergence. Existing input holds remain in force.

## NOVA-filter zero records (v3)

The current handoff is under `research_archive/nova_surrogate_export_20261117_v3/`:

| File | Coverage |
| --- | --- |
| `nova_surrogate_v3_complete_cases.h5` | **193 cases**, all n=1–10; **8,788 modes**, **545 K=0 records** |
| `nova_surrogate_v3.h5` | 200 inventory IDs / 199 populated equilibria; all **8,943 modes**, 1,974 continua, **564 K=0 records** |

The full file has 1,966 usable K/mode records. Among the 198 processed cases,
only the five with existing scientific holds remain incomplete. E205059A01t025
and the empty nstxu_202806 entry are the other two inventory cases. Previous
v1/v2 files remain unchanged. The v3 handoff includes a reader, confirmation
manifest, coverage tables and validation summaries.

On November 17 the user confirmed that NOVA removes singular/large-amplitude
grid-scale numerical modes before writing the mode files. For the identified
23 empty case/n directories in 13 processed cases, zero files therefore means
zero modes surviving NOVA's internal filter. Those records can be valid K=0
targets without a downstream classifier summary. This supersedes the earlier
unknown-K treatment for these records.

`configs/nova_confirmed_empty_scopes_20261117.json` records the user's statement,
the named scopes and their continuum hashes. The shared confirmation logic
checks that each directory still exists, contains no `egn*` entries, and has
the expected continuum. Existing holds and missing equilibrium/continuum
records cannot be released by this rule. No saved mode classifications change.

New records use `status=available_nova_filtered_zero`, `K=0`,
`has_good_tae=0`, `mode_targets_valid=True`, empty mode arrays, and
`continuum_status=available_upstream_zero`. `zero_K_provenance` is
`nova_internal_filter`; `zero_K_evidence_json` records the confirmation and
input checks. `catalog_K=-1` still correctly records that no downstream
classifier summary exists. The original continuum provenance and radial masks
remain unchanged; no mode/continuum pair fingerprint is invented.

## Previous v2 handoff

`research_archive/nova_surrogate_export_20261117/nova_surrogate_v2.h5`
contains all **8,943 selected catalog modes**, 200 inventory IDs, 199 populated
13-parameter sets, and 1,974 available case/n continua. There are 1,943 usable
K/mode records, including **541 verified K=0 records**. All 4,426 modes withheld
in v1 are restored. The added provenance covers 638 scopes with selected
modes and 104 zero-GOOD scopes; the latter had continuum files already but
needed historical-version reconciliation.

The separate `nova_surrogate_v2_complete_cases.h5` contains **180 equilibrium
cases complete for n=1–10**, with **8,491 modes and 470 K=0 records**.
All 70,540 copied datasets pass exact comparison with the full file.
It retains the radial masks: 108 cases have all
ten continua valid throughout the nominal r=0.01–0.99 interval, while the
remaining cases also have masked samples inside it. The full inventory file
keeps usable n records from the other cases. Its remaining unavailable mode
targets are 31 scopes without an installed selection, 16 existing held scopes,
and ten scopes in the empty inventory case. None is labeled K=0.

The independent handoff audit checks exact catalog membership and frequencies,
saved full-precision radial scalars, family/review metadata, and original
selection fingerprints. It also confirms that all 4,517 v1 mode targets,
1,201 v1 paired continua, and all equilibrium parameter/profile values remain
unchanged. The previously delivered 94-case v1 file retains its original hash.
Full-file mode families are 8,941 TAE and two mixed; review states are 865
reviewed, 6,590 pending and 1,488 unknown.

The registry distinguishes 459 reconstructed scopes with historical centroid
checks, 228 with crossing checks, and 55 with settings-only evidence. All 104
new zero-GOOD scopes pass file-membership and saved-scalar checks for 3,869
modes; 1,766 saved continuum crossings in 93 of those scopes reproduce to
3.46e-12 maximum relative error. Five G-shot N1 scopes require their historical
m=0 ranges rather than the common -3–108 range; see the archived audit and
[project state](project_state.md). The extra zero-GOOD reconstructions remain
project-local and have not been published to DiTw.

`README.txt`, `read_surrogate.py`, product summaries, coverage/case-list CSVs,
and `handoff_validation.json` accompany the files. The products are
self-contained; stored source paths are provenance, not runtime dependencies.

## Previous handoff

The November 16 product is
`research_archive/nova_surrogate_export_20261116/nova_surrogate_v1.h5`
(about 49 MiB), with 200 inventory IDs, 199 populated parameter sets, 1,974
available case/n continua and 4,517 verified selected mode targets.
Of the catalog's 8,943 modes, 4,426 lack matching input pairs in the available
live/training copies and are withheld. Coverage details are in the companion
`nova_surrogate_v1.scopes.csv`; no input holds or catalog labels were changed.

The separate handoff file `nova_surrogate_v1_complete_cases.h5`, in the same
directory, contains **94 equilibrium cases with all ten n records available**:
940 case/n combinations and 3,584 selected modes, including 320 valid K=0
combinations. It preserves the source data and metadata exactly and changes
only case membership. Each retained case has all 13 scalars, available continuum
data and verified saved K/mode targets for n=1–10. There are 69 non-G cases and
25 G cases; time slices/equilibrium variants count as separate cases.

Completeness across n does not imply a continuum value at every radial point.
All profiles retain the native 201-point r grid and their validity masks;
the nominal datcon interval is r=0.01–0.99, and some tails are also masked.
49 of the 94 cases have all ten continua valid throughout that nominal interval;
the others need the saved masks within it. No missing values are extrapolated.
Mode arrays are empty when K=0. The `.cases.csv` companion lists each retained
case's ten K values and valid radial-sample counts; `.summary.json` records
source/product hashes and validation. The HDF5 attributes state the filter.
`extract_complete_cases.py` in the artifact directory reproduces this subset.
Selection and review status remain distinct: 540 exported modes are reviewed,
2,744 pending and 300 unknown. This availability subset is not a random sample
of the complete database.

## Generate and read

From the repository on Flux (`tcsh`):

```tcsh
source configs/paths/nova_paths.flux.csh
python scripts/export_nova_surrogate.py \
  --data-root "$NOVA_DITW_ROOT" \
  --snapshot-root ../data_mixed \
  --reconstruction-manifest research_archive/nova_surrogate_export_20261117/reconstruction_registry.json \
  --confirmed-empty-manifest configs/nova_confirmed_empty_scopes_20261117.json \
  --output research_archive/nova_surrogate.h5
python scripts/extract_complete_surrogate_cases.py \
  --input research_archive/nova_surrogate.h5 \
  --output research_archive/nova_surrogate_complete_cases.h5
```

The output filename must be new. The exporter also writes `.summary.json`
(counts, validation, product SHA-256) and `.scopes.csv` (one row per case/n,
with availability and detailed issues). Optional snapshot roots are searched
only for named input pairs whose bytes match the installed selection's saved
fingerprint. It never recursively searches data roots. The default sort root
is `../sort_outputs`; override with `--sort-root` when needed.
Omit `--reconstruction-manifest` to permit only exact saved pairs. The
complete-case extractor copies case groups unchanged, validates all copied
datasets against the source, and writes `.cases.csv` and `.summary.json`
companions. Both commands require new output filenames.

To apply only the confirmed zeros to the preserved v2 baseline, without
reloading or recomputing mode/scalar/continuum arrays:

```tcsh
python scripts/confirm_nova_empty_scopes.py \
  --input research_archive/nova_surrogate_export_20261117/nova_surrogate_v2.h5 \
  --output research_archive/nova_surrogate_v3.h5 \
  --manifest configs/nova_confirmed_empty_scopes_20261117.json \
  --data-root "$NOVA_DITW_ROOT"
```

The versioning command checks the baseline hash, changes only six status/count
fields per confirmed scope and adds provenance. Every other original dataset
and its attributes are compared exactly to the source. It writes a new file,
summary and coverage CSV; the source is preserved. The standard exporter uses
the same confirmation logic when `--confirmed-empty-manifest` is provided.

```python
import h5py
import numpy as np

with h5py.File("nova_surrogate.h5", "r") as f:
    case = f["equilibria/nstxuE202855A01t020"]
    names = f["parameter_names"].asstr()[:]
    inputs = np.array([case["parameters"][name][()] for name in names])
    group = case["n_01"]
    if group["continuum_available"][()]:
        mask = group["continuum_valid"][:]
        psi = group["psi_N"][:][mask]
        lower = group["omega_minus"][:][mask]
        upper = group["omega_plus"][:][mask]
    if group["mode_targets_valid"][()]:
        K = int(group["K"][()])
        omega = group["modes/omega"][:]
        location = group["modes/psi_N_loc"][:]
        width = group["modes/psi_N_width"][:]
```

Keep an entire physical discharge, including all its time slices/equilibrium
variants, in one training/validation/test partition. Case IDs are preserved
verbatim; reducing them to a discharge number would merge distinct equilibria.

## Layout and units

```text
parameter_names[13]
case_ids[number_of_inventory_cases]
provenance/                         # repository input hashes, source roots
equilibria/<case_id>/
    equilibrium_id
    inventory_status
    parameters_valid
    parameters/<name>              # 13 scalar datasets, units and definitions
    metadata/                      # normalization, axis estimates, hashes
    equilibrium_profiles/          # supporting q, pressure, density and grids
    n_01/ ... n_10/
        n_tor
        status, status_detail
        continuum_status, continuum_available
        r[Nr], psi_N[Nr]
        omega_minus[Nr], omega_plus[Nr], continuum_valid[Nr]
        continuum_path, continuum_source_root, continuum_sha256  # if available
        continuum_provenance       # exact_saved_pair, historical_reconstruction, standalone_current
        reconstruction_evidence_json  # reconstructed scopes only
        mode_targets_valid
        catalog_K                  # saved selection count, -1 if no n summary
        K                          # usable exported count, -1 if unknown/withheld
        has_good_tae               # 1=yes, 0=no selected GOOD TAE, -1=unknown
        zero_K_provenance, zero_K_evidence_json  # confirmed NOVA-filter zeros
        modes/
            mode_id[K], path[K]
            omega[K]
            r_loc[K], r_width[K], psi_N_loc[K], psi_N_width[K]
            family[K], review_status[K], rule_config[K], manual_override[K]
            source_root[K], input_fingerprint[K]
            selection_input_fingerprint[K], mode_sha256[K], pair_provenance[K]
```

All frequency arrays contain **native normalized omega**, not omega squared
or Hz. `datcon` stores omega squared; the shared loader's repaired boundaries
are square-rooted exactly once. Negative, unordered, undefined and masked
samples become NaN with `continuum_valid=False`. Missing intervals are not
extrapolated. The preprocessing version is saved per n.

The reference is `omega_ref = V_A_ref / (q_lcfs * R_v)`, with
`V_A_ref = B_v / sqrt(mu0 * rho_axis)`. Metadata records the reconstructed
reference in rad/s and the constants used. `rho_axis` is physical mass density
in kg/m³ and **does not include mu0**. These normalization metadata do not
convert the exported targets or assert the identity of every historical binary.

Both coordinate arrays refer to the same native samples. The grid is uniform
in `r`, so `psi_N` is nonuniform. Coordinate conversion does not resample the
continuum. Supporting mapped q/pressure have their own `psi_N_mapped` grid;
its first point is near the axis, not zero. Density has its processed TRANSP
knots and its reconstructed values on the mode grid.

## Thirteen equilibrium inputs

| Dataset | Unit | Definition/source |
| --- | --- | --- |
| `a` | m | Half the LCFS midplane width at mapped `Zmag` |
| `R_v` | m | Midpoint of those two midplane intersections |
| `delta` | 1 | Mean upper/lower mapped LCFS triangularity relative to `R_v` |
| `kappa` | 1 | Mapped LCFS height divided by `2a` |
| `B_v` | T | Magnitude of TRANSP `BZXR/(100 R_v)` |
| `q_0` | 1 | First two mapped `qa` points extrapolated linearly in `psi_N` to zero |
| `q_lcfs` | 1 | Last mapped `qa` point |
| `q_min` | 1 | Minimum including the estimated axis and mapped samples |
| `p_axis` | Pa | First mapped `pa` point times `B_v²/mu0`; near-axis proxy |
| `rho_axis` | kg/m³ | Reconstructed NOVA ion-species mass sum extrapolated to axis |
| `delta_axis` | 1 | `(Xmag-R_v)/a` |
| `psi_N_qmin` | 1 | Flux at the sampled q minimum; exactly zero for an axis minimum |
| `rho_half_over_axis` | 1 | NOVA spline `rho(psi_N=0.5)/rho_axis` |

For an axis minimum, `q_min=q_0` exactly. Only a relative `1e-12` roundoff
tolerance collapses equality; no finite reversed-shear threshold is imposed.
The minimum is not a fitted continuous minimum. Upper/lower triangularity,
physical axis location, near-axis q and pressure extraction method, and mapped
rotation flag `imach` remain in metadata. The five rotating mapped cases are
not silently filtered.

`src/nova_equilibrium.py` implements the previously audited species smoothing,
boundary-to-center flux mapping and axis/edge extrapolation. It interpolates
normalized mass density with a cubic spline whose endpoint derivatives match
NOVA `depose`. Independent comparison with compiled original Fortran routines
for all 199 populated cases gives a maximum absolute normalized-density
difference of `3.35e-10`, including knots, 201 mode-grid points and `psi_N=0.5`.
See [input sources](surrogate_input_sources.md) for the species recipe and the
NOVA/NOVA-C difference.

## Mode features and eligibility

Location is the amplitude-squared centroid of `xi_psi`, summed over all
harmonics and native radial samples. Width is the RMS deviation about that
centroid, using the same discrete weights. They are not a peak position,
FWHM, or volume-weighted kinetic-energy moment.

For coordinate `x`, the definitions are
`x_loc = sum(|xi|² x)/sum(|xi|²)` and
`x_width² = sum(|xi|² (x-x_loc)²)/sum(|xi|²)`.
The shared feature implementation retains its tiny denominator regularizer.
Consequently `psi_N_loc = r_loc² + r_width²` to numerical precision.
`psi_N_width` requires the original mode amplitudes and is computed separately;
it is not `r_width²`. Values are recomputed without the compact catalog's
four-significant-digit rounding. Modes are sorted by frequency, then filename.

`mode_targets_valid=True` is required before using `K`, presence, or mode
targets. A verified processed zero-GOOD selection gives `K=0` and empty arrays.
Missing, held or unmatched input pairs give `K=-1`, presence `-1`, and empty
mode arrays. `catalog_K` preserves the old selection count where one exists.
For a zero-GOOD scope, an installed n summary and a matching saved classified
input pair anchor the continuum version. An audited reconstruction can also
anchor it, but needs explicit zero-selection evidence: complete classified
file membership, agreement with saved mode scalars, and historical continuum
checks. This does not reclassify rejected modes or prove absence of a physical
TAE. User-confirmed NOVA-filter zeros provide a separate valid K=0 source as
described above. An unlisted missing-summary record remains unknown.

`continuum_available=True` permits use of the standalone continuum and its
sample mask. `available_unpaired` means the current continuum is available
but cannot be joined to the saved K/mode targets. `available_paired` identifies
the continuum used by the exported mode targets. All registered invalid or
held scopes are masked for both tasks. Scalar availability does not release
an input hold. A wholly empty inventory case retains NaN parameters, empty
coordinates and explicit unavailable status.

The accepted catalog's two mixed-family entries remain included, identified
by `family`; historical rule versions and review status are retained. GOOD
means workflow selection, not necessarily individual human review. Pending
review and known training-label conflicts are not silently adjudicated.
`input_fingerprint` hashes the mode/continuum bytes actually used in this
export; `selection_input_fingerprint` retains the original saved fingerprint.
They agree for `pair_provenance=exact_saved_pair` and can differ for
`reconstructed_continuum`. `mode_sha256` pins the mode independently.
For reconstructed scopes, `status=available_reconstructed`; the evidence
record identifies settings and historical centroid/crossing checks or
settings-only evidence. The complete registry and its hash are embedded in
`provenance`. These records distinguish numerical reconstruction from recovery
of the original bytes and do not certify convergence.

Signed lobe counts remain unassessed. The file records that fact in mode-group
metadata and does not invent a peak threshold or a count. A later amplitude
export must retain physical poloidal `m` labels, including zero/negative m.
