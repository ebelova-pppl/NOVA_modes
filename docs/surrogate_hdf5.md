# NOVA surrogate HDF5 v1

`scripts/export_nova_surrogate.py` exports the fixed shot inventory, mapped
equilibrium scalars, normalized TAE continuum boundaries, and existing selected
GOOD mode scalars. It does not rerun classification or change the accepted
catalog. Requires NumPy, SciPy and h5py; no RF/CNN model or NOVA executable.

The user approved both `r` and `psi_N=r**2`, and reconstructed **NOVA** `iden=4`
ion mass density for all populated cases. NOVA-C `otaep`/`outb_*` densities are
not substituted. No two-dimensional mode amplitudes are exported in v1.

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
  --output research_archive/nova_surrogate.h5
```

The output filename must be new. The exporter also writes `.summary.json`
(counts, validation, product SHA-256) and `.scopes.csv` (one row per case/n,
with availability and detailed issues). Optional snapshot roots are searched
only for named input pairs whose bytes match the installed selection's saved
fingerprint. It never recursively searches data roots. The default sort root
is `../sort_outputs`; override with `--sort-root` when needed.

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
        mode_targets_valid
        catalog_K                  # saved selection count, -1 if no n summary
        K                          # usable exported count, -1 if unknown/withheld
        has_good_tae               # 1=yes, 0=no selected GOOD TAE, -1=unknown
        modes/
            mode_id[K], path[K]
            omega[K]
            r_loc[K], r_width[K], psi_N_loc[K], psi_N_width[K]
            family[K], review_status[K], rule_config[K], manual_override[K]
            source_root[K], input_fingerprint[K]
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
input pair anchor the continuum version. This does not reclassify every
rejected mode or prove absence of a physical TAE.

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
Input fingerprints establish exact mode/continuum bytes, not independent
historical equilibrium/executable provenance.

Signed lobe counts remain unassessed. V1 records that fact in mode-group
metadata and does not invent a peak threshold or a count. A later amplitude
export must retain physical poloidal `m` labels, including zero/negative m.
