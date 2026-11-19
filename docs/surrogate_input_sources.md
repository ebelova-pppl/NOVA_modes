# NOVA surrogate: equilibrium input sources

Discovery date: November 10, 2026; database-wide corrections confirmed by the
user on November 13; scalar availability audited November 15 and saved
density profiles and NOVA-C central-density diagnostics checked November 16. Scope:
frequencies and mode structures for the accepted TAE database. Paths below
are relative to `$NOVA_DITW_ROOT/SHOT`.
This source map records the confirmed conventions and the discovery evidence.
The [HDF5 exporter and schema](surrogate_hdf5.md) implement the scalar,
continuum and mode-moment dataset with both `r` and `psi_N`, using reconstructed
NOVA ion mass density. Mode amplitudes and signed lobe counts remain deferred.

## Conventions established with the user

- `m` is the physical poloidal harmonic number. Its minimum may be zero or
  negative, for example `-3`; a stored array row index is not its physical m
  label. Preserve an explicit `m_values` array with each mode. For a verified
  consecutive ascending range, row `i` corresponds to `m_min + i`.
- The native frequency is normalized by
  `Omega_N = V_A_ref / (q_lcfs * R_v)`, with
  `V_A_ref = B_vacuum / sqrt(mu0 * sum_j(m_ij * n_ij(axis)))` and
  `R_v = (R_min_midplane_LCFS + R_max_midplane_LCFS) / 2`.
  `B_vacuum` is the vacuum toroidal field at this midplane geometric center.
  The species masses and densities are obtained from `transp.dat`; the
  reference density is evaluated on the magnetic axis. The scale varies
  between shots. Global LCFS R extrema cannot generally replace its midplane
  intersections, especially for strongly shaped cross sections.
- To keep units explicit, let `rho_i = sum_j(m_ij * n_ij)` denote physical
  mass density. The user's `rho_mass` in `V_A_ref = B_vacuum/sqrt(rho_mass)`
  is `mu0 * rho_i(axis)`; include `mu0` exactly once.
- The `iden=4` density profile is built from TRANSP ion species, including
  beam ions and impurities, with smoothing of selected profiles, axis
  normalization, and interpolation in normalized poloidal flux.
  The November 16 NOVA-C outputs use a mass sum that omits thermal hydrogen.
  The November 19 source now includes it; see the
  [ion-mass consistency check](#ion-mass-consistency-check-november-19).
  Retain historical NOVA-C normalization separately until corrected runs
  have been verified.
- **Density source decision, November 16:** the user selected reconstructed
  NOVA `iden=4` axial mass density for the surrogate across all 199 populated
  cases. These values are already saved as `rho_axis_kg_m3` in the November
  15 scalar audit. Use the same NOVA reconstruction for the normalized
  density profile and planned `rho(psi_N=0.5)/rho_axis` feature. NOVA-C output
  remains comparison evidence; its completion is not a dependency for the
  surrogate. This choice is consistent with the inspected NOVA continuum/
  mode density path and the user's confirmed database convention, without
  asserting verification of every historical executable.
- `qa`, `pa`, and `ga` agree with the corresponding mapped quantities in
  `mapdsk` and `mpout1` and are the profiles used by NOVA after q-solver and
  mapping. The user confirmed this and the density prescription for the
  entire database; both questions are resolved.
- Old NOVA calculations assume up-down symmetry. The future superNOVA input
  representation should retain asymmetric geometry as well.

## Scalar-first surrogate: availability audit, November 15

Alvaro's initial target is the lower and upper TAE continuum boundaries,
`omega_min(psi_N)` and `omega_max(psi_N)`, conditioned on equilibrium scalars
and `n_tor`. Possible later targets are the number of selected GOOD TAEs, K,
and their frequency, radial location, and width. A quadratic q-profile fit is
under consideration; no fit or surrogate training was performed here.

**All ten requested scalar inputs are recoverable for all 199 populated
inventory cases.** The remaining entry, `nstxu_202806`, is empty. A root-level
check found four aliases and seven analysis/derived directories beyond the
200 inventory entries, not additional independent equilibrium cases.

| Requested input | Preferred extraction | Populated cases |
| --- | --- | --- |
| a, R_v | `mapdsk.cdf: xinf,zinf`; half-width and midpoint of LCFS intersections with the symmetry midplane | 199/199 |
| delta, kappa | Same mapped LCFS; triangularity at upper/lower tips and total vertical extent divided by 2a | 199/199 |
| B_v | `abs(transp.dat:BZXR)/(100*R_v_m)` in tesla | 199/199 |
| q_0, q_lcfs, q_min | `mapdsk.cdf: qa`, with `PsiBig` for radial coordinates; retain the near-axis sampling qualification below | 199/199 |
| p_0 (requested peak) | Maximum mapped `pa`; preserve native normalization, with SI conversion `p_Pa = pa*B_v**2/mu0` | 199/199 |
| rho_0 (requested peak) | Reconstruct physical mass density from `transp.dat` with the confirmed `iden=4` processing | 199/199 |

The geometry audit intersects the LCFS with `Zmag` (the symmetry plane in
these cases). It computes `delta_upper/lower=(R_v-R_at_Z_extremum)/a`, retains
both values, and uses their mean as scalar delta. Upper/lower values agree
in every case. The derived midplane R_v agrees with both mapped `X0` and
TRANSP edge `RMJMP/100` within 1e-9 m. The LCFS arrays agree exactly with the
last mapped surface. Shaping uses the sampled contour, without fitting its
extrema. `mapdsk.cdf:B0=1` is not the dimensional vacuum field.

Use distinct names for **axis** and **peak** values. At a relative tolerance
of 1e-6, the sampled pressure maximum exceeds the first mapped value in
22 cases, and the processed density-knot maximum exceeds the axis value in
77 cases. The frequency reference still uses axial mass density. In the
audit CSV, q and pressure axis proxies are labeled `*_axis_near`: the first
mapped point is at `psi_N≈2.5e-7`, not exactly zero. q and pressure minima/
maxima are sampled-grid extrema. Density maxima are processed-knot extrema;
finding extrema of NOVA's subsequent spline remains export implementation
work. This distinction does not indicate missing input data.

For reproducibility, the audited density mass sum, before axis normalization,
is proportional to
`NH + 2*ND + 3*NT + 2*BDENS_D + 3*BDENS_T + 4*NFI + 3*NHE3 + AIMPJ*NIMP`.
The current source smooths each contributing density except `NHE3` once;
it does not smooth `AIMPJ`. It then extrapolates the mass sum to axis/edge.
`PLFLX` is interpolated from `XB` to `X`, extrapolated, and normalized to
poloidal flux. The SI audit uses proton-mass units and converts cm^-3 to m^-3;
`rho_*_kg_m3` does not include mu0. All 199 files supply every required block.
The inactive `NT`, `NFI`, `BDENS_T`, and `NHE3` blocks explicitly contain zeros
in every case. No missing species block was silently filled. All processed
density knots and mapped pressure samples are positive.

**November 17 impurity-mass clarification:** all 199 populated cases have
`AIMPJ=12` and `XZIMPJ=6` at every TRANSP radial point; all 199 `transp.dat`
hashes still match the November 15 audit. The reconstruction uses the supplied
`AIMPJ` profile, not a separately chosen impurity mass or an A-from-Z rule.
Thus the impurity contribution in this database is `12*smooth(NIMP)` in
proton-mass units before extrapolating the complete mass sum to the axis.
This follows the code's integer-mass convention (`m_imp=12*m_proton`), not
an exact atomic/isotopic mass conversion. It describes the supplied effective
impurity model, without independently establishing the plasma composition.
Both inspected sources retain an obsolete comment about impurity mass 13;
the corresponding `13*dzdat` term is commented out. The active term is
`aimpj(i)*dzdat(i)` in `Equil/mhdidin_flux.f:4276` and
`NOVAK3/tae_flux.f:7226`. `XZIMPJ` is read but is not used in this mass term.

The density audit agrees with compiled original `smooth`, `extrap1`, and
`boundtoc`/`fun4` Fortran helpers across all 199 cases: maximum differences
are 1.06e-22 kg/m^3 in density and 1.23e-15 in normalized poloidal flux.
Current NetCDF, `qpvspsi.dat`, and `mpout1` q/pressure copies again agree
exactly. The pressure conversion follows the active normalization
`p0=p0*mu0/bzero**2` in `eqsolvAnis.f`; the mapped quantity is the equilibrium
pressure flux function, not an independently substituted TRANSP pressure.

Per-case values, source-file hashes, coverage/ranges, the bounded audit
script, and the Fortran comparison are in the local ignored directory
`research_archive/surrogate_scalar_availability_20261115/`:
`shot_scalars.csv`, `input_fingerprints.json`, `summary.json`, `audit.py`,
`density_validation.json`, and `validate_density.py`. This is an availability
audit, not a released training table. Existing invalid-input and continuum/
mode association holds remain in force; complete scalar coverage does not
clear them. No production processing or scientific selections changed.

### Follow-up: stored shaping, axis definitions, and q extrapolation

The reported delta and kappa were calculated from the mapped LCFS in
`mapdsk.cdf`; they were not copied from TRANSP shaping scalars. All 199
TRANSP files also contain `TRIANG`, `TRIANGL`, and `ELONG` profiles, whose
edge values describe the original TRANSP geometry. Keep those separate from
the shaping of NOVA's mapped equilibrium. `qpvspsi.dat` additionally prints
`ellipt` and `triangl`, but its source uses global R extrema; the contour
calculation explicitly preserves the agreed midplane definition of a/R_v.
The NetCDF variable named `delta` is a mapping array, not the scalar
triangularity. q_0, q_min, and q_lcfs likewise come from mapped `qa`.

The user is reconsidering the initial peak-value request and tentatively
prefers axial p_0 and rho_0. The recommended initial schema uses axis values
for both, preserving maxima separately. Axial density is the normalization
reference; the pressure axis estimate should retain its extraction method.
The original audit CSV keeps both definitions and is unchanged.

A follow-up checked all 199 mapped q profiles, using the text copies whose
hashes still match the original audit. Linear extrapolation in psi_N from
the first two points (approximately 2.5e-7 and 2.5e-5) to zero gives:

| Correction from first mapped q to estimated q(0) | Median | Maximum |
| --- | --- | --- |
| Absolute difference in q | 3.70e-7 | 4.24e-5 |
| Relative difference, percent | 0.0000176% | 0.000880% |

The largest relative change is `nstxuG121123N22`: 3.9293604394 becomes
3.9293950155. Local three-point quadratic and four-point cubic extrapolation
change the linear estimate by at most 1.62e-6 and 2.31e-6 in q, respectively.
These are small extrapolation corrections, not an independent measurement
of the exact axis value or a global quadratic q-profile fit. Reproduction
and results are `q_axis_check.py`, `q_axis_comparison.csv`, and
`q_axis_summary.json` in the same ignored audit directory.

All 199 NetCDF files share the same variable schema and contain no mass-
density field. The inspected NOVA initialization calls `datin_new` for
`iden=4` to construct density from TRANSP, separately from reading mapped
equilibrium data. The November 16 check below additionally establishes a
saved normalized profile in `N<n>/otaep` for part of the database. Retain
the physical axis scale from the species sum separately. The existing audit saves
`rho_axis_kg_m3` and `rho_peak_knots_kg_m3`; it does not yet export the full
NOVA-interpolated density profile.

## Saved RHOPRF profiles in otaep: November 16 audit

The user identified `N<n>/otaep`, under the exact heading `RHOPRF profile`,
as a saved density-profile source, and subsequently confirmed that these
files are produced by NOVA-C growth-rate calculations. Only 89 cases have
been processed with NOVA-C so far. This corrects the earlier implication
that every normalized profile must be reconstructed from TRANSP: the mapped
NetCDF files lack density, but the run output can contain it.

A bounded scan of all inventory cases, N1 through N10, completed at
12:42 EST on November 16. The 199 populated cases have 1,990 N directories;
the remaining inventory entry, `nstxu_202806`, is empty.

| Check | Result |
| --- | --- |
| Active `gridparam: nmbrgrd` | 201 in all 1,990 N directories; commented alternatives ignored |
| Exact `otaep` files found | 761 |
| Complete `RHOPRF profile` blocks | 749, one per file, all 201 values |
| `otaep` files without a profile | 12 header-only files, each 333 bytes |
| N directories without `otaep` | 1,229 |
| Cases with at least one complete profile | 89 of 199 populated cases |
| Cases with multiple complete copies | 88; all copies agree exactly at printed precision |
| Cases with all ten complete copies | 23 |
| Case with only one complete copy | `nstxuE204118F01t035` |
| Populated cases with no complete copy | 110; listed in the coverage CSV |

All 749 profiles are finite and positive, start at 1.0, and span a combined
range 0.116 to 1.15. Their `NNSURF` headers match the active `gridparam`
dimension, and their `NTOR` headers match their directories. Maximum
within-case difference between saved profile samples is zero. An independent
parser rechecked all 749 arrays against the saved extraction.

The inspected `NOVAK3/tae_flux.f` sets `rho0=rhoprf(1)` at line 3012 and
divides `rhoprf(i)` by it at line 3084. `printprf` writes `RHOPRF profile`
with format `10E12.3` at lines 3279–3284. Thus this block supplies normalized
mass-density shape, approximately three significant digits per sample,
not physical density in kg/m^3. The physical axis scale remains a separate
quantity. Exact agreement of printed copies does not imply full-precision
identity of their internal arrays.

The audit preserves surface ordering without inventing a radial coordinate
from the element count. The inspected source evaluates density on
`rgrid**2` (`tae_flux.f:6025–6034`); eventual export must attach the matching
run grid. No interpolation to psi_N=0.5, complete 13-parameter export, or
HDF5 generation was performed. Saved-profile/current-input correspondence
and existing mode/continuum holds remain separate from this consistency check.

Evidence is in the ignored local directory
`research_archive/otaep_density_audit_20261116/`: `file_audit.csv`,
`shot_coverage.csv`, `rhoprf_profiles.npz`, `summary.json`, `validation.json`,
and the bounded reproduction script `audit.py`. The CSVs include file and
profile hashes. NPZ rows map to the per-file `profile_index` and carry shot
and n identifiers. Coverage is a dated snapshot: two additional header-only
files appeared between the preliminary inventory and the final scan.

## NOVA-C central density and Alfvén speed in outb outputs

Following the user's pointer, a second November 16 audit searched
`N1`–`N10/outb_*` within the 200 inventory cases. The scan ran from
14:23 to 14:29 EST. It found the exact diagnostic
`the alfven speed and center density are:` in **5,338 output logs covering
all 89 NOVA-C cases**. Each successful file has one two-value diagnostic.
Both values are identical across all available modes and n for each case.
Another 34 output files lack the diagnostic; 4,895 empty lock markers were
recorded separately. No file changed while being read. Coverage exactly
matches the 89 cases with saved `otaep` density profiles.

The inspected writer is `NOVAK3/tae_flux.f:7534–7540`:
`va=2.18e11*b0/sqrt(rho0)`, followed by `write(*,'(2e16.7)') va,rho0`.
Here B0 is in gauss, va is in cm/s, and rho0 is the code's mass-weighted
density expressed in proton-mass units per cm^3. Convert with
`va_m_s = 0.01*va` and `rho_kg_m3 = rho0*m_proton_kg*1e6`.
The second number is not electron number density or SI mass density.
It is the pre-normalization scale used for NOVA-C's `RHOPRF`.

The user's example occurs in `nstxuE204708F03t017`, including
`N2/outb_egn02w.1122E+02_gc062_e09:1294`:

```text
the alfven speed and center density are:
  0.2622642E+09   0.2953330E+14
```

This gives va = 2.622642e6 m/s and NOVA-C central mass density
4.9398045e-8 kg/m^3 using the same proton-mass constant as the prior audit.

**NOVA and NOVA-C central-density prescriptions differed in the November 16
sources and these outputs.** NOVA `iden=4` uses smoothed thermal hydrogen
`NH` (`Equil/mhdidin_flux.f:4274–4277`), whereas NOVA-C uses `NMINI_H`
through `dhmindat` (`NOVAK3/tae_flux.f:7029–7031`, `:7222–7226`) in that mass
term. `NMINI_H` is zero in all 89 audited cases. Replacing the processed NH
contribution by NMINI_H reproduces **all 89 logged densities to their seven
significant printed digits**. The largest residual before output rounding
is 2.69e-7 relative. The logged Alfvén speeds also match the source formula
using the saved field scale within 3.77e-7 relative.

Consequently, the logged NOVA-C density is lower than the prior NOVA
reconstruction by 0.000276%–0.451813%, with median 0.282130%. The largest
difference is `nstxuE204708A06t017`. Keep `rho_axis_novac` and the NOVA
`iden=4` scale separate; do not silently replace one with the other.
The NOVA-C thermal-H treatment should be checked with the code runner before
adopting its scale for NOVA frequency normalization. Full radial-profile
agreement across the two codes was not established by this central-value
audit, so `otaep` should not yet replace the NOVA profile without that check.
The subsequent user decision is to use reconstructed NOVA density for the
surrogate; the NOVA-C hydrogen question does not block that choice.

The local ignored directory `research_archive/outb_density_audit_20261116/`
contains `file_audit.csv`, all extracted pairs in `diagnostic_values.csv`,
the per-case `shot_density.csv` with raw/SI NOVA-C values and both density
recipe comparisons, `summary.json`, `validation.json`, and `audit.py`.
All 89 reference logs were independently reparsed and their hashes checked;
the previous NOVA density reconstruction is unchanged. Source hashes are
retained, without claiming identity of every historical executable. No NOVA
inputs, selections, normalization conventions, or HDF5 exports were changed.

On November 17, the NOVA sources still matched their audited hashes, and
all 89 comparison TRANSP inputs were unchanged. The current NOVA-C source
hash differed, but its adjacent `tae_flux.f~` exactly matched the November 16
audit hash. Their sole difference is a comment correction at line 7060:
`XZIMPJ` is zonal-average **Z**, rather than **A**. The executable statements,
including the hydrogen and impurity mass terms, are unchanged.

## Ion-mass consistency check: November 19

The current Flux TRANSP workflow uses consistent ion/proton mass factors in
NOVA initialization, the continuum calculation, and NOVA-C. This is the
code's integer-mass convention, not an exact isotope-mass conversion.

| Species | Mass / proton mass | Active representation |
| --- | --- | --- |
| Thermal H | 1 | `NH`; NOVA-C `rmcp(4)=1` |
| Thermal and beam D | 2 | `ND`, `BDENS_D`; `rmcp(2)=rmhp(1)=2` |
| Thermal and beam T | 3 | `NT`, `BDENS_T`; `rmcp(3)=rmhp(2)=3` |
| Helium-3 | 3 | `NHE3` in the mass-density sum |
| Fusion alphas | 4 | `NFI`; default `rmhp(3)=4` |
| Impurity | supplied `AIMPJ`, here 12 | `AIMPJ*NIMP`; kinetic `rmcp(5)=12`, `zc(5)=6` |

The electron mass is separately `rmcp(1)=1/1836`; electrons are not included
in the ion mass-density sum. Mass and charge are separate inputs: impurity
charge 6 is not used as the mass multiplier.

The bounded check reread all **199 populated TRANSP inputs**. Their hashes
still match the November 15 audit. Every case has `AIMPJ=12`, `XZIMPJ=6`,
and zero `NMINI_H`, `NT`, `BDENS_T`, `NFI`, and `NHE3` profiles. Thermal H,
thermal D, and beam D are present in every case. Among 1,990 named
`N1`–`N10/NOVAK_param` paths, **964 files exist** and all specify
`rmhp=(2,3,4)`, `zh=(1,1,2)`, `ihsps=1`, `itransp=1`, and `_dtype=l`.
The remaining 1,026 paths have no parameter file; they are not inferred runs.
Thus every available parameter file selects deuterium beam ions with mass
2 and charge 1.

All **5,338 historical output logs** from the November 16 diagnostic cohort
were also checked directly. Their printed namelists agree exactly on
`rmcp=(1/1836,2,3,1,12)`, `zc=(-1,1,1,1,6)`, `rmhp=(2,3,4)`, and
`zh=(1,1,2)` within floating-point print precision. No log changed during
its read. This check fingerprints each log's first 65,536 bytes containing
the namelist; it does not claim a full-file hash for every log.

**The current NOVA-C source already contains the thermal-H correction.**
Compared with the preserved November 17 source, `NOVAK3/tae_flux.f` now
adds `+1.*dhdat(i)` to `rhodat` and retains `+1.*dhmindat(i)`
(lines 7222–7224). Its modification time is November 19, 16:10 EST.
The only other change is the thermal-H read comment. NOVA initialization
(`Equil/mhdidin_flux.f:4274–4276`) and the continuum source
(`Equil/contx_flux.f:2243–2245`) are unchanged from the November 17 snapshots.
Since `NMINI_H=0` in all 199 cases, the current NOVA-C mass sum now agrees
with NOVA's mass sum throughout the input profile, with the same contributing
species smoothing. This does not establish identity of the subsequent
interpolated profiles or of installed executables.

The historical density discrepancy was an omitted species contribution,
not an incorrect D/H or impurity mass ratio. All **89 reference density
logs** were reread in full and their hashes still match November 16; all
89 reproduce the old mass sum at printed precision, and none reproduce
the corrected sum at that precision. The November 16 logs have
central density up to 0.451813% below NOVA's value, implying up to 0.226675%
higher Alfvén speed at fixed B. This is a normalization-scale comparison,
not a measured change in kinetic growth rates or a bound on normalized
continuum changes. The existing surrogate density already uses the NOVA
prescription and needs no change from this correction.

Additional limitations matter when extending the workflow. The kinetic impurity
mass/charge remain fixed at 12/6 while the density sum reads `AIMPJ`; they
agree for this database but are not automatically synchronized for another
impurity. NOVA-C also retains a separate `NMINI_H` contribution absent from
the NOVA sum; the agreement established here relies on that profile being
zero. A nonzero minority-H case needs its species accounting checked.
Also, `orbit2dlib.f:3490–3493` retains an older `datin` routine
with an active impurity factor 13 and no thermal-H term. The inspected
TRANSP path calls `datin_pest` instead (`tae_flux.f:6015`); its `call datin`
is commented out. The obsolete mass-13 comments in the active density
routines should not be mistaken for executed formulas.

Source snapshots, the November 17/19 diff, input hashes, per-case species
checks, parameter-file checks, and historical-log checks are retained in
`research_archive/ion_mass_consistency_20261119/`. This was a read-only audit
of solver sources, inputs, and results. No solver was rebuilt or rerun, and
no production selection or HDF5 artifact was changed. Next: verify a binary
built from the corrected source against the reconstructed density before
treating any new NOVA-C outputs as corrected.

## Continuum changes traced to poloidal-harmonic limits (November 17)

The NOVA-C workflow invokes a separate `gocont_tr` continuum calculation
before running `gotae3`. Its `Equil/contx_flux.f` density sum at lines
2243–2245 includes thermal `NH`, as does NOVA initialization. It does not
use the `NMINI_H` mass sum in `NOVAK3/tae_flux.f`. Thus the NOVA-C central
density difference above does not explain these regenerated datcon changes.
The normalized squared continuum eigenvalues contain the factor
`r2*q1*q1*rho0/rhoms` (`contx_flux.f:1379–1382`); a uniform density-scale
change cancels in this normalized ratio. Physical frequencies at fixed B and
geometry scale as rho^-1/2, but axial density alone cannot bound a change
in the full normalized density shape.

The retained original `runcontrange.scr`, root `outcon`, and example older
`plotcontN10` establish the historical range m=-3..108. New `N*/outcon`
files use m_min=0 and smaller per-n upper limits. In E204645A16t015,
`runStabNovakBeams.py:73–74` sets `lsub=min(int(n*qa),100)` with script
parameter `qa=12`, whereas the mapped q_lcfs is about 16.349. This is an
input-script parameter, not a read of the mapped edge q. The N10 launcher
passes `lmax=100` and calls `gocont_tr` at lines 65 and 72 respectively.
In the solver, `lmin`/`lmax` set the actual matrix basis
(`contx_flux.f:1280–1301`), not just a plotting range. The datcon writer
selects positive eigenvalues from that truncated spectrum (lines 117–137).

Controlled runs in independent directories cover all 18 scopes containing
the 32 modes with >1% boundary changes in the October 29/November 16 audit.
For each scope, a common retained executable and identical copies of
`mapdsk`, `mpout1`, and `transp.dat` were used with old/new harmonic settings
and the corresponding recorded gamma decimal precision. Restoring the old
settings recovers both October boundaries for all 61 selected modes in
these scopes within 4.79e-11 relative. New settings reproduce all 18 frozen
November datcon files byte-for-byte. Exported q profiles are identical
between the two input ranges. Separate runs changing one limit with gamma
held fixed establish the harmonic-range effects directly.

For E204645A16t015/N10 at r=0.8417083979 (nq about 99.65), the upper boundary
is 4.848143081 with m=0..100 and 3.298538479 with m=0..108; the historical
value is 3.298538479 (46.9785% smaller-denominator relative change).
These centroid values use the existing continuum preprocessing. The new
raw upper boundary interpolated at the same radius is 5.516104181, a
67.2287% change; the existing monotonic-tail repair holds the new terminal
rise at its r=0.84 value, 4.848143081. The repair reduces the apparent
discrepancy and does not create it. The comparison plot distinguishes the
raw and repaired new upper curves.
Changing m_min to -3 at m_max=108 has negligible effect there. At low n,
negative m can matter: for E203981A01t022/N1 at r=0.5370464, the upper
boundary is 3.700051574 with m=0..10, 3.630915986 with m=-3..10,
3.693851555 with m=0..108, and 3.624761457 with m=-3..108.

Restoring historical settings is not a convergence certification: in the
N10 example, m=0..166 gives 3.257479829, another 1.24% below the m_max=108
result at the selected location. No range here is asserted to be converged
over the entire radius, and no mode reclassification or production/HDF5
replacement was performed. The 1,360 previously mismatched modes and other
scopes were not included in this controlled experiment.

Reproduction scripts, copied inputs/executables, input/output SHA-256
receipts, source/script snapshots, full-profile byte checks and centroid
comparisons are in `research_archive/continuum_cause_20261117/`.
`reproduction_summary.json` and `reproduction_comparison.csv` retain the
numerical results; `harmonic_range_comparison.png`/`.pdf` illustrate the
large-change example. These are diagnostic reconstructions, not substitutes
for missing historical raw-file hashes.

## Saved mode harmonics versus internal surface-field harmonics (November 18)

**The matching rule: calculate the continuum with NOVA's internal harmonic
range, including the two extra harmonics on each side in these legacy runs.**
For a saved range `m_min_saved..m_max_saved`, the continuum task therefore
uses `lmin=m_min_saved-2`, `lmax=m_max_saved+2`: `nhar_saved+4` harmonics.
Check the matching eigenmode run's internal limits rather than inferring
physical m values from the number of saved components. The two-per-side
padding is verified for the configurations below; check other builds and
superNOVA independently. Matching the existing basis does not establish
convergence.

The four held training N1 cases expose a separate source of continuum/mode
inconsistency. NOVA has an `imode` range for the saved mode components and
a wider `jmode` range for internal surface-field matrices. In these cases,
`jmin=minm-2` and `jmax=maxm+2`: four additional harmonics. `mhdidin_flux.f`
builds the surface matrices using `kmin=jmin`, `ktot=jt`; `mhdidst.f` and
`mhdidst_esp.f` invert that block before selecting the `imode` components.
Consequently, the harmonic count in an `egn*` file alone is insufficient
to specify a continuum calculation consistent with that eigenmode run.

| Case | Saved physical m range (`nhar`) | Internal / continuum m range (count) |
| --- | --- | --- |
| `nstx_135388`, original 250-mode cohort | 0..20 (21) | -2..22 (25) |
| `nstxuG121123B12` | 0..17 (18) | -2..19 (22) |
| `nstxuG142301W29` | 0..13 (14) | -2..15 (18) |
| `nstxuG142301Y93` | 0..15 (16) | -2..17 (20) |

B12/W29/Y93 matching logs record these internal limits directly as
`mmin,mtot`. For 135388, the available harmonic header and
modotv support the inference, and unchanged modes retain saved September
singularity references; its matching full log is no longer available.
The separate newer 23-harmonic 135388 mode is not covered by that inference.
Replacing `datcon1` does not change the saved eigenmode components or the
`nhar` displayed by the viewer. To inspect the published profiles, select
DiTw with `--base_dir $NOVA_DITW_ROOT`; the separate `data_mixed` copy still
contains the older continua.

Controlled continuum-only runs first reproduce the current 135388 file at
0..12, then test saved-component and internal ranges. Merely changing
135388 to 0..20 leaves 38/41 historical training interior crossing offsets
above two grid intervals. Using the internal ranges reduces flags to 0/42
(135388), 2/6 (B12), 0/14 (W29), 1/21 (Y93). Across all matched modes,
105/108 interior crossings are within two intervals. Three crossings lack
nearby logged counterparts and remain unresolved; nearest-distance tests
do not identify a specific branch. These raw-profile results do not rely
on tail repairs and do not establish convergence.

Frozen source evidence, settings, receipts, four diagnostic `datcon1`
files, comparison CSVs and PNG/PDF plots are under
`research_archive/training_n1_matched_harmonics_20261118/`. Identical input
and executable hashes, fixed non-harmonic settings, and byte-identical
exported q profiles isolate the range effect. No DiTw/HDF5 replacement
or input-hold release was performed during the diagnostic experiment.
The user subsequently authorized publication, completed November 18 with
UTC stamp `20261119T000250Z`. These four live N1/datcon1 files now contain
the internal-basis results. Originals are preserved in project and adjacent
DiTw backups, with a live-root `.continuum_matching_20261119T000250Z.json`
provenance manifest. The archived reconstruction receipts, rather than the
unchanged historical N1/outcon files, identify the new continuum settings.
HDF5 products and input holds remain unchanged.

The subsequent [November 19 convergence checks](project_state.md#november-19-n1-harmonic-convergence-checks)
varied both harmonic cutoffs with fixed inputs and reproduced all four matched
baselines. Wider ranges change the upper boundary by up to 5.63%, 9.92%,
28.97% and 34.65% for 135388/B12/W29/Y93 over native r=0.01..0.99.
The final negative-cutoff increments change either boundary by at most
0.0105%, 0.0179%, 0.3485% and 0.8032%, respectively. W29/Y93 retain edge
sensitivity; these increments are not error bounds. Both cutoffs and the
compiled harmonic capacity must be checked. The diagnostic profiles and
full comparisons are in `research_archive/training_n1_convergence_20261119/`;
the published mode-matched profiles and HDF5 remain unchanged.

## Preferred sources and checks

The subsequent November 17 MPI reconstruction expanded the historical-setting
calculation to all 638 affected case/n scopes in 80 cases. All executions
succeeded. Both October centroid boundaries agree for all 3,066 modes with
available original October measurements (maximum relative residual
4.784e-11). All 1,372 original crossing witnesses for the earlier-overwritten
group agree within 5.851e-12, spanning 645 modes in 135 of that group's 179
scopes. These local checks do not establish full-profile identity. None of
the 459 available original raw-file hashes matches the reconstructed bytes;
use new input hashes and explicit reconstruction provenance. Results and
receipts are under `research_archive/continuum_reconstruction_20261117/`;
see [project state](project_state.md#november-17-mpi-reconstruction).
The reconstruction phase left the delivered HDF5, source DiTw files, and
selection decisions unchanged. In the subsequent authorized November 17
publication, all 638 reconstructed numbered continua were installed in DiTw
after two independent backup sets were verified. The live manifest is
`DiTw/.continuum_restoration_20261117T230824Z.json`; project records are under
`research_archive/continuum_publication_20261117/`. It records the new hashes
and settings; preexisting NOVA-C `outcon` logs describe the earlier runs.
Mode files, selections, and the delivered HDF5 remain unchanged.

The subsequent full-profile audit compares all 638 restored/backed-up pairs
in frequency, using both raw sentinel-masked boundaries and the unchanged
shared loader. The largest raw upper-boundary difference is +911.075% at
r=0.99 for E204669M03t030/N9; changes concentrate near the outer harmonic
cutoff. Lower-boundary relative errors need care near zero frequency, and
the historical high-n profiles are not convergence references. Methods,
radial coverage, representative cases, and raw/processed distinctions are
recorded in [project state](project_state.md#november-17-full-profile-continuum-comparison)
and `research_archive/continuum_full_profile_comparison_20261117/`.

| Quantity | Source | Interpretation / checks |
| --- | --- | --- |
| Physical poloidal harmonics | Matching mode/run metadata, including `N<n>/out_go` and harmonic-range settings | Recover the actual exported m range and ordering; preserve `m_values` separately from array indices. Account for any solver-only extra harmonics when interpreting settings. |
| Mapped q profile | `mapdsk.cdf: qa`; `qpvspsi.dat`, column 2 | Confirmed NOVA input after q-solver/mapping; agrees with mapped binary quantities. |
| Mapped pressure | `mapdsk.cdf: pa`; `qpvspsi.dat`, column 3 | Confirmed NOVA pressure after equilibrium solution and mapping, including mapper edge treatment. |
| Radial coordinate | `mapdsk.cdf: PsiBig`; `qpvspsi.dat`, column 1 | `qpvspsi[:,0] = PsiBig / PsiBig[-1]`, normalized poloidal flux. Use its square root for the mapped radial coordinate. |
| Field function | `mapdsk.cdf: ga`, `gpa` | ga is confirmed as the mapped field function used by NOVA; retain the code convention for g and its derivative and verify any conversion to dimensional F. |
| Flux surfaces | `mapdsk.cdf: xa`, `za` | In the examined files, Python reads `(radius, poloidal_angle)`. The poloidal endpoint is repeated. Retain the mapped angle convention. |
| LCFS | `mapdsk.cdf: xinf`, `zinf` | Full boundary coordinates; derive shaping consistently from contours. |
| Geometric center and axis | `mapdsk.cdf: X0`, `Xmag`, `Zmag`; LCFS contours `xinf`, `zinf` | Validate `X0` against the midpoint of the inboard/outboard LCFS intersections with the equilibrium midplane. Global `Xmin`, `Xmax` are not the general definition of R_v. Preserve the magnetic-axis location separately. |
| Scalar equilibrium summary | Final named blocks in `outequ` | Includes `q-zero`, `q-edge`, `xmagax`, `x-maj`, `arad`, `a-ratio`, `dness`, and several distinctly defined betas. Match labels, not fixed line numbers. |
| Dimensional vacuum-field scale | `transp.dat: BZXR` | Header gives tesla-cm. The vacuum toroidal-field magnitude at midplane `R_v` in meters is `abs(BZXR)/(100*R_v)`; retain the signed field separately. |
| Saved NOVA-C normalized mass-density profile | `N<n>/otaep`, `RHOPRF profile` block | 201 samples where present; axis-normalized, printed to approximately three significant digits. November 16: 89 cases covered; all available copies within each case agree. Attach the matching radial grid. Cross-code agreement with NOVA remains to be checked given the NH/NMINI_H difference above. |
| Logged NOVA-C axial density and Alfvén speed | `N<n>/outb_*`, two numbers after `the alfven speed and center density are:` | Speed in cm/s; density in proton-mass units per cm^3. 89 cases covered with exact within-case agreement at printed precision. Keep this scale separate from the NOVA iden=4 scale. |
| Density and composition | `transp.dat`: ion-species and beam densities, `NIMP`, `AIMPJ` | Use the confirmed species mass sum and reproduce the `iden=4` smoothing, endpoint treatment, axis normalization, and radial mapping. NE alone does not specify this mass density. |
| Original profile coordinates | `transp.dat: X`, `XB`, `PLFLX`, `TRFLX` | Center and boundary grids differ. Headers distinguish them. Do not identify TRANSP X with NOVA sqrt(poloidal flux). |
| Original shaping / asymmetry | TRANSP `RMC*`, `RMS*`, `YMC*`, `YMS*`, `ELONG`, `TRIANG`, `TRIANGL` | Preserve for future superNOVA work. Current NOVA labels belong to the mapped symmetric geometry, not automatically to the original asymmetric reconstruction. |
| TAE continuum | `N<n>/datcon<n>` | Use the existing shared loader and its validity handling. Frequencies are squared and must remain consistent with native mode frequencies. |
| Run settings | `inequa_01`, `torot.nml`, `N<n>/commands`, `modinv`, `out_go`, `gridparam`, `clichlmax` | Preserve settings and actual run evidence, especially density choice, q rescaling, rotation, grids, and symmetry. A current input card alone does not establish historical execution. |

## What the three q sections mean

The inspected `eqsolvAnis.f` source distinguishes these explicitly:

1. **`q profile from g`:** q reconstructed from the calculated field function
   and equilibrium geometry (`eqsolvAnis.f:5002`). It is a consistency
   diagnostic and can differ from the prescribed profile.
2. **`exact q profile`:** for `iqfunc=2`, the code replaces the diagnostic
   `qfac` with `qfa(psival)` (`eqsolvAnis.f:5023`). Here "exact" denotes the
   prescribed q function, not proof of an exact numerical equilibrium.
3. **`x q x q psi q`:** the later table presents q along the two midplane
   branches and against flux. These columns are coordinate presentations,
   not three independent physical q inputs.

The equilibrium writer uses `qfa` for the exported q array when `iqfunc=2`
(`eqsolvAnis.f:5453`). The mapper interpolates it to `q_cdf`, then writes the
same array to `mpout1`, `qpvspsi.dat`, and `mapdsk.cdf:qa`
(`dmapb2_flx.f:513`, `:776`, `:1282`). NOVA initialization explicitly reads
the edge q from `mpout1` and uses `oma2=1/(q1*q1*r*r)`
(`mhdidin_flux.f:78`). Any per-run `qscale` change must also be checked.

Example: `nstxuE202806A02t045/outequ` has the two labeled profiles at lines
721 and 741, the coordinate table at 1110, and the final q summary at 1256.
The reconstructed edge q is printed as `7.487`, whereas the prescribed edge
q is `7.536` at that block's precision. The mapped array gives
`qa[0]=0.9302274187038345`, `qa[-1]=7.535695780662116`.
The front input cards' `q0=1` and `qlim=3` are not these final q values.

## Files that need care

- **`qprofile.txt`:** the continuum writer uses
  `sqrt(normalized toroidal flux)` for column 1 (`contx_flux.f:170`). The first
  two rows are zero-filled in the four initial examples, and the third radius
  is also zero despite a finite q. It is not the preferred canonical q input.
- **`bprofile.txt`:** uses the same toroidal coordinate, with axis NaNs in the
  examples. The inspected writer also contains a fixed `1.729/1.6434` factor
  (`contx_flux.f:177`). Do not adopt its field amplitudes without tracing the
  executable version and this correction.
- **`novashapes.txt`:** the inspected continuum writer exports ten surfaces
  at `sqrt(normalized toroidal flux)=0.1,...,1` (`contx_flux.f:145`). Useful
  for plots, but less complete than the mapped contour arrays.
- **`XZsurfs.txt`:** selected surfaces are hardcoded in the inspected mapper,
  including a repeated index (`dmapb2_flx.f:636`). The labels do not establish
  general rational-surface identities across shots.
- **NetCDF scalar names:** the writer sets `B0=1` explicitly. Its `BetaStar`
  assignment is marked `is not correct` in the source. `Beta=0.78` in the
  initial four examples also equals an equilibrium input shape parameter.
  Treat `Beta`, `BetaStar`, `BetaN`, `Ip`, `li`, and `PPF` as unverified;
  named scalar fields are not automatically physical diagnostics.
- **Near-axis grid:** the examined mapper profiles start at
  `sqrt(psi_N)≈0.0005`, followed by `0.005`, rather than an exact zero.
  Preserve the actual radial coordinates when preparing surrogate inputs.

## Confirmed density treatment and remaining implementation work

The current `iden=4` paths in `mhdidin_flux.f:4091` and
`contx_flux.f:2094` read TRANSP species, smooth selected density profiles,
move the poloidal-flux data from boundary to center locations, extrapolate to
axis/edge, and spline the normalized mass-density profile versus normalized
poloidal flux. The active mass sum includes hydrogen, deuterium, tritium,
beam ions, helium, and impurity mass. The electron-only density prescription
is commented out. On November 13 the user confirmed this species-based
treatment for all database cases. Saved NOVA-C normalized profiles are now
verified in `otaep` for 89 cases, with NOVA-C axial scales available in
`outb_*`. The thermal-H difference documented above must be resolved before
substituting those NOVA-C profiles/scales for NOVA inputs. Reconstruction
remains available for missing profiles and greater numerical precision;
it follows the confirmed NOVA species-based treatment. Selecting between
electron-only and species-based density is no longer an open scientific
question. Both normalized shape and axial mass-density scale are needed.

Rotation also needs a per-case check. Five mapped cases have `imach=1` and
nonzero `pM`: `nstx_135388`, `nstxuG142301E34`, `nstxuG142301E77`,
`nstxuG142301L89`, and `nstxu_204202`. Their current `torot.nml` files agree:
`imach=1`, `mach20=0.01`, and `mach21=0`. Preserve these settings and the
mapped rotation data when deciding the surrogate's physics scope.

The November 13 confirmation establishes the density and mapped-profile
conventions across the database. The November 10 audit did not fingerprint
every historical executable source revision; preserve input hashes when
building the training set. Existing mode/continuum fingerprint holds remain
in force and are separate from these resolved source-convention questions.

## Discovery evidence and next step

The initial examples were `nstx_120113`, `nstx_135388`,
`nstxuE202806A02t045`, and `nstxuG142301W29`. A bounded audit then checked the
200 cases listed in `audits/main_dataset_shots/shot_status.csv`:

- All 199 populated cases have nonempty files for all eleven names checked:
  `outequ`, `qpvspsi.dat`, `mapdsk.cdf`, `mapdsk`, `mpout1`, `transp.dat`,
  `qprofile.txt`, `novashapes.txt`, `bprofile.txt`, `torot.nml`, `inequa_01`.
  The only missing case is the known empty `nstxu_202806` entry.
- In all 199, `qpvspsi.dat` agrees exactly with NetCDF q, pressure, and
  normalized poloidal flux. NetCDF q, pressure, and g agree exactly with the
  corresponding binary `mpout1` profiles. This verifies the mapped profile
  copies, not every metric array or historical mode association.
- All mapped profiles have 201 radial samples; contour arrays have shape
  `(201,257)`. All `qprofile.txt` files have their first two rows zero-filled.
- `nstxuG142301E55/outequ` stops at line 152, partway through wall coordinates.
  It has no final q sections, although its mapped profile files agree.
- The 198 complete printed q comparisons have different output precision:
  do not interpret rounding differences between the packed profile and final
  coordinate table as a physical change in q.
- All 199 NetCDF files have `B0=1` and `Beta=0.78`; all TRANSP files have the
  checked asymmetric moment block names. These names alone do not establish
  nonzero asymmetry or validate the underlying reconstruction.

Some logs contain an `unconverged` diagnostic. The audit records these strings
without assigning new scientific exclusions; residuals and executable history
need separate interpretation.

Detailed local evidence, the bounded audit script, and source/example-file
fingerprints are under `research_archive/surrogate_input_discovery_20261110/`
(ignored research artifacts). The audit records availability separately from
scientific eligibility; invalid or held cases are not cleared by this work.
Its historical `Rv_midpoint_absdiff` diagnostic compared `X0` with global
`Xmin`/`Xmax`. That comparison is not a general validation of the midplane
definition clarified on November 13; future extraction must use midplane
intersections. The original discovery artifacts retain their dated scope.

Next: implement a reusable loader for the confirmed mapped profiles and
species-based density, recover the physical m labels, validate the midplane
geometry reference, and retain per-run settings and input fingerprints.
