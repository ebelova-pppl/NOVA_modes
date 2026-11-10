# NOVA surrogate: equilibrium input sources

Discovery date: November 10, 2026. Scope: frequencies and mode structures for
the accepted TAE database. Paths below are relative to `$NOVA_DITW_ROOT/SHOT`.
This is a source map, not an implemented training-data loader or a certification
that current equilibrium files reproduce every historical eigenmode.

## Conventions established with the user

- For this dataset, physical poloidal `m` equals stored harmonic index `m`.
  This is a dataset convention; the generic loader does not infer it.
- The native frequency is normalized by
  `Omega_N = V_A_ref / (q_lcfs * R_v)`, with
  `V_A_ref = B(R_v) / sqrt(mu0 * m_i * n_e(axis))` and
  `R_v = (R_min_LCFS + R_max_LCFS) / 2`.
  The reference field is evaluated at the geometric center, and the reference
  electron density at the magnetic axis. The scale varies between shots.
- Old NOVA calculations assume up-down symmetry. The future superNOVA input
  representation should retain asymmetric geometry as well.
- The density profile actually used by the historical executables remains an
  open check. The user confirmed that this needs investigation; the frequency
  reference above does not establish the radial density-profile prescription.

## Preferred sources and checks

| Quantity | Source | Interpretation / checks |
| --- | --- | --- |
| Mapped q profile | `mapdsk.cdf: qa`; `qpvspsi.dat`, column 2 | Compare against binary `mpout1`, the input used by NOVA initialization. |
| Mapped pressure | `mapdsk.cdf: pa`; `qpvspsi.dat`, column 3 | NOVA pressure after equilibrium solution and mapping, including mapper edge treatment. Do not substitute raw TRANSP pressure without checking. |
| Radial coordinate | `mapdsk.cdf: PsiBig`; `qpvspsi.dat`, column 1 | `qpvspsi[:,0] = PsiBig / PsiBig[-1]`, normalized poloidal flux. Use its square root for the mapped radial coordinate. |
| Field function | `mapdsk.cdf: ga`, `gpa` | Mapped g and its derivative; retain the code convention and verify any conversion to dimensional F. |
| Flux surfaces | `mapdsk.cdf: xa`, `za` | In the examined files, Python reads `(radius, poloidal_angle)`. The poloidal endpoint is repeated. Retain the mapped angle convention. |
| LCFS | `mapdsk.cdf: xinf`, `zinf` | Full boundary coordinates; derive shaping consistently from contours. |
| Geometric center and axis | `mapdsk.cdf: X0`, `Xmag`, `Zmag`, `Xmin`, `Xmax` | Cross-check `X0` against `(Xmin+Xmax)/2`; `Xmag` is a different reference location. |
| Scalar equilibrium summary | Final named blocks in `outequ` | Includes `q-zero`, `q-edge`, `xmagax`, `x-maj`, `arad`, `a-ratio`, `dness`, and several distinctly defined betas. Match labels, not fixed line numbers. |
| Dimensional vacuum-field scale | `transp.dat: BZXR` | Header gives tesla-cm. Candidate vacuum field at `R_v` in meters is `abs(BZXR)/(100*R_v)`; preserve the sign separately and verify the historical conversion convention. |
| Density and composition | `transp.dat: NE`, `NH`, `ND`, `NT`, `NIMP`, beam/species densities, `AIMPJ` | Source data for the density reader; needs the original smoothing, endpoint treatment, and radial mapping. |
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

## Density and remaining provenance work

The current `iden=4` paths in `mhdidin_flux.f:4091` and
`contx_flux.f:2094` read TRANSP species, smooth selected density profiles,
move the poloidal-flux data from boundary to center locations, extrapolate to
axis/edge, and spline the normalized mass-density profile versus normalized
poloidal flux. The active mass sum includes hydrogen, deuterium, tritium,
beam ions, helium, and impurity mass. The electron-only density prescription
is commented out. The historical executable provenance must determine which
recipe applies; this audit does not replace the user's frequency reference
with an inferred one.

Rotation also needs a per-case check. Five mapped cases have `imach=1` and
nonzero `pM`: `nstx_135388`, `nstxuG142301E34`, `nstxuG142301E77`,
`nstxuG142301L89`, and `nstxu_204202`. Their current `torot.nml` files agree:
`imach=1`, `mach20=0.01`, and `mach21=0`. Preserve these settings and the
mapped rotation data when deciding the surrogate's physics scope.

The current source tree explains the formats, but it is not proven to be the
exact source revision compiled into every stored executable. Equality between
present-day mapped files also does not establish correspondence with existing
`egn*` files. Existing mode/continuum fingerprint holds remain in force.

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

Next: settle the historical density recipe and per-n overrides, confirm the
remaining normalization conventions, and link each mode to a fingerprinted
equilibrium before building the reusable surrogate input loader.
