# TAE-like mode labeling policy

Use this policy to reason from physical morphology and continuum geometry. It
is deliberately qualitative: scalar measurements support inspection but do
not replace it.

## Evidence to inspect

For every mode inspect:

- all relevant signed poloidal harmonics `xi_m(r)`;
- the radial energy envelope `W(r) = sum_m |xi_m(r)|^2`;
- radial width, smoothness, coherence, and peak location;
- near-axis behavior and radial-boundary artifacts;
- the absolute-frequency lower and upper TAE continuum boundaries;
- true sign-change crossings of the mode frequency with either boundary;
- pointwise and integrated `W(r)` near each crossing;
- alignment with a local upper-boundary minimum or lower-boundary maximum;
- whether the mode intersects continuum elsewhere at appreciable amplitude.

Do not decide from a single scalar or one plotted harmonic.

## GOOD evidence

A GOOD mode normally has a coherent, resolved radial envelope shared by
multiple harmonics. Its structure is smooth enough to be physical and broad
enough to be resolved on the available radial grid. Its main amplitude lies
in the TAE gap or at a physically plausible gap edge.

A narrow mode localized where the upper continuum has a local minimum or the
lower continuum has a local maximum can be physical. A sign change of the
mode structure at that extremum is not by itself numerical junk. Accept such
an extremum-localized mode when its harmonics are coherent and any other
continuum crossings occur only where mode energy is negligible.

Do not reject a mode merely because it is core localized, low-n, narrow
relative to older non-G modes, or has a remote continuum crossing in a tiny
tail. G-shot continuum geometry can suppress global mid-radius modes and
leave a physically distinct population of narrow inner-extremum modes.

## BAD evidence

Strong BAD evidence includes:

- one- or few-grid-point spikes or an implausibly thin radial structure;
- a significant spike or concentration at `r=0` consistent with a boundary
  problem;
- jagged or mutually incoherent dominant harmonics;
- a continuum crossing through the main mode envelope or another location
  carrying appreciable pointwise or integrated mode energy;
- a mode that looks extremum localized at one radius but crosses continuum
  elsewhere at significant amplitude;
- unresolved, corrupt, or clearly numerical mode data.

Do not use a simple presence-of-crossing rule. Crossing severity depends on
where the mode energy is located.

## Touching versus crossing

Distinguish tangency or closest approach at a continuum extremum from a true
crossing. A true crossing requires a sign change of
`omega^2 - omega_boundary(r)^2` across the boundary. Touching an extremum can
support the physical localized mode under review.

The legacy `r_star` is a closest-approach location and can represent touching;
it is not proof of crossing. `W_star` measures energy near that location and
is also not proof of crossing. A sign-change crossing diagnostic such as
`continuum_crossing_records` is appropriate for locating actual crossings.

Pointwise crossing energy alone can overstate a very small tail. When a case
is close, also inspect the fraction of integrated `W(r)` near the crossing
and the mode energy beyond an outer crossing.

## Continuum extrema

For inner-extremum modes, examine jointly:

- radial separation between the `W(r)` peak and the candidate extremum;
- frequency clearance on the gap side of that extremum;
- fraction of integrated mode energy around the extremum;
- all other crossings and their local energy;
- structure width and axis behavior.

Proximity to an extremum is supporting geometry, not an automatic GOOD label.
Ultra-narrow structures and axis spikes remain BAD even when centered on an
extremum.

## Frequency presentation

Plot continuum and mode frequency in absolute units for within-shot review.
For a cross-shot normalized comparison, divide the entire panel by one scalar
mean or median gap-center frequency over a stated radial interval. Never
divide pointwise by the radius-dependent center
`c(r) = 0.5 * (u(r) + l(r))`, because that hides radial gap variation.

## Labels

- `good`: physical-looking TAE suitable for training and downstream NOVA-C
  consideration.
- `bad`: numerical/unphysical structure or physically disqualifying
  continuum interaction.
- `skip`: honestly unresolved, contaminated by prior-label exposure, or
  limited by data quality. Preserve it for adjudication and exclude it from
  model training.

Confidence describes confidence in the morphology decision, not agreement
with another reviewer. Use a concise reason that identifies the decisive
structure and continuum evidence.
