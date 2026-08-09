# TAE-like mode labeling policy

Use this policy to reason from physical morphology and continuum geometry. It
is deliberately qualitative: scalar measurements support inspection but do
not replace it.

## Evidence to inspect

For every mode inspect:

- all relevant signed poloidal harmonics `xi_m(r)`;
- the radial energy envelope `W(r) = sum_m |xi_m(r)|^2`;
- radial width, smoothness, coherence, and peak location;
- which morphology family best describes the mode, while allowing mixed
  cases that combine multiple families;
- near-axis behavior and radial-boundary artifacts, including both
  pointwise normalized amplitude and integrated energy near `r=0`;
- the absolute-frequency lower and upper TAE continuum boundaries;
- true sign-change crossings of the mode frequency with either boundary;
- pointwise and integrated `W(r)` near each crossing;
- whether each true crossing lies inside the connected mode body or only in
  a detached/negligible tail;
- whether a low-amplitude tail near a crossing is smooth and detached, or
  visibly distorted in a way consistent with continuum interaction;
- alignment with a local upper-boundary minimum or lower-boundary maximum;
- whether the mode intersects continuum elsewhere at appreciable amplitude.

Do not decide from a single scalar or one plotted harmonic.

## GOOD evidence

A GOOD mode normally has a coherent, resolved radial envelope shared by
multiple harmonics. Its structure is smooth enough to be physical and broad
enough to be resolved on the available radial grid. Its main amplitude lies
in the TAE gap or at a physically plausible gap edge.

A narrow mode localized where the upper continuum has a local minimum or the
lower continuum has a local maximum can be physical. In this family, the mode
frequency may nearly touch the continuum extremum without a true crossing
through the mode body, and the radial envelope can be much narrower than a
global TAE. A sign change of the mode structure at that extremum is not by
itself numerical junk. Accept such an extremum-localized mode when its
harmonics are coherent and any other continuum crossings occur only where
mode energy is negligible.

Do not reject a mode merely because it is core localized, low-n, narrow
relative to older non-G modes, or has a remote continuum crossing in a tiny
tail. G-shot continuum geometry can suppress global mid-radius modes and
leave a physically distinct population of narrow inner-extremum modes.

## TAE morphology families

Use these families as pattern language for review, not as mutually exclusive
classes or hard numerical cuts. A mode can mix features from several families.
The expected width, radial location, and harmonic content depend on the
family, so do not apply one morphology standard to all modes.

1. Wide/global modes: broad modes that often peak below about `r ~ 0.5`,
   and are usually dominated by two neighboring poloidal harmonics.
   Lower-`n` examples, roughly `n < 4`, tend to be wider than high-`n`
   examples, roughly `n > 6`.
2. Edge-localized modes: modes located farther out, often `r >~ 0.5`, with
   many coupled poloidal harmonics of similar amplitude. They can look more
   structured or spiky near the outer boundary than wide/global modes. Large
   magnetic shear at large radius can make the separate poloidal harmonics
   radially narrower and visually spikier near `r ~ 0.9`, even while the
   total mode envelope remains wide; this is not automatically a numerical
   boundary artifact. Distinguish physical shear-narrowed harmonic structure
   from a detached grid-scale boundary artifact.
3. Continuum-extremum-localized modes: modes localized near a maximum of the
   lower continuum boundary or a minimum of the upper continuum boundary,
   often at `r <~ 0.5`. These may be quite narrow and often have two dominant
   poloidal harmonics. Near-tangency to the extremum, without a true crossing
   through the connected mode body, supports the physical interpretation.
4. Mixed modes: modes with a combination of global, edge-localized, and/or
   extremum-localized features in the same structure. Review all connected
   lobes and continuum interactions before deciding.

Extend this list when new repeatable morphology families appear. For all
families, BAD evidence such as detached axis spikes, unresolved grid-scale
features, or continuum crossings inside the connected mode body still
overrides otherwise plausible morphology.

## BAD evidence

Strong BAD evidence includes:

- one- or few-grid-point spikes or an implausibly thin radial structure;
- a significant spike or concentration at `r=0` consistent with a boundary
  problem; use pointwise amplitude near the axis as the primary screen, not
  only integrated near-axis energy;
- jagged or mutually incoherent dominant harmonics;
- a continuum crossing through the main mode envelope or another location
  carrying appreciable pointwise or integrated mode energy;
- a true continuum crossing inside the connected mode body, including a
  shoulder, valley between connected lobes, or secondary lobe with
  appreciable amplitude, because it should produce significant continuum
  damping even when no large grid-point spike appears exactly at resonance;
- a mode that looks extremum localized at one radius but crosses continuum
  elsewhere at significant amplitude;
- unresolved, corrupt, or clearly numerical mode data.

Do not use a simple presence-of-crossing rule. Crossing severity depends on
whether the resonant point is in the connected mode body and on the nearby
mode energy. Absence of a sharp spike at the resonant grid point is not
evidence that the crossing is harmless, because the resonance can fall
between numerical grid points.

## Axis artifacts

For near-axis artifacts, inspect the maximum pointwise normalized energy
`W(r) / max(W)` in the first few radial grid points, for example
`r <= 0.03`, as well as the signed harmonics there. Integrated energy in this
region is only supporting evidence, because a one- or few-grid-point spike
can have small integrated energy while still being a clear boundary problem.

Treat a detached near-axis spike as BAD when its pointwise normalized
amplitude is appreciable, for example `max(W / max(W)) >= 0.1` near `r=0`,
especially when it is one or a few grid points wide and separated from the
main mode envelope. A very large detached axis spike, for example
`max(W / max(W)) >= 0.3`, is strong BAD evidence even if its integrated
energy fraction is small.

Use integrated near-axis energy as a secondary check. Large integrated
near-axis energy strengthens the BAD decision, but small integrated energy
does not rescue a visually detached axis spike. Do not reject a genuinely
smooth core-localized mode merely for having amplitude near the axis; the
disqualifying pattern is a narrow, separated, boundary-like feature.

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

A crossing does not have to pass through the global maximum of `W(r)` to be
disqualifying. If the crossing lies on a connected shoulder of the main
envelope, in a valley between connected lobes, or in a connected secondary
lobe with appreciable amplitude, treat it as a physically significant
continuum interaction and label the mode BAD. Reserve GOOD for crossings
only in detached or negligible tails where both pointwise and local
integrated energy are very small.

For tail crossings, do not decide from integrated energy alone. A tail can
carry little total energy but still be physically disqualifying when its
pointwise amplitude is not negligible and the local signed harmonics or
energy envelope look distorted around the crossing. Treat GOOD tail
crossings as requiring all of the following: small local integrated energy,
small pointwise normalized amplitude, and a smooth tail that is visually
detached from the main mode body. If the low-r or high-r tail is connected
to the envelope and appears resonantly distorted at the crossing, label BAD
with low confidence when the amplitude is modest but not clearly negligible.

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
