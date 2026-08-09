# TAE-like mode labeling policy

Use this policy to reason from physical morphology and continuum geometry. It
is deliberately qualitative: scalar measurements support inspection but do
not replace it.

## Evidence to inspect

For every mode inspect:

- all relevant signed poloidal harmonics `xi_m(r)`;
- the radial energy envelope `W(r) = sum_m |xi_m(r)|^2`;
- radial width, smoothness, coherence, and peak location;
- grid-scale oscillations or multiple closely spaced peaks at the mode peak
  or within the connected mode body, including artifacts visible in only one
  signed harmonic;
- which morphology family best describes the mode, while allowing mixed
  cases that combine multiple families;
- near-axis behavior and radial-boundary artifacts, including both
  pointwise normalized amplitude and integrated energy near `r=0`;
- outer-boundary behavior near `r=1`, including endpoint spikes in the
  total envelope and in any individual signed harmonic;
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

## Red-flag precedence

Before assigning a morphology family, first apply disqualifying gates for
continuum crossings and radial-boundary artifacts. A plausible family label
such as edge-localized, continuum-extremum-localized, or mixed can explain
the overall shape, but it must not rescue a mode with appreciable continuum
interaction or grid-scale numerical structure.

If a true continuum crossing carries pointwise `W(r_cross) / max(W) >= ~0.1`,
label the mode BAD unless the crossing is unmistakably in a smooth,
visually detached tail with very small local integrated energy. If the
crossing pointwise energy is large, roughly `>= ~0.2-0.3`, treat it as
strong BAD evidence even when it is away from the global energy peak. A
crossing on a connected shoulder, a connected secondary lobe, or a valley
between connected lobes is part of the mode body for this purpose.

Near-axis artifacts include both isolated one/few-grid-point spikes and
short grid-scale oscillatory packets in the signed harmonics near `r=0`.
If such a packet is visually separated from the smooth envelope, or has
pointwise `W / max(W)` near `0.08-0.1` or larger, label BAD unless the mode
is genuinely smooth and core-localized all the way to the axis.

Outer-boundary artifacts near `r=1` are also disqualifying. Inspect both the
summed envelope `W(r)` and the individual signed harmonics. A large endpoint
spike or one/few-grid-point oscillatory packet in any single harmonic is BAD
evidence even when the summed `W(r)` is modest because other harmonics
dominate elsewhere. Accept edge-localized modes only when high-r structure is
smoothly connected, radially resolved, and coherent across neighboring
harmonics rather than localized at the endpoint.

Grid-scale oscillations at the mode peak or inside the connected mode body
are disqualifying even away from both radial boundaries. This includes
several closely spaced grid-point peaks, sawtooth-like oscillations, or a
large one/few-grid-point spike in a single dominant harmonic. A smooth
resolved envelope with ordinary nodes or phase changes can still be physical;
the red flag is unresolved radial structure carrying appreciable amplitude
inside the mode body.

When any red-flag gate is borderline but not clearly clean, prefer `bad`
with low confidence over `good` with low confidence. Use `good` only after
the mode passes these red-flag checks.

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
   from a detached grid-scale boundary artifact. Sign changes or phase
   variation among coupled edge harmonics are common in this family and do
   not by themselves imply a separated spike or numerical boundary problem;
   judge separation from the connected radial energy envelope and harmonic
   coherence, not from sign reversal alone.
3. Continuum-extremum-localized modes: modes localized near a maximum of the
   lower continuum boundary or a minimum of the upper continuum boundary,
   often at `r <~ 0.5`. These may be quite narrow and often have two dominant
   poloidal harmonics. Near-tangency to the extremum, without a true crossing
   through the connected mode body, supports the physical interpretation.
4. Mixed modes: modes with a combination of global, edge-localized, and/or
   extremum-localized features in the same structure. Review all connected
   lobes and continuum interactions before deciding.

Extend this list when new repeatable morphology families appear. For all
families, BAD evidence such as detached axis spikes, short grid-scale
near-axis packets, unresolved grid-scale features, or continuum crossings
inside the connected mode body still overrides otherwise plausible
morphology.

## BAD evidence

Strong BAD evidence includes:

- one- or few-grid-point spikes or an implausibly thin radial structure;
- a significant spike or concentration at `r=0` or `r=1` consistent with a
  boundary problem; use pointwise amplitude and individual-harmonic structure
  as primary screens, not only integrated boundary energy or summed `W(r)`;
- a short grid-scale oscillatory packet near `r=0` in the signed harmonics,
  especially when separated from the smooth envelope or accompanied by a
  narrow energy bump;
- a single one/few-grid-point peak at `r=0` in any appreciable harmonic,
  even if the rest of the mode looks smooth or the summed/integrated
  near-axis energy is small;
- a one/few-grid-point endpoint spike or grid-scale oscillatory packet near
  `r=1` in any individual signed harmonic, even if the summed envelope
  `W(r)` is not large at the boundary;
- grid-scale oscillations at the main peak or within the connected mode body,
  especially several closely spaced peaks, a sawtooth-like radial pattern, or
  a large one/few-grid-point spike in a single harmonic;
- jagged or mutually incoherent dominant harmonics;
- an apparently sharp signed-harmonic feature that is detached from the
  connected energy envelope or grid-scale in radius; do not count an
  ordinary sign change within a coherent edge-localized envelope as
  detachment by itself;
- a continuum crossing through the main mode envelope or another location
  carrying appreciable pointwise or integrated mode energy;
- a true continuum crossing with pointwise `W(r_cross) / max(W) >= ~0.1`
  unless it is unmistakably in a smooth detached tail with very small local
  integrated energy;
- a true continuum crossing with large pointwise energy, roughly
  `W(r_cross) / max(W) >= ~0.2-0.3`, even away from the global peak;
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
`r <= 0.03`, as well as the signed harmonics there. Also inspect the next
few grid points beyond that window when the plot shows a short oscillatory
packet at low radius. Integrated energy in this region is only supporting
evidence, because a one- or few-grid-point spike or short grid-scale packet
can have small integrated energy while still being a clear boundary problem.

Treat a detached near-axis spike as BAD when its pointwise normalized
amplitude is appreciable, for example `max(W / max(W)) >= 0.1` near `r=0`,
especially when it is one or a few grid points wide and separated from the
main mode envelope. A single one/few-grid-point peak at the axis is also BAD
when it appears in any appreciable individual harmonic, even if the summed
`W(r)` or integrated near-axis energy is modest. A very large detached axis
spike, for example `max(W / max(W)) >= 0.3`, is strong BAD evidence even if
its integrated energy fraction is small.

Treat a near-axis grid-scale oscillatory packet as BAD when it is visually
separated from the smooth envelope or accompanied by a narrow energy bump
with pointwise `W / max(W)` near `0.08-0.1` or larger. Do not rescue this
pattern because the integrated near-axis energy is small. The relevant
distinction is smooth continuation of a core-localized mode versus a
boundary-like packet or spike.

Use integrated near-axis energy as a secondary check. Large integrated
near-axis energy strengthens the BAD decision, but small integrated energy
does not rescue a visually detached axis spike. Do not reject a genuinely
smooth core-localized mode merely for having amplitude near the axis; the
disqualifying pattern is a narrow, separated, boundary-like feature.

## Outer-boundary artifacts

For high-r boundary artifacts, inspect both the summed radial envelope
`W(r) / max(W)` and the largest individual signed-harmonic amplitude near
`r=1`, for example
`max_m |xi_m(r)| / max_{m,r} |xi_m(r)|` in the last few grid points. The
summed envelope can hide a boundary problem when only one harmonic has a
large endpoint spike while other harmonics dominate the physical body.

Treat a one/few-grid-point endpoint spike or short grid-scale oscillatory
packet near `r=1` as BAD when it appears in any individual harmonic and is
not part of a smooth resolved edge-localized envelope. This remains BAD even
if `W(r) / max(W)` is modest at the boundary. A high-r packet with pointwise
`W / max(W) >= ~0.1` is suspicious, and `>= ~0.2-0.3` is strong BAD evidence,
but an individual-harmonic endpoint spike can be disqualifying below those
summed-energy thresholds.

Do not confuse this with ordinary type-2 edge structure: in a physical
edge-localized mode, separate harmonics can be radially narrow because of
large magnetic shear, but the total envelope remains broad/resolved and the
high-r structure is smoothly connected across neighboring harmonics rather
than a single endpoint blow-up.

## Grid-scale structure inside the mode body

Inspect the radial smoothness of every appreciable signed harmonic at the
mode peak and throughout the connected mode body. A mode is BAD when the
peak or body contains unresolved grid-scale oscillations, multiple closely
spaced grid-point peaks, or a large one/few-grid-point spike in any dominant
or otherwise appreciable harmonic. This remains BAD even if the artifact is
confined to a single poloidal harmonic; a single corrupted harmonic can make
the mode unsuitable for training.

Do not reject ordinary physical nodes, sign changes, or shear-narrowed
harmonic structure when the envelope is smooth and resolved. The
disqualifying pattern is radial structure at the grid scale carrying
appreciable amplitude inside the connected mode body, especially when it
creates several neighboring sharp peaks rather than one smooth lobe.

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
disqualifying. If the crossing carries pointwise
`W(r_cross) / max(W) >= ~0.1`, label BAD unless it is unmistakably in a
smooth detached tail with very small local integrated energy. If the crossing
lies on a connected shoulder of the main envelope, in a valley between
connected lobes, or in a connected secondary lobe with appreciable
amplitude, treat it as a physically significant continuum interaction and
label the mode BAD. Crossings with large pointwise energy, roughly
`W(r_cross) / max(W) >= ~0.2-0.3`, are strong BAD evidence even when the
global peak is elsewhere. Reserve GOOD for crossings only in detached or
negligible tails where both pointwise and local integrated energy are very
small.

For tail crossings, do not decide from integrated energy alone. A tail can
carry little total energy but still be physically disqualifying when its
pointwise amplitude is not negligible and the local signed harmonics or
energy envelope look distorted around the crossing. Treat GOOD tail
crossings as requiring all of the following: small local integrated energy,
small pointwise normalized amplitude, and a smooth tail that is visually
detached from the main mode body. If the low-r or high-r tail is connected
to the envelope and appears resonantly distorted at the crossing, label BAD
with low confidence when the amplitude is modest but not clearly negligible.
In borderline cases, prefer `bad, low` over `good, low` unless the crossing
is clearly detached and negligible by all three checks.

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
