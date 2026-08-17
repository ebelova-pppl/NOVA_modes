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
Any narrow local maximum centered at `r <= 0.03` is a boundary artifact and
is BAD; an otherwise plausible continuum-extremum or core-localized family does
not rescue it. A broad smooth component that extends beyond this window, or a
rising flank whose center lies outside it, is not a narrow axis maximum. If a
short oscillatory packet is visually separated from the smooth envelope, or
has pointwise `W / max(W)` near `0.08-0.1` or larger, also label it BAD.

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

Reserve terms such as ultra-narrow, one/few-grid-point, or unresolved for
structures whose active lobe is actually at the radial grid scale: roughly a
single-point peak, only one to three radial intervals across the core/FWHM,
vertical-looking sides, a cusp-like top, or obvious sawtooth/packet behavior.
Do not use ultra-narrow as a synonym for a compact but smooth localized mode.
If the envelope is smooth, coherent across the dominant harmonics, and spans
several radial samples with a resolved rise and fall, treat it as resolved
even when its radial width is small compared with global TAEs.

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

For this family, narrowness by itself is not BAD evidence. A compact
bell-shaped envelope near a continuum extremum, with smooth sides and a
resolved width over several radial grid samples, should not be called
ultra-narrow. Reject it for width only when the peak is truly grid-unresolved
or point-like, or when another red flag such as a body crossing, axis spike,
or corrupted harmonic is present.

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

- unresolved one- or few-grid-point spikes, or an implausibly thin radial
  structure whose core/FWHM is only at the radial grid scale;
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

The NOVA files use normalized radius and normalized mode amplitude. For the
first deterministic boundary gate, set `r_ax=0.03` and calculate, for each
stored harmonic index `h`,

```text
A(h) = max_{r <= r_ax} |mode(h, r)|
axis_peak = max_h A(h)
```

Record `axis_peak_harmonic_index` as the zero-based stored array index, not an
inferred physical poloidal `m`. Also record `axis_peak_r`, whether the candidate
is a local maximum on that harmonic's complete radial profile, the connected
half-maximum width in normalized radius and radial-grid intervals, its outer
edge, and whether that component includes `r=0`.

Search for both half-maximum edges over the entire radial grid. Never stop at
`r_ax`: doing so would make a broad physical structure extending past the
search window look artificially narrow. Likewise, compare samples beyond the
window when deciding whether the candidate is a true peak or only the rising
flank of a mode centered farther out.

The ordered gate is:

```text
IF axis_peak_is_local_max
AND axis_peak >= axis_amplitude_min
AND axis_halfmax_width_grid <= axis_width_max_grid
THEN BAD_AXIS_SPIKE
AND stop evaluating later decision gates
```

Keep `axis_amplitude_min` and `axis_width_max_grid` configurable. The current
non-blind calibrated defaults are `0.2` normalized amplitude and `10` radial
grid intervals, with an inclusive `r_ax=0.03`. Every sufficiently narrow local
maximum centered at `r <= 0.03` is a boundary artifact; do not make a
family-specific exception for continuum-extremum-localized modes. Feature-only
runs may disable the decision gate while retaining all measurements.

Continue to inspect summed `W(r)` and signed-harmonic packets as supporting
evidence. Integrated near-axis energy can strengthen a BAD decision, but small
integrated energy does not rescue a narrow individual-harmonic spike.

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

The fourth deterministic BAD gate is intentionally narrower than this full
visual policy. It screens the low-hanging case in which the global total-energy
maximum itself is at the edge and its connected energy envelope is unresolved.
Calculate `W(r)=sum_m |xi_m(r)|^2`, normalize it by its global radial maximum,
and find the connected half-maximum component on the complete radial grid. The
current provisional non-blind calibration is:

```text
IF edge_energy_peak_r >= 0.97
AND edge_energy_halfmax_width_grid <= 10
THEN BAD_EDGE_SPIKE
AND stop evaluating later decision gates
```

The radius comparison is inclusive, and this gate runs after
`BAD_CONT_CROSS`. Keep `r_edge_min` and `edge_width_max_grid` configurable.
Record the global-energy peak radius, interpolated inner and outer half-maximum
edges, width in normalized radius and grid intervals, and whether the component
touches `r=1`. Also record the strongest individual-harmonic peak within the
edge window, its stored harmonic index, radius, local-maximum status, full-grid
half-maximum edges and widths, and boundary-touch status for audit.

Do not fire this gate from the individual-harmonic audit alone. The initial
`nstxu_204202` calibration showed that a literal mirrored axis rule rejects
physical edge-localized modes whose narrow harmonics are consistent with shear.
Those cases remain subject to visual review or a future structure-aware edge
rule; this deterministic gate rejects only a narrow, globally dominant total
envelope.

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

The second deterministic BAD gate screens the clearest unresolved spikes
before a later alternating-packet rule is developed. Search every stored
harmonic over the complete normalized radial grid. Treat positive local maxima
and negative local minima as separate signed lobes, and linearly interpolate
the connected half-maximum edges on the sign-adjusted harmonic profile. Do not
measure the component on `abs(mode)`, which can join adjacent `+A/-A` samples
and make an unresolved oscillation appear broad.

With normalized NOVA amplitudes, the current non-blind calibrated gate is:

```text
IF any signed local extremum has abs(amplitude) >= 0.3
AND signed_halfmax_width_grid <= 1
THEN BAD_GRID_SCALE_SPIKE
AND stop evaluating later decision gates
```

Apply it after `BAD_AXIS_SPIKE`, include one-sided extrema at `r=0` and `r=1`,
and keep both thresholds configurable. For audit output, record the strongest
candidate meeting the configured width limit, its signed amplitude and sign,
zero-based stored harmonic index, radius, interpolated inner and outer edges,
width in normalized radius and radial-grid intervals, and whether the
half-maximum component touches a radial boundary. A mode that does not fire
this strict one-grid gate can still be BAD under a later alternating-sign rule
or another numerical-structure rule.

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

The current third deterministic BAD gate is a provisional non-blind calibrated
screen applied after the axis-artifact and grid-scale-spike gates:

```text
IF n_cross > 0
AND W_star_max > 0.03
THEN BAD_CONT_CROSS
AND stop evaluating later decision gates
```

Here `W_star_max` is the largest true-crossing value of
`sum_m |xi_m(r)|^2 / max_r sum_m |xi_m(r)|^2`. The comparison is strictly
greater than the configurable threshold. The initial `0.05` calibration cleanly
separated the labeled crossing-related survivors in the 141-mode
`nstxu_204202` subset. The current `0.03` default additionally rejects two
labeled-BAD modes in the complete shot without rejecting another labeled-GOOD
mode, but it remains a provisional cross-shot threshold. Keep the
full crossing records for audit and continue to inspect borderline cases when
developing or recalibrating the rule.

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
Grid-unresolved ultra-narrow structures and axis spikes remain BAD even when
centered on an extremum. In this context, ultra-narrow means point-like or
one/few-grid-interval structure, not a smooth compact envelope. A localized
extremum mode that is radially narrow but smooth, coherent, and resolved over
several grid samples can be GOOD when crossings are absent or only in
negligible detached tails.

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
