# C50 N1 continuum / stability-run inconsistency, 2026-09-09

**Status: unresolved upstream data consistency; no rule or label change.**
The user visually inspected every TAE-side mode in `nstxuG142301C50` and
reported displaced resonance structure for n=1, with other n appearing
consistent. This follow-up is explicitly non-blind.

All **14 N1 TAE-side modes** have an inner TAE-gap crossing at 0.03<=r<0.75
lying **2.42–9.73 native grid intervals farther out** than the nearest
singularity listed by the original NOVA stability run at that exact mode
frequency. Median separation is 4.63 intervals; one interval is 0.005.
The first two modes use the lower boundary and the remaining twelve the
upper boundary. This comparison does not assign a continuum-branch identity
to every logged singularity or assess the crowded outer-edge spectrum.

| N1 mode | Stability-log singularity r | Datcon upper crossing r | Separation in grid intervals |
| --- | ---: | ---: | ---: |
| 8889 | 0.470 | 0.517203 | 9.44 |
| 9040 | 0.465 | 0.511017 | 9.20 |
| 9225 | 0.455 | 0.503648 | 9.73 |

Raw signed-mode inspection independently shows sharp structure near the
logged locations. In these three examples, the strongest signed second
difference within 0.4<r<0.55 occurs at 0.475, 0.465, and 0.455, respectively.
The largest harmonic amplitudes there are approximately 0.499, 0.898, and
0.726. Thus the structure is appreciable and the displacement is large
compared with the +/-2-grid crossing window. N1/8889 and N1/9040 currently
survive the rules; this provides a concrete reason to treat their GOOD
classification as provisional while the data are investigated. N1/9225
currently fails BAD_CONT_CROSS_WINDOW. No manual decisions were applied.

## Checks and limits

- Raw mode/datcon fingerprints match the current saved output for all 14
  modes. Recomputed crossing records match the saved records exactly.
- Every inspected mode has 22 stored harmonics, nr=201, and ntor=1.
  `datcon1` covers the expected 1-based radial indices 3 through 199.
  Its raw lower/upper arrays are exactly equal to the shared loader's output:
  **the recently adopted continuum cleanup changes zero points** here.
- The matching `out_go` frequency is omega squared, and agrees with each
  mode's squared binary-header frequency to floating-point precision.
  The logged singularity radius therefore supplies evidence independent of
  the Python viewer and sorter. This is not just a plotting-marker offset.
- Current qprofile, bprofile, rgrid, gridparam, mapdsk, mpout1, and transp.dat
  bytes are identical across N1 through N10. N1's saved continuum input and
  compiled harmonic-range configuration both specify n=1 and harmonics 0–21.
  No obvious current file-selection, point-count, or harmonic-count mismatch
  was found. Array plots retain stored harmonic indices.
- The N1 mode/log timestamps are March 25; the current continuum timestamp
  is June 20. Dates alone do not establish which inputs were actually used.
  The original `equout` and `equou1` targets under `/local/` are absent.
  Consequently this audit cannot prove the historical stability and
  continuum calculations used the same equilibrium or identify which is
  wrong. The earlier wrong-q-profile incident in project history is a
  possible diagnostic lead, not an established cause here.
- The user reports N2–N10 look consistent. This audit does not independently
  certify all their modes or infer an automatic mismatch threshold from
  nearest-log-singularity distance; logged singularities span other branches
  and need not exhaust every boundary crossing.

Recommendation: keep these N1 modes aside from rule calibration and
physical-mode acceptance pending a
paired upstream check/recalculation of continuum and modes using the same
equilibrium, density settings, radial mesh, harmonic range, and solver setup.
There is no established radius or frequency correction to apply. Other n
can continue through the existing review. The user question list, training
labels, production rules, and saved classifications remain unchanged.
The inventory notes flag the issue without changing checked membership.

## Evidence and reproduction

`measurements.csv` records all 14 comparisons, exact input fingerprints,
frequencies, and current rule outcomes. `summary.json` records input hashes,
current input equality checks, and missing historical cache targets.
`check_alignment.py` recomputes these checks without modifying raw data:

```text
python audits/c50_n1_alignment_20260909/check_alignment.py --shot-dir /path/to/nstxuG142301C50 --rules-dir /path/to/sort_outputs/nstxuG142301C50 --out-dir outputs/review_c50_n1_alignment
```

Diagnostic figures remain local under
`outputs/review_c50_n1_alignment_20260909/`: `n1_all.png` shows all 14 signed
profiles with true crossing markers; `continuum_raw_loaded.png` compares
raw and loaded continua for n=1,2,3. Large plots are excluded from Git.
