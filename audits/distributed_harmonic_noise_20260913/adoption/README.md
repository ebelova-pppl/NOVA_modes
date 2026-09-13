# Production v12: distributed harmonic noise

The user approved the calibrated top-two setup on 2026-09-13. Production
`tae_rules_production_v12` enables `BAD_DISTRIBUTED_HARMONIC_NOISE` as a final
additional rejection gate, retaining every earlier primary reason. The canonical
`sort_shot_mixed.py --method rules` default and conservative `sort_shot_rules.py`
share the implementation. RF-CNN classification and preprocessing are unchanged.

## Adopted condition and outputs

In a closed radial window of width 0.05, select only native high-pass centers
with simultaneous `N_hf>=4`. All three strict cuts must then hold for that
same selected population in that window:

```text
HF / full-domain top-two-harmonic energy > 0.005
AND HF / raw energy in the whole window > 0.05
AND effective radial length > 0.03
```

All harmonics remain in the numerator. The reference sums the energies of
the two strongest individual harmonics integrated over the whole radius,
without adjacency or a physical-m assumption. The local denominator includes
all raw-energy nodes in the window, including unselected structure.

`src/continuum_noise.py` supplies one shared native quadrature/high-pass helper
to both noise gates. The operator is signed `diff(xi,2)/4`, calculated before
selection. There is no continuum mask, smoothing or resampling. Each stencil
must fit wholly inside its window. The existing outside-gap gate retains its
three thresholds, region boundaries, and primary-reason precedence.

The grouped feature schema is v24, rejection ruleset v23, and severity schema
v2. `numerical_structure_features.distributed_harmonic_noise` saves the strongest
joint witness with selected center indices, pointwise participation, weighted
HF energies, ratios and effective length, plus window counts and configuration.
It does not store every scanned window in every production CSV. Witness
selection maximizes the minimum of the three normalized cut ratios, with
earliest-window tie breaking. The per-gate severity uses that same witness;
N_hf is the population-selection prerequisite, not a separately maximized ratio.

The new flat column is `gate_severity_BAD_DISTRIBUTED_HARMONIC_NOISE`; its
severity participates in `overall_rule_severity`, `rule_margin`, `nearest_gate`
and existing duplicate ranking. Strict equality at a rejection cut does not
fire. Disabled gates retain diagnostic measurements and have null severity.
Frozen v5-v11 explicitly disable the new branch. V11 keeps severity ranking.

The branch evaluates native nr>=3 whenever its requested window fits a complete
stencil. Otherwise it reports `WINDOW_UNRESOLVED`, unavailable enabled severity,
and an explicit entry in the existing resolution-warning reports. For the
default width the minimum is nr=41. Feasible effective lengths still depend on
the native window's center count. Calibration evidence is nr=201; synthetic
101/201/401 checks confirm the intended nearby-grid behavior.

## Validation and saved outputs

**Completed:** all **206 repository tests pass**, and all **39 rules shot
outputs are regenerated, verified and installed** in the existing
`sort_outputs` root. Prior exports are preserved under
`sort_outputs/before_distributed_noise_v12_20260913/`. Every installed and
backup tree matches its receipt; the RF-CNN output trees are unchanged.

The integrated gate reproduces the calibration flags and joint-witness metrics
on the full training list and every pilot TAE-side mode. All earlier pilot
measurements, input fingerprints, routing and primary reasons are unchanged
except for the one approved new rejection:

- Pilot: **E205042A01t022 N10/3470**, GOOD -> BAD_DISTRIBUTED_HARMONIC_NOISE.
- Training: **G121123J38 N8/2222**, labeled BAD, now rejected by the rules.
- **0/575 GOOD training labels** trigger the new gate. Thirty-five BAD labels
  trigger it, including the one corrected rule survivor and one EAE-routed
  mode. Existing training labels are unchanged.
- Of 25,967 pilot inputs, 6,012 are TAE-side, 19,248 EAE-routed and 707 INVALID.
  Native nr=201 throughout the measured cohort; no resolution or ranking
  fallback occurs. Final GOOD count is 1,646 before deduplication and 1,635
  representatives. No representative changes apart from removal of N10/3470.
- Rules/RF-CNN disagreements decrease **370 ->369** over 6,004 matched
  classified inputs. Eight recalculated C50/N1 modes lack a usable current AI
  classification and are excluded from comparison. Legacy AI CSVs do not
  contain input fingerprints; their preserved tree receipts, frequencies,
  resolutions and routing support this comparison. The latest 12-shot subset
  decreases **144 ->143**. These are agreement statistics, not accuracy.

Review outputs:

- [Current latest-12 disagreements](latest12_disagreements.csv), preserving
  the membership of `audits/pilot12_v11_20260910/selection.csv`.
- [All 39-shot disagreements](current_disagreements.csv).
- [Removed disagreement](disagreements_removed.csv); no new disagreement was
  introduced. Historical disagreement and Elena review files remain intact.
- [Adopted pilot change](adopted_changes.csv), [training change](training_changes.csv),
  [selection changes](selection_changes.csv), and [per-shot counts](shot_summary.csv).
- [Verification](verification.json), [publication/backups](publication.json),
  and [test receipt](tests.json).

The [pre-adoption calibration](../top2_pilot39/README.md) found zero flagged
GOOD training labels out of 575, one newly rejected BAD-labeled training mode
J38 N8/2222, and only E205042A01t022 N10/3470 among all 1,647 pilot survivors.
Adoption checks require the integrated gate to reproduce all prior scan flags
and joint-witness metrics, with every previous production feature and primary
reason preserved unless this final gate newly rejects the mode.

`adopt_gate.py` stages all 39 checked rules shot exports, verifies the full
active training list and both compared input inventories, and installs only
verified outputs. Earlier rules trees receive verified backups. AI trees remain
unchanged. The current training list, manual labels and invalid-input registry
are not modified. The full-database rollout remains paused for input repairs.

The versioned preset SHA-256 is
`6cec796ae20bac12f2f66bd18ac20a14d9e502aa64453c6f2b5ad10f7b54f925`.
Full regenerated outputs and training comparisons are ignored under
`outputs/review_distributed_noise_v12_20260913/`. Compact verification,
decision/selection changes, current disagreements and publication receipts
are kept beside this README.

For a new shot with the configured environment (Flux/tcsh):

```tcsh
python scripts/sort_shot_mixed.py --method rules \
  --shot_dir "$NOVA_DITW_ROOT/SHOT" --out_dir /path/to/new/output
```

For exact v12 calibration without promoting survivors to GOOD, pass
`--rule_config tae_rules_production_v12` to `scripts/sort_shot_rules.py`.
