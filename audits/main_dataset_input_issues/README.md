# Main DiTw dataset input issues

This is the maintained, incremental registry of source-data problems found
under `$NOVA_DITW_ROOT`. It is not yet a complete audit of every shot in the
live DiTw tree.

- `issues.csv` contains one stable issue record per shot, toroidal-number,
  issue-type, and affected field.
- `affected_files.csv` contains one row per affected file and uses paths
  relative to `$NOVA_DITW_ROOT`; it does not embed a Flux absolute path.

## Current coverage

- `nstxuG121123K34` was checked across all 1,258 nonempty `egn*` files and
  all ten populated `N1`--`N10` directories. All ten required `datcon#`
  files are present, nonempty, parseable, and usable for the common 201-point
  radial grid. There are no zero-byte mode files, load/shape failures,
  toroidal-number mismatches, undersized harmonic arrays, non-finite mode
  arrays, or invalid weights. Exactly 97 N4 files have `gamma_d=NaN`; every
  other stored scalar inspected in those files is finite. Those 97 paths
  exactly match `rejected_modes.csv` from the canonical K34 rules run.
- `nstxuG121123K51/N4/egn04w.8769E+01` was rechecked because it is the known
  invalid training-list input. Its only non-finite binary value is the stored
  `gamma_d` scalar.
- `nstx_135388/N8` was rechecked for empty files. Three of its 112 live
  `egn*` paths are zero-byte files; their exact names are in
  `affected_files.csv`.

Legacy continuum values greater than 999 are intentional missing-value
sentinels handled by the shared continuum loader. They are not entered here as
source defects. Likewise, a file is recorded as missing only when an
authoritative manifest or paired source establishes that it should exist;
absence cannot be inferred from a filename sequence alone.

When another shot is audited, add a stable `DITW-###` group to `issues.csv`
and one row for every affected path to `affected_files.csv`. Record the
bounded audit scope here so that an empty finding is not mistaken for
full-tree coverage.
