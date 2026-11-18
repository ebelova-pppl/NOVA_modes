"""Version an existing surrogate HDF5 with confirmed NOVA-filter zero records.

Example (Flux/tcsh): python scripts/confirm_nova_empty_scopes.py \
  --input research_archive/nova_surrogate_v2.h5 \
  --output research_archive/nova_surrogate_v3.h5 \
  --manifest configs/nova_confirmed_empty_scopes_20261117.json \
  --data-root "$NOVA_DITW_ROOT"

Only listed missing-selection records are changed. All original scientific
arrays are compared to the source, which is never modified.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from export_nova_surrogate import SCHEMA, scope_report, validate_export
from nova_surrogate_provenance import ConfirmedEmptyRegistry, digest


def confirm_empty_scopes(source, output, manifest, data_root):
    source, output, data_root = Path(source).resolve(), Path(output).resolve(), Path(data_root).resolve()
    summary, coverage = output.with_suffix('.summary.json'), output.with_suffix('.scopes.csv')
    if any(p.exists() for p in (output, summary, coverage)):
        raise FileExistsError('Output or companion already exists; choose a new filename')
    if output.is_relative_to(data_root):
        raise ValueError('Output must be outside the NOVA input tree')
    registry = ConfirmedEmptyRegistry(manifest)
    registry.verify_sources(data_root)
    source_digest = digest(source)
    if source_digest != registry.data['source_hdf5_sha256']:
        raise ValueError('Source HDF5 differs from the confirmed manifest baseline')
    before = validate_export(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.nova_zero_', suffix='.h5', dir=output.parent)
    os.close(fd)
    changed, reports = set(), []
    try:
        shutil.copyfile(source, temporary)
        with h5py.File(temporary, 'r+') as out:
            out.attrs['schema'] = SCHEMA
            out.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
            out.attrs['upstream_zero_parent_filename'] = source.name
            out.attrs['upstream_zero_parent_sha256'] = source_digest
            registry.write_provenance(out)
            code_paths = [Path(__file__), Path(__file__).with_name('export_nova_surrogate.py'),
                          Path(__file__).parents[1]/'src/nova_surrogate_provenance.py']
            out['provenance'].create_dataset('confirmed_empty_update_code_sha256_json',
                data=json.dumps({p.name:digest(p) for p in code_paths}, sort_keys=True),
                dtype=h5py.string_dtype())
            for scope in registry.scopes:
                shot, n_folder = scope.split('/')
                group = out[f'equilibria/{shot}/n_{int(n_folder[1:]):02}']
                if not registry.apply(group, data_root):
                    raise ValueError(f'Confirmed scope is not eligible for this update: {scope}')
                changed.update(f'{group.name.lstrip("/")}/{key}' for key in registry.CHANGED_FIELDS)
            for shot, case in out['equilibria'].items():
                for n in range(1, 11):
                    reports.append(dict(shot=shot, n_tor=n, **scope_report(case[f'n_{n:02}'])))
        result = validate_export(temporary)
        if result['exported_modes'] != before['exported_modes']:
            raise ValueError('Updating zero records changed the mode count')
        # The allowed delta is six scalar/status fields per confirmed scope.
        # Every other original dataset and all its attributes must be identical.
        compared = 0
        with h5py.File(source, 'r') as src, h5py.File(temporary, 'r') as out:
            def compare(name, obj):
                nonlocal compared
                if not isinstance(obj, h5py.Dataset):
                    return
                assert dict(obj.attrs) == dict(out[name].attrs)
                if name in changed:
                    return
                np.testing.assert_array_equal(obj[()], out[name][()])
                compared += 1
            src.visititems(compare)
        registry.verify_sources(data_root)
        if digest(source) != source_digest:
            raise RuntimeError('Source HDF5 changed during update')
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    with coverage.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    complete_cases = sum(all(r['K'] >= 0 and r['continuum_valid_samples'] > 0
                            for r in reports if r['shot'] == shot)
                         for shot in {r['shot'] for r in reports})
    result.update(schema=SCHEMA, output=str(output), sha256=digest(output), byte_size=output.stat().st_size,
                  source_hdf5_sha256=source_digest, confirmed_empty_scopes=len(registry.scopes),
                  confirmed_empty_cases=len({scope.split('/')[0] for scope in registry.scopes}),
                  unchanged_dataset_comparisons=compared, changed_scalar_status_fields=len(changed),
                  zero_K_scopes=sum(r['K'] == 0 for r in reports), complete_cases=complete_cases)
    summary.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Existing full surrogate HDF5, preserved unchanged')
    parser.add_argument('--output', type=Path, required=True, help='New version; also writes summary.json and scopes.csv')
    parser.add_argument('--manifest', type=Path, required=True, help='Confirmed-empty scope manifest with baseline/continuum hashes')
    parser.add_argument('--data-root', type=Path, required=True, help='Read-only NOVA case root; checks each named N directory')
    args = parser.parse_args()
    print(json.dumps(confirm_empty_scopes(args.input, args.output, args.manifest, args.data_root), indent=2))


if __name__ == '__main__':
    main()
