"""Copy complete n=1..10 cases without changing scientific data or labels.

Example: python scripts/extract_complete_surrogate_cases.py \
  --input research_archive/nova_surrogate_v2.h5 \
  --output research_archive/nova_surrogate_v2_complete_cases.h5
Completeness requires valid parameters and all ten K/mode selections and
continua. Radial masks are preserved unchanged.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from export_nova_surrogate import validate_export
from nova_surrogate_provenance import digest


def extract_complete_cases(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    summary = output.with_suffix('.summary.json')
    case_list = output.with_suffix('.cases.csv')
    if any(p.exists() for p in (output, summary, case_list)):
        raise FileExistsError('Subset destination or companion already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    source_digest = digest(source)
    rows = []
    fd, temporary = tempfile.mkstemp(prefix='.nova_subset_', suffix='.h5', dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(source, 'r') as src, h5py.File(temporary, 'w') as out:
            source_case_count = len(src['case_ids'])
            names = []
            for name in src['case_ids'].asstr()[:]:
                case = src['equilibria'][name]
                if case['parameters_valid'][()] and all(
                        case[f'n_{n:02}/mode_targets_valid'][()]
                        and case[f'n_{n:02}/continuum_available'][()] for n in range(1, 11)):
                    names.append(name)
            out.attrs.update(dict(src.attrs))
            out.attrs.update(
                subset_created_utc=datetime.now(timezone.utc).isoformat(),
                subset_definition='parameters_valid and mode_targets_valid and continuum_available for every n=1..10; radial masks retained',
                source_hdf5_sha256=source_digest,
                source_hdf5_filename=source.name,
                subset_builder_sha256=digest(__file__),
                radial_coverage_note='Completeness is by n, not by radius. Use continuum_valid; masked radii remain NaN.')
            src.copy('parameter_names', out)
            out.create_dataset('case_ids', data=names, dtype=h5py.string_dtype('utf-8'))
            src.copy('provenance', out)
            equilibria = out.create_group('equilibria')
            for name in names:
                src.copy(src['equilibria'][name], equilibria, name=name)
                case = src['equilibria'][name]
                row = {'case_id': name}
                for n in range(1, 11):
                    group = case[f'n_{n:02}']
                    row[f'K_n{n}'] = int(group['K'][()])
                    row[f'valid_continuum_samples_n{n}'] = int(group['continuum_valid'][:].sum())
                row['total_modes'] = sum(row[f'K_n{n}'] for n in range(1, 11))
                rows.append(row)
        result = validate_export(temporary)
        # Compare every copied dataset and its attributes, including strings,
        # masks, profiles, and empty arrays for K=0 records.
        compared = 0
        with h5py.File(source, 'r') as src, h5py.File(temporary, 'r') as out:
            assert set(out['case_ids'].asstr()[:]) == set(out['equilibria']) == set(names)

            def compare(name, obj):
                nonlocal compared
                if not isinstance(obj, h5py.Dataset) or name == 'case_ids':
                    return
                before, after = src[name][()], obj[()]
                if np.asarray(before).dtype.kind in 'SUO':
                    np.testing.assert_array_equal(before, after)
                else:
                    np.testing.assert_allclose(before, after, rtol=0, atol=0, equal_nan=True)
                assert dict(src[name].attrs) == dict(obj.attrs)
                compared += 1

            out.visititems(compare)
        if digest(source) != source_digest:
            raise RuntimeError('Source HDF5 changed during extraction')
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    with case_list.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['case_id', 'total_modes'])
        writer.writeheader()
        writer.writerows(rows)
    result.update(output=str(output), sha256=digest(output), bytes=output.stat().st_size,
                  source_hdf5_sha256=source_digest, source_case_count=source_case_count,
                  copied_dataset_comparisons=compared, all_n_complete=True,
                  radial_masks_preserved=True,
                  zero_K_scopes=sum(row[f'K_n{n}'] == 0 for row in rows for n in range(1, 11)))
    summary.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Validated source surrogate HDF5')
    parser.add_argument('--output', type=Path, required=True,
                        help='New subset HDF5; also writes summary.json and cases.csv')
    args = parser.parse_args()
    print(json.dumps(extract_complete_cases(args.input, args.output), indent=2))


if __name__ == '__main__':
    main()
