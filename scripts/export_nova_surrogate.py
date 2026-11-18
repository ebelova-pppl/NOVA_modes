#!/usr/bin/env python3
"""Export inventoried NOVA equilibria, TAE continua and selected mode scalars.

Example (Flux/tcsh, after sourcing configs/paths/nova_paths.flux.csh):
  python scripts/export_nova_surrogate.py --data-root "$NOVA_DITW_ROOT" \
      --snapshot-root ../data_mixed --output research_archive/nova_surrogate.h5

No sorting or relabeling is performed. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from _repo_bootstrap import ensure_repo_src_on_path
REPO = ensure_repo_src_on_path()
from build_accepted_catalog import read_csv
from cont_features import CONTINUUM_PREPROCESSING_VERSION, load_datcon_for_mode
from input_validity import load_input_validity_registry
from mode_features import radial_centroid, radial_width
from nova_equilibrium import PARAMETERS, load_equilibrium
from nova_mode_loader import load_mode_from_nova
from nova_surrogate_provenance import ConfirmedEmptyRegistry, ReconstructionRegistry
from tae_rule_io import input_fingerprint, sha256_file

SCHEMA = 'nova-surrogate-v3'
STR = h5py.string_dtype('utf-8')
MODE_NUMBERS = ('omega', 'r_loc', 'r_width', 'psi_N_loc', 'psi_N_width')
MODE_STRINGS_V1 = ('mode_id', 'path', 'source_root', 'input_fingerprint',
                'family', 'review_status', 'rule_config', 'manual_override')
MODE_STRINGS = MODE_STRINGS_V1 + ('selection_input_fingerprint', 'mode_sha256', 'pair_provenance')


def put(group, name, data, units=None, description=None):
    if isinstance(data, str):
        ds = group.create_dataset(name, data=data, dtype=STR)
    else:
        ds = group.create_dataset(name, data=data)
    if units is not None:
        ds.attrs['units'] = units
    if description is not None:
        ds.attrs['description'] = description
    return ds


def mode_moments(mode):
    """Use the shared amplitude-squared discrete moments, with both abscissae."""
    if mode.ndim != 2 or mode.shape[1] < 2 or not np.all(np.isfinite(mode)):
        raise ValueError('Invalid mode amplitude array')
    if not np.any(mode):
        raise ValueError('All-zero mode amplitude array')
    r = np.linspace(0, 1, mode.shape[1])
    result = {}
    for name, coordinate in (('r', r), ('psi_N', r*r)):
        loc = float(radial_centroid(mode, coordinate))
        result[name+'_loc'] = loc
        result[name+'_width'] = float(radial_width(mode, coordinate, loc))
    return result


def resolve_pair(row, roots, reconstruction=None, *, zero_selection=False):
    """Prefer the saved exact pair; allow an explicitly audited reconstruction."""
    key = row['mode_key']
    parts = Path(key).parts
    if len(parts) != 3 or any(p in ('.', '..') for p in parts) or Path(key).is_absolute():
        raise ValueError(f'Invalid portable mode key: {key}')
    ntor = int(row['ntor'])
    if parts[0] != row['shot'] or parts[1] != f'N{ntor}':
        raise ValueError(f'Mode key disagrees with shot/n: {key}')
    if len(row['input_fingerprint']) != 64:
        raise ValueError(f'Missing saved fingerprint: {key}')
    for label, root in roots:
        path = root / key
        continuum = path.parent / f'datcon{ntor}'
        if not path.is_file() or not path.stat().st_size or not continuum.is_file():
            continue
        if input_fingerprint(path, continuum) == row['input_fingerprint']:
            return label, path
    return reconstruction.resolve(row, roots, zero_selection=zero_selection) if reconstruction else None


def pair_continuum(resolved, ntor):
    return resolved[2] if len(resolved) == 3 else resolved[1].parent/f'datcon{ntor}'


def load_selected(row, catalog_row, resolved, nr):
    label, path = resolved[:2]
    continuum = pair_continuum(resolved, int(row['ntor']))
    reconstructed = len(resolved) == 3
    fingerprint = input_fingerprint(path, continuum) if reconstructed else row['input_fingerprint']
    mode, omega, _, ntor = load_mode_from_nova(path)
    if mode.shape[1] != nr or ntor != int(row['ntor']):
        raise ValueError(f'Mode grid/n differs from equilibrium: {path}')
    if not np.isfinite(omega) or omega <= 0 or not np.isclose(
            omega, float(catalog_row['omega']), rtol=1e-12, atol=0):
        raise ValueError(f'Mode frequency differs from accepted catalog: {path}')
    record = dict(omega=omega, **mode_moments(mode), mode_id=path.name,
                  path=row['mode_key'], source_root=label,
                  input_fingerprint=fingerprint, selection_input_fingerprint=row['input_fingerprint'],
                  mode_sha256=sha256_file(path),
                  pair_provenance='reconstructed_continuum' if reconstructed else 'exact_saved_pair')
    for key in ('family', 'review_status', 'rule_config', 'manual_override'):
        record[key] = catalog_row[key]
    # Recheck after reading; these source trees can be updated by ongoing runs.
    if input_fingerprint(path, continuum) != fingerprint:
        raise ValueError(f'Input changed while reading: {path}')
    return record


def scope_holds(shot, ntor, registry, holds):
    reasons = []
    diagnostic = registry.diagnostic(shot, ntor)
    if diagnostic:
        reasons.append(diagnostic)
    for row in holds:
        if row['shot'] != shot:
            continue
        scope = row['scope']
        if scope == 'all n' or f'N{ntor}' in scope.split('/'):
            reasons.append(row['hold'] + ': ' + row['reason'])
    return '; '.join(reasons)


def zero_selection_anchors(path, empty_n):
    """Retain valid classified pairs; INVALID diagnostics can lack parsed n."""
    anchors = defaultdict(list)
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('processing_status') == 'INVALID' or not row.get('ntor', '').isdigit():
                continue
            n = int(row['ntor'])
            if n in empty_n and len(anchors[n]) < 5 and row.get('input_fingerprint'):
                anchors[n].append({k: row[k] for k in ('mode_key', 'shot', 'ntor', 'input_fingerprint')})
    return anchors


def write_scope(group, *, ntor, nr, rows, catalog, summaries, roots,
                hold, equilibrium_ok, anchors, reconstruction=None, confirmed_empty=None):
    """Keep continuum availability separate from verified selected-mode labels."""
    put(group, 'n_tor', ntor)
    r = np.linspace(0, 1, nr) if nr else np.array([], dtype=float)
    put(group, 'r', r, '1', 'sqrt(normalized poloidal flux); native mode grid')
    put(group, 'psi_N', r*r, '1', 'normalized poloidal flux; no uniform-psi resampling')
    group.attrs['continuum_preprocessing_version'] = CONTINUUM_PREPROCESSING_VERSION
    group.attrs['continuum_frequency_conversion'] = 'sqrt(datcon omega_squared) exactly once'
    catalog_k = len(rows) if ntor in summaries else -1
    put(group, 'catalog_K', catalog_k)
    records, issues, source = [], [], None
    state = 'available'
    if not equilibrium_ok:
        state = 'missing_equilibrium'
    elif hold:
        state = 'held'
        issues.append(hold)
    elif ntor not in summaries:
        state = 'missing_selection'
    else:
        if int(summaries[ntor]['n_final_good']) != len(rows):
            raise ValueError(f'{group.name}: selected-list count differs from n summary')
        matches = [resolve_pair(row, roots, reconstruction) for row in rows]
        if any(m is None for m in matches):
            state = 'unmatched_input_pair'
            issues.extend('No saved-fingerprint match: '+row['mode_key']
                          for row, match in zip(rows, matches) if match is None)
        elif rows:
            # One n scope must use a single continuum version.
            digests = {sha256_file(pair_continuum(match, ntor)) for match in matches}
            if len(digests) != 1:
                state = 'mixed_continuum_versions'
                issues.append('Selected modes resolve to different continuum versions')
            else:
                source = matches[0]
                records = [load_selected(row, catalog[row['mode_key']], match, nr)
                           for row, match in zip(rows, matches)]
        else:
            # An empty selected list is a label only with an installed n summary
            # and a saved classified-input pair tying it to this continuum.
            for row in anchors.get(ntor, []):
                source = resolve_pair(row, roots, reconstruction, zero_selection=True)
                if source:
                    put(group, 'zero_K_anchor_path', row['mode_key'])
                    put(group, 'zero_K_anchor_fingerprint', input_fingerprint(source[1], pair_continuum(source,ntor)))
                    put(group, 'zero_K_anchor_selection_fingerprint', row['input_fingerprint'])
                    break
            if source is None:
                state = 'unverified_zero_selection'
                issues.append('No classified-input pair verifies the zero-GOOD continuum')
    if state == 'available' and source is not None and len(source) == 3:
        state = 'available_reconstructed'
    paired = state in ('available', 'available_reconstructed')
    continuum_state = 'missing'
    low, high = np.full(nr, np.nan), np.full(nr, np.nan)
    valid = np.zeros(nr, dtype=bool)
    if state == 'held':
        continuum_state = 'held'
    elif equilibrium_ok:
        if source is None:
            # Standalone current continuum can support the first surrogate task.
            # It does not validate the old K/mode targets.
            source = ('live', roots[0][1]/group.parent.name.split('/')[-1]/f'N{ntor}'/'egn_placeholder')
        label, path = source[:2]
        cpath = pair_continuum(source, ntor)
        try:
            before = sha256_file(cpath)
            low2, high2, i1, i2 = load_datcon_for_mode(str(cpath.parent/'egn_placeholder'), nr)
            valid = (np.isfinite(low2) & np.isfinite(high2) & (low2 >= 0)
                     & (high2 >= low2))
            low[valid], high[valid] = np.sqrt(low2[valid]), np.sqrt(high2[valid])
            if sha256_file(cpath) != before:
                raise ValueError(f'Continuum changed while reading: {cpath}')
            put(group, 'continuum_path', '/'.join(cpath.parts[-3:]))
            put(group, 'continuum_source_root', reconstruction.continuum_label(cpath) if len(source)==3 else label)
            put(group, 'continuum_sha256', before)
            put(group, 'continuum_provenance', 'historical_reconstruction' if len(source)==3 else
                ('exact_saved_pair' if paired else 'standalone_current'))
            if len(source)==3:
                scope = '/'.join(cpath.parts[-3:-1])
                put(group, 'reconstruction_evidence_json', json.dumps(reconstruction.evidence_for(scope), sort_keys=True))
            group.attrs['datcon_i1'] = i1
            group.attrs['datcon_i2'] = i2
            continuum_state = ('available_paired' if paired else 'available_unpaired') if valid.any() else 'no_valid_samples'
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
            continuum_state = 'missing_or_invalid'
            valid[:] = False
            low[:] = high[:] = np.nan
    if not valid.any() and paired:
        state = 'missing_or_invalid_continuum'
        paired = False
    if not paired:
        records = []
    records.sort(key=lambda x: (x['omega'], x['mode_id']))
    put(group, 'status', state)
    put(group, 'status_detail', '\n'.join(issues))
    put(group, 'continuum_status', continuum_state)
    put(group, 'continuum_available', bool(valid.any()))
    put(group, 'omega_minus', low, '1', 'omega/omega_ref; lower TAE continuum boundary')
    put(group, 'omega_plus', high, '1', 'omega/omega_ref; upper TAE continuum boundary')
    put(group, 'continuum_valid', valid)
    put(group, 'mode_targets_valid', paired)
    put(group, 'K', len(records) if paired else -1, description='-1 means unknown/withheld, never a no-TAE label')
    put(group, 'has_good_tae', int(bool(records)) if paired else -1,
        description='1=yes, 0=no selected GOOD mode, -1=unknown/withheld')
    modes = group.create_group('modes')
    modes.attrs['location_definition'] = 'sum_mi |xi_psi(m,i)|^2 x_i / sum_mi |xi_psi(m,i)|^2'
    modes.attrs['width_definition'] = 'RMS deviation about the centroid using the same discrete weights'
    modes.attrs['weighting'] = 'Native uniform-r samples; no Jacobian, volume weighting, or psi resampling'
    modes.attrs['lobes_status'] = 'unassessed; no calibrated signed-lobe definition'
    for key in MODE_NUMBERS:
        put(modes, key, np.array([row[key] for row in records], dtype=float), '1')
    for key in MODE_STRINGS:
        modes.create_dataset(key, data=[row[key] for row in records], dtype=STR)
    if confirmed_empty:
        confirmed_empty.apply(group, roots[0][1])
    return scope_report(group)


def scope_report(group):
    return dict(status=group['status'].asstr()[()],
                continuum_status=group['continuum_status'].asstr()[()],
                K=int(group['K'][()]), catalog_K=int(group['catalog_K'][()]),
                continuum_valid_samples=int(group['continuum_valid'][:].sum()),
                details=group['status_detail'].asstr()[()])


def validate_export(path):
    """Reopen the product and enforce its coordinate, mask and count contract."""
    counts, continuum_counts = Counter(), Counter()
    cases, scalars, modes = 0, 0, 0
    with h5py.File(path, 'r') as f:
        if f.attrs['schema'] not in ('nova-surrogate-v1', 'nova-surrogate-v2', SCHEMA):
            raise ValueError('Unexpected HDF5 schema')
        for case in f['equilibria'].values():
            cases += 1
            p = case['parameters']
            assert set(p) == set(PARAMETERS)
            if case['parameters_valid'][()]:
                assert all(np.isfinite(p[k][()]) for k in PARAMETERS)
                assert p['rho_axis'][()] > 0
                assert 0 <= p['psi_N_qmin'][()] <= 1
                if p['psi_N_qmin'][()] == 0:
                    assert p['q_0'][()] == p['q_min'][()]
                scalars += 1
            for n in range(1, 11):
                g = case[f'n_{n:02}']
                r, psi = g['r'][:], g['psi_N'][:]
                assert np.array_equal(psi, r*r)
                assert len(r) == 0 or (r[0] == 0 and r[-1] == 1 and np.all(np.diff(r) > 0))
                lo, hi, valid = g['omega_minus'][:], g['omega_plus'][:], g['continuum_valid'][:]
                assert lo.shape == hi.shape == valid.shape == r.shape
                assert np.all(np.isfinite(lo[valid])) and np.all(hi[valid] >= lo[valid])
                assert np.all(np.isnan(lo[~valid])) and np.all(np.isnan(hi[~valid]))
                k, present, ok = g['K'][()], g['has_good_tae'][()], g['mode_targets_valid'][()]
                assert (k >= 0 and present == int(k > 0) and valid.any()) if ok else (k == present == -1)
                m = g['modes']
                strings = MODE_STRINGS if f.attrs['schema']!='nova-surrogate-v1' else MODE_STRINGS_V1
                assert all(len(m[key]) == (k if ok else 0) for key in MODE_NUMBERS+strings)
                if f.attrs['schema']!='nova-surrogate-v1':
                    for current, saved, kind in zip(m['input_fingerprint'].asstr()[:],
                            m['selection_input_fingerprint'].asstr()[:],m['pair_provenance'].asstr()[:]):
                        assert len(current)==len(saved)==64
                        assert kind in ('exact_saved_pair','reconstructed_continuum')
                        if kind=='exact_saved_pair': assert current==saved
                if g['status'].asstr()[()] == ConfirmedEmptyRegistry.STATUS:
                    assert ok and k == present == 0 and g['catalog_K'][()] == -1
                    assert g['zero_K_provenance'].asstr()[()] == 'nova_internal_filter'
                    evidence = json.loads(g['zero_K_evidence_json'].asstr()[()])
                    assert evidence['mode_file_count'] == 0 and not evidence['classifier_run']
                    assert evidence['confirmation']['statement']
                    assert evidence['continuum_sha256'] == g['continuum_sha256'].asstr()[()]
                    assert g['continuum_status'].asstr()[()] == 'available_upstream_zero'
                assert np.all(np.diff(m['omega'][:]) >= 0)
                for key in MODE_NUMBERS:
                    assert np.all(np.isfinite(m[key][:]))
                np.testing.assert_allclose(m['psi_N_loc'][:], m['r_loc'][:]**2+m['r_width'][:]**2, atol=2e-13)
                assert not any(key in m for key in ('amp', 'amplitude', 'mode_structure'))
                modes += max(0, int(k))
                counts[g['status'].asstr()[()]] += 1
                continuum_counts[g['continuum_status'].asstr()[()]] += 1
    return dict(validation_passed=True, inventory_cases=cases, scalar_cases=scalars,
                exported_modes=modes, scope_status_counts=dict(counts),
                continuum_status_counts=dict(continuum_counts))


def export(args):
    roots = [('live', args.data_root.resolve())]
    roots += [(f'snapshot_{i+1}', root.resolve()) for i, root in enumerate(args.snapshot_root)]
    reconstruction = ReconstructionRegistry(args.reconstruction_manifest) if getattr(args,'reconstruction_manifest',None) else None
    confirmed_empty = ConfirmedEmptyRegistry(args.confirmed_empty_manifest) if getattr(args,'confirmed_empty_manifest',None) else None
    if confirmed_empty:
        confirmed_empty.verify_sources(args.data_root)
    inventory_path = REPO/'audits/main_dataset_shots/shot_status.csv'
    catalog_path = REPO/'catalogs/accepted_modes.csv'
    meta_path = catalog_path.with_suffix('.metadata.json')
    holds_path = REPO/'audits/main_dataset_shots/active_shot_holds.csv'
    invalid_path = REPO/'configs/known_invalid_inputs.csv'
    inventory = read_csv(inventory_path, ('shot', 'status'))
    catalog_rows = read_csv(catalog_path, ('path', 'shot', 'ntor', 'omega'))
    catalog = {row['path']: row for row in catalog_rows}
    if len(catalog) != len(catalog_rows):
        raise ValueError('Duplicate accepted catalog keys')
    catalog_meta = json.loads(meta_path.read_text())
    if sha256_file(catalog_path) != catalog_meta['catalog_sha256']:
        raise ValueError('Accepted catalog does not match its metadata hash')
    receipts = {row['shot']: row for row in catalog_meta['source_shots']}
    holds = read_csv(holds_path)
    registry = load_input_validity_registry(invalid_path)
    provenance = {str(p.relative_to(REPO)): sha256_file(p) for p in
                  (inventory_path, catalog_path, meta_path, holds_path, invalid_path,
                   Path(__file__), REPO/'src/nova_equilibrium.py', REPO/'src/cont_features.py',
                   REPO/'src/mode_features.py', REPO/'src/nova_mode_loader.py', REPO/'src/nova_surrogate_provenance.py')}
    destinations = [args.output, args.output.with_suffix('.summary.json'), args.output.with_suffix('.scopes.csv')]
    if any(p.exists() for p in destinations):
        raise ValueError('Output or companion file already exists; choose a new output filename')
    if any(args.output.resolve().is_relative_to(root) for _, root in roots) or args.output.resolve().is_relative_to(args.sort_root.resolve()):
        raise ValueError('Output must be outside the input and sort-output trees')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.nova_surrogate_', suffix='.h5', dir=args.output.parent)
    os.close(fd)
    reports = []
    try:
        with h5py.File(temp, 'w') as h:
            h.attrs.update(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                           frequency_convention='Native NOVA omega/omega_ref; omega_ref=V_A_ref/(q_lcfs R_v)',
                           density_convention='NOVA iden=4 reconstructed from TRANSP ion species; rho excludes mu0',
                           selection_convention='Saved selected GOOD catalog; includes its two mixed-family entries; review status retained',
                           case_split_advice='Split by physical discharge, grouping all equilibrium variants/time slices',
                           pairing_scope='Exact saved pairs or explicitly audited historical continuum reconstructions; per-scope provenance distinguishes them',
                           continuum_validity_scope='Finite ordered samples after existing tail treatment; not a harmonic-convergence certificate')
            put(h, 'parameter_names', np.array(list(PARAMETERS), dtype=STR))
            put(h, 'case_ids', np.array([x['shot'] for x in inventory], dtype=STR))
            prov = h.create_group('provenance')
            put(prov, 'inputs_json', json.dumps(provenance, sort_keys=True))
            source_roots = {label:str(root) for label,root in roots}
            if reconstruction:
                source_roots.update({'reconstructed:'+label:str(root) for label,root in roots})
                source_roots.update(reconstruction.continuum_roots)
            put(prov, 'source_roots_json', json.dumps(source_roots))
            put(prov, 'sort_root', str(args.sort_root.resolve()))
            if reconstruction:
                put(prov, 'reconstruction_registry_json', json.dumps(reconstruction.data, sort_keys=True))
                put(prov, 'reconstruction_registry_sha256', reconstruction.sha256)
            if confirmed_empty:
                confirmed_empty.write_provenance(h)
            equilibria = h.create_group('equilibria')
            for index, item in enumerate(inventory):
                shot = item['shot']
                if '/' in shot or shot in ('.', '..'):
                    raise ValueError(f'Invalid case ID {shot}')
                case = equilibria.create_group(shot)
                put(case, 'equilibrium_id', shot)
                put(case, 'inventory_status', item['status'])
                source_dir = args.data_root/shot
                param_hashes = {name: sha256_file(source_dir/name) for name in
                                ('mapdsk.cdf', 'transp.dat', 'mpout1', 'qpvspsi.dat') if (source_dir/name).is_file()}
                if not param_hashes:
                    params, profiles, metadata = {}, {}, {}
                    eq_ok = False
                else:
                    params, profiles, metadata = load_equilibrium(source_dir)
                    eq_ok = True
                    if any(sha256_file(source_dir/name) != digest for name, digest in param_hashes.items()):
                        raise ValueError(f'Equilibrium inputs changed while reading {shot}')
                put(case, 'parameters_valid', eq_ok)
                pg = case.create_group('parameters')
                for key, (unit, definition) in PARAMETERS.items():
                    put(pg, key, params.get(key, np.nan), unit, definition)
                mg = case.create_group('metadata')
                for key, val in metadata.items():
                    put(mg, key, val)
                put(mg, 'equilibrium_input_sha256_json', json.dumps(param_hashes, sort_keys=True))
                profiles_group = case.create_group('equilibrium_profiles')
                for key, value in profiles.items():
                    put(profiles_group, key, value, 'Pa' if key == 'p_Pa' else '1')
                rows, summaries, anchors = [], {}, defaultdict(list)
                source_hashes = {}
                if shot in receipts:
                    directory = args.sort_root/shot
                    for name, expected in receipts[shot]['sha256'].items():
                        if sha256_file(directory/name) != expected:
                            raise ValueError(f'Saved catalog receipt differs: {directory/name}')
                    rows = read_csv(directory/'good_tae_final.csv')
                    summary_path = directory/'shot_summary_by_n.csv'
                    source_hashes['shot_summary_by_n.csv'] = sha256_file(summary_path)
                    summaries = {int(row['n']): row for row in read_csv(summary_path)}
                    empty_n = {n for n, row in summaries.items() if int(row['n_final_good']) == 0}
                    if empty_n:
                        all_path = directory/'all_modes_rules.csv'
                        source_hashes['all_modes_rules.csv'] = sha256_file(all_path)
                        anchors = zero_selection_anchors(all_path, empty_n)
                    source_hashes.update(receipts[shot]['sha256'])
                if {row['mode_key'] for row in rows} != {row['path'] for row in catalog_rows if row['shot'] == shot}:
                    raise ValueError(f'{shot}: installed selection differs from accepted catalog')
                for row in rows:
                    if row['final_decision'] != 'GOOD' or row['selected_final'].lower() != 'true':
                        raise ValueError(f'Not selected GOOD: {row["mode_key"]}')
                put(mg, 'selection_input_sha256_json', json.dumps(source_hashes, sort_keys=True))
                for n in range(1, 11):
                    report = write_scope(case.create_group(f'n_{n:02}'), ntor=n,
                        nr=len(profiles.get('r', [])), rows=[row for row in rows if int(row['ntor']) == n],
                        catalog=catalog, summaries=summaries, roots=roots,
                        hold=scope_holds(shot, n, registry, holds), equilibrium_ok=eq_ok, anchors=anchors,
                        reconstruction=reconstruction, confirmed_empty=confirmed_empty)
                    reports.append(dict(shot=shot, n_tor=n, **report))
                if any(sha256_file(args.sort_root/shot/name) != digest for name, digest in source_hashes.items()):
                    raise ValueError(f'Selection sources changed while reading {shot}')
                if (index+1) % 10 == 0:
                    print(f'Exported {index+1}/{len(inventory)} cases', flush=True)
        result = validate_export(temp)
        if reconstruction:
            reconstruction.verify_sources()
        if confirmed_empty:
            confirmed_empty.verify_sources(args.data_root)
        if any(sha256_file(REPO/name) != digest for name, digest in provenance.items()):
            raise ValueError('Repository input changed during export')
        # Link is atomic and fails if another process created the requested output.
        os.link(temp, args.output)
        result.update(schema=SCHEMA, output=str(args.output), sha256=sha256_file(args.output),
                      catalog_modes=len(catalog_rows), byte_size=args.output.stat().st_size)
        from tae_rule_io import write_dict_csv
        write_dict_csv(destinations[2], list(reports[0]), reports)
        destinations[1].write_text(json.dumps(result, indent=2)+'\n')
    finally:
        Path(temp).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=os.environ.get('NOVA_DITW_ROOT'),
                        help='Live case root (default NOVA_DITW_ROOT); read only')
    parser.add_argument('--snapshot-root', type=Path, action='append', default=[],
                        help='Optional archived case root for exact saved input pairs; may repeat')
    parser.add_argument('--sort-root', type=Path, default=REPO.parent/'sort_outputs',
                        help='Installed per-case selections matching accepted catalog receipts')
    parser.add_argument('--output', type=Path, required=True, help='New .h5 file; writes summary.json and scopes.csv companions')
    parser.add_argument('--reconstruction-manifest', type=Path,
                        help='Audited reconstruction registry: explicit scope hashes, mode identities, and historical evidence')
    parser.add_argument('--confirmed-empty-manifest', type=Path,
                        help='User-confirmed case/n zeros after NOVA internal filtering; named empty directories and continuum hashes are checked')
    args = parser.parse_args()
    if args.data_root is None:
        parser.error('Supply --data-root or set NOVA_DITW_ROOT')
    if args.output.suffix != '.h5':
        parser.error('--output must end in .h5')
    try:
        result = export(args)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
