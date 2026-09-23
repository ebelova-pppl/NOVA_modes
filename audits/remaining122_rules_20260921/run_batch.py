#!/usr/bin/env python3
"""Run, verify and install the explicitly authorized 122-shot rules batch.

Example: python audits/remaining122_rules_20260921/run_batch.py stage \
  --data-root /path/to/DiTw --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_remaining122_rules_20260921
Use the same arguments with publish after successful staging.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
from tae_rule_io import input_fingerprint, datcon_path_for_mode, sha256_file
from tae_rule_config import load_rule_run_configuration

CONFIG = 'tae_rules_production_v13'
SELECTION = REPO / 'audits/runtime_estimate_20260921/remaining_shots.csv'
INVENTORY = REPO / 'audits/main_dataset_shots/shot_status.csv'
COMPACT = ['path', 'label', 'shot', 'ntor', 'nr', 'nhar', 'omega', 'gamma_d',
           'rad_loc', 'rad_width', 'gap_region', 'rule_decision',
           'rule_primary_reason', 'final_decision', 'decision_source',
           'overall_rule_severity', 'nearest_gate', 'selected_final',
           'input_fingerprint', 'review_status']


def read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write(path, rows, fields=None):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sources():
    paths = set((REPO / 'scripts').glob('*.py')) | set((REPO / 'src').rglob('*.py'))
    paths |= set((REPO / 'configs/rules').glob('*.yaml'))
    paths |= {Path(__file__).resolve(), SELECTION, INVENTORY,
              REPO / 'configs/known_invalid_inputs.csv',
              REPO / 'training_labels/tae_like_train.csv',
              REPO / 'audits/processed40_20260915/accepted_tae_modes.csv',
              REPO / 'audits/processed40_20260915/shot_summary.csv'}
    return {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(paths)}


def tree(path):
    return {str(p.relative_to(path)): sha256_file(p)
            for p in sorted(path.rglob('*')) if p.is_file()}


def files_for(shot_dir):
    files = []
    for n in range(1, 11):
        directory = shot_dir / f'N{n}'
        modes = sorted(p for p in directory.glob('egn*') if p.is_file())
        if modes and not (directory / f'datcon{n}').is_file():
            raise ValueError(f'Missing datcon{n} in {directory}')
        files.extend(modes)
    if not files:
        raise ValueError(f'No mode inputs: {shot_dir}')
    return files


def fingerprints(shot_dir):
    return {f'{shot_dir.name}/{p.parent.name}/{p.name}':
            input_fingerprint(p, datcon_path_for_mode(p)) for p in files_for(shot_dir)}


def check_output(directory, before, shot_dir, config_hash):
    rows = read(directory / 'all_modes_rules.csv')
    by_key = {r['mode_key']: r for r in rows}
    assert len(rows) == len(by_key) == len(before), 'Input/output coverage differs'
    assert {k: r['input_fingerprint'] for k, r in by_key.items()} == before
    assert fingerprints(shot_dir) == before, 'Raw inputs changed during processing'
    summary = dict(csv.reader((directory / 'shot_summary.csv').open()))
    assert summary['rule_configuration_sha256'] == config_hash
    assert summary['continuum_preprocessing_version'] == 'datcon-monotonic-tail-v1'
    assert summary['rule_survivor_policy'] == 'accept-as-good-v1'
    assert summary['n_invalid'] == '0', summary['primary_reason_counts_json']
    assert summary['n_final_review'] == '0'
    assert summary['n_manual_override_rows'] == summary['n_overrides_applied'] == '0'
    assert summary['n_severity_unavailable'] == '0'
    assert summary['duplicate_processing_status'] in (
        'COMPLETED_RULE_SEVERITY', 'SKIPPED_NO_GOOD_MODES', 'NO_CLOSE_FREQUENCY_CLUSTERS')
    assert not read(directory / 'resolution_warnings.csv'), 'Uncalibrated radial grid'
    for r in rows:
        assert r['nr'] == '201', f"Unexpected grid: {r['mode_key']} nr={r['nr']}"
        assert not r['diagnostic_message'], r['diagnostic_message']
        if r['processing_status'] == 'ROUTED_EAE':
            assert r['gap_region'] == 'eae_like'
        else:
            assert r['processing_status'] == 'RULE_EVALUATED'
            assert r['severity_complete'] == 'True'
            assert r['rule_decision'] in ('BAD', 'REVIEW')
            assert r['final_decision'] == ('BAD' if r['rule_decision'] == 'BAD' else 'GOOD')
    selections = {
        'good_tae_final.csv': lambda r: r['selected_final'] == 'True',
        'good_tae_unchecked.csv': lambda r: r['final_decision'] == 'GOOD',
        'bad_tae_like.csv': lambda r: r['final_decision'] == 'BAD',
        'eae_like.csv': lambda r: r['processing_status'] == 'ROUTED_EAE',
        'rejected_modes.csv': lambda r: r['final_decision'] == 'INVALID',
    }
    for name, select in selections.items():
        exported = read(directory / name)
        expected = [r for r in rows if select(r)]
        assert {r['mode_key'] for r in exported} == {r['mode_key'] for r in expected}, name
        assert len(exported) == len(expected), name
        assert all(r == by_key[r['mode_key']] for r in exported), name
    assert int(summary['n_total_files']) == len(rows)
    for column, predicate in [('n_final_good', selections['good_tae_final.csv']),
                              ('n_final_bad', selections['bad_tae_like.csv']),
                              ('n_eae_like', selections['eae_like.csv'])]:
        assert int(summary[column]) == sum(predicate(r) for r in rows), column
    return summary


def run_one(args, item, config_hash):
    shot = item['shot']
    began = time.monotonic()
    shot_dir = args.data_root / shot
    output = args.runtime_dir / 'rules' / shot
    assert not output.exists(), f'Output already exists: {output}'
    before = fingerprints(shot_dir)
    save(args.runtime_dir / 'inputs' / f'{shot}.json', before)
    command = [sys.executable, str(REPO / 'scripts/sort_shot_mixed.py'),
               '--method', 'rules', '--rule_config', CONFIG,
               '--shot_dir', str(shot_dir), '--out_dir', str(output)]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (args.runtime_dir / 'logs' / f'{shot}.log').open('w') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = check_output(output, before, shot_dir, config_hash)
    result = {'shot': shot, 'status': 'verified', 'input_modes': len(before),
              'estimated_input_modes': int(item['current_mode_files']),
              'tae_like': int(summary['n_tae_like']), 'mixed': int(summary['n_mixed']),
              'eae_like': int(summary['n_eae_like']), 'invalid': int(summary['n_invalid']),
              'good_before_dedup': int(summary['n_final_good_before_clustering']),
              'selected_good': int(summary['n_final_good']), 'bad': int(summary['n_final_bad']),
              'elapsed_seconds': round(time.monotonic() - began, 2),
              'output_sha256': tree(output)}
    save(args.runtime_dir / 'verified' / f'{shot}.json', result)
    return result


def stage(args):
    selected = [r for r in read(SELECTION) if r['candidate_after_primary_holds'] == 'True'
                and r['secondary_n1_hold'] == 'False']
    assert len(selected) == 122 and sum(int(r['current_mode_files']) for r in selected) == 71656
    inv = {r['shot']: r for r in read(INVENTORY)}
    for r in selected:
        assert inv[r['shot']]['post_training_checked'] == inv[r['shot']]['active_training_shot'] == 'no'
        assert not (args.rules_root / r['shot']).exists(), f"Destination exists: {r['shot']}"
    assert not args.runtime_dir.exists()
    args.runtime_dir.mkdir(parents=True)
    for name in ('rules', 'logs', 'inputs', 'verified'):
        (args.runtime_dir / name).mkdir()
    configuration = load_rule_run_configuration(CONFIG)
    snapshot = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'source_sha256': sources(), 'configuration_sha256': configuration.sha256,
                'data_root': str(args.data_root), 'rules_root': str(args.rules_root),
                'runtime_dir': str(args.runtime_dir), 'workers': args.workers,
                'shots': [r['shot'] for r in selected]}
    save(HERE / 'run_inputs.json', snapshot)
    write(HERE / 'selection.csv', selected)
    results, failures = [], []
    # Largest shots first reduces the tail after most workers finish.
    selected.sort(key=lambda r: int(r['current_mode_files']), reverse=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_one, args, r, configuration.sha256): r['shot'] for r in selected}
        for future in as_completed(pending):
            shot = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(f"{len(results)}/{len(selected)} verified: {shot}: "
                      f"{result['selected_good']} GOOD, {result['bad']} BAD "
                      f"({result['elapsed_seconds']:.0f}s including verification)", flush=True)
            except Exception as exc:
                failure = {'shot': shot, 'error': f'{type(exc).__name__}: {exc}'}
                failures.append(failure)
                print(f'FAILED: {failure}', flush=True)
            save(HERE / 'progress.json', {'verified': len(results), 'failed': failures,
                                         'total': len(selected), 'updated_utc': datetime.now(timezone.utc).isoformat()})
    assert sources() == snapshot['source_sha256'], 'Source/configuration changed during batch'
    save(HERE / 'stage_results.json', {'verified': results, 'failures': failures})
    assert not failures, f'{len(failures)} shots require investigation; see stage_results.json'
    write_verified_batch(args, results, configuration.sha256)
    print('All 122 shots verified; ready for installation.', flush=True)


def write_verified_batch(args, results, config_hash):
    """Write verified selections, recording any explicitly documented input holds."""
    holds = json.loads((HERE / 'input_holds.json').read_text()) if (HERE / 'input_holds.json').exists() else []
    assert len(results) + len(holds) == 122
    assert {r['shot'] for r in results}.isdisjoint(r['shot'] for r in holds)
    write(HERE / 'shot_summary.csv', [{k: v for k, v in r.items() if k != 'output_sha256'}
                                    for r in sorted(results, key=lambda r: r['shot'])])
    good = []
    for r in sorted(results, key=lambda r: r['shot']):
        for mode in read(args.runtime_dir / 'rules' / r['shot'] / 'good_tae_final.csv'):
            compact = {k: mode.get(k, '') for k in COMPACT}
            compact.update(path=mode['mode_key'], label='good', review_status='not_visually_reviewed')
            good.append(compact)
    write(HERE / 'good_tae_final_batch.csv', good, COMPACT)
    artifacts = ['selection.csv', 'shot_summary.csv', 'good_tae_final_batch.csv', 'stage_results.json']
    if holds:
        artifacts += ['input_holds.json', 'invalid_input_files.csv']
    save(HERE / 'verification.json', {'status': 'verified', 'shots': len(results),
         'attempted_shots': 122, 'held_input_shots': [r['shot'] for r in holds],
         'counts': {k: sum(r[k] for r in results) for k in
                    ('input_modes', 'tae_like', 'mixed', 'eae_like', 'invalid', 'good_before_dedup', 'selected_good', 'bad')},
         'all_nr': 201, 'all_raw_fingerprints_verified': True,
         'configuration_sha256': config_hash,
         'output_sha256': {n: sha256_file(HERE / n) for n in artifacts}})


def publish(args):
    snapshot = json.loads((HERE / 'run_inputs.json').read_text())
    receipt = json.loads((HERE / 'verification.json').read_text())
    assert receipt['status'] == 'verified'
    assert sources() == snapshot['source_sha256'], 'Source/configuration changed'
    assert str(args.rules_root) == snapshot['rules_root'] and str(args.runtime_dir) == snapshot['runtime_dir']
    for name, digest in receipt['output_sha256'].items():
        assert sha256_file(HERE / name) == digest
    results = json.loads((HERE / 'stage_results.json').read_text())['verified']
    assert len(results) == receipt['shots']
    assert {r['shot'] for r in results} == set(snapshot['shots']) - set(receipt['held_input_shots'])
    installed = []
    for result in sorted(results, key=lambda r: r['shot']):
        shot = result['shot']
        source = args.runtime_dir / 'rules' / shot
        target = args.rules_root / shot
        temporary = args.rules_root / f'.staging_remaining122_20260921_{shot}'
        assert not target.exists() and not temporary.exists(), f'Destination exists: {shot}'
        assert tree(source) == result['output_sha256'], shot
        before = json.loads((args.runtime_dir / 'inputs' / f'{shot}.json').read_text())
        assert fingerprints(args.data_root / shot) == before, f'Inputs changed before publication: {shot}'
        shutil.copytree(source, temporary)
        assert tree(temporary) == result['output_sha256'], shot
        temporary.rename(target)
        installed.append(shot)
        save(HERE / 'publication.json', {'status': 'installing', 'installed': installed,
                                       'rules_root': str(args.rules_root)})
        print(f'{len(installed)}/{len(results)} installed: {shot}', flush=True)
    save(HERE / 'publication.json', {'status': 'installed', 'installed': installed,
         'rules_root': str(args.rules_root), 'completed_utc': datetime.now(timezone.utc).isoformat(),
         'existing_outputs_replaced': 0, 'held_input_shots': receipt['held_input_shots']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('stage', 'publish'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--rules-root', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for name in ('data_root', 'rules_root', 'runtime_dir'):
        setattr(args, name, getattr(args, name).resolve())
    if args.workers < 1:
        parser.error('--workers must be positive')
    globals()[args.phase](args)
