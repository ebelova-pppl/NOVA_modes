#!/usr/bin/env python3
"""Finalize completed outputs after correcting the batch verifier's status list.

The initial verifier omitted the valid NO_CLOSE_FREQUENCY_CLUSTERS outcome.
This command rechecks affected outputs without rerunning or changing rules.
Run with the same --data-root, --rules-root and --runtime-dir as run_batch.py.
"""
import argparse
import json
from pathlib import Path

import run_batch as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'rules-root', 'runtime-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('data_root', 'rules_root', 'runtime_dir'):
        setattr(args, name, getattr(args, name).resolve())
    snapshot = json.loads((batch.HERE / 'run_inputs.json').read_text())
    initial = json.loads((batch.HERE / 'stage_results.json').read_text())
    assert len(initial['verified']) + len(initial['failures']) == 122
    holds = json.loads((batch.HERE / 'input_holds.json').read_text())
    held = {r['shot'] for r in holds}
    assert held <= {r['shot'] for r in initial['failures']}
    driver = str(Path(batch.__file__).resolve().relative_to(batch.REPO))
    current = batch.sources()
    expected = snapshot['source_sha256']
    assert {k: v for k, v in current.items() if k != driver} == {
        k: v for k, v in expected.items() if k != driver}, 'Scientific inputs or code changed'
    results = []
    recovered = []
    for item in batch.read(batch.HERE / 'selection.csv'):
        shot = item['shot']
        if shot in held:
            continue
        output = args.runtime_dir / 'rules' / shot
        marker = args.runtime_dir / 'verified' / f'{shot}.json'
        if marker.exists():
            result = json.loads(marker.read_text())
            assert batch.tree(output) == result['output_sha256'], shot
        else:
            before = json.loads((args.runtime_dir / 'inputs' / f'{shot}.json').read_text())
            s = batch.check_output(output, before, args.data_root / shot,
                                   snapshot['configuration_sha256'])
            result = {'shot': shot, 'status': 'verified', 'input_modes': len(before),
                      'estimated_input_modes': int(item['current_mode_files']),
                      'tae_like': int(s['n_tae_like']), 'mixed': int(s['n_mixed']),
                      'eae_like': int(s['n_eae_like']), 'invalid': int(s['n_invalid']),
                      'good_before_dedup': int(s['n_final_good_before_clustering']),
                      'selected_good': int(s['n_final_good']), 'bad': int(s['n_final_bad']),
                      'elapsed_seconds': '', 'output_sha256': batch.tree(output)}
            batch.save(marker, result)
            recovered.append(shot)
        results.append(result)
    batch.save(batch.HERE / 'initial_stage_results.json', initial)
    batch.save(batch.HERE / 'verifier_correction.json', {
        'reason': 'Accept valid NO_CLOSE_FREQUENCY_CLUSTERS; no scientific rule change.',
        'driver_before_sha256': expected[driver], 'driver_after_sha256': current[driver],
        'rechecked_shots': recovered,
        'note': 'Elapsed time is blank for recovered rows; sorting was not repeated.'})
    snapshot['initial_driver_sha256'] = expected[driver]
    snapshot['source_sha256'] = current
    batch.save(batch.HERE / 'run_inputs.json', snapshot)
    batch.save(batch.HERE / 'stage_results.json', {'verified': results, 'failures': [], 'input_holds': holds})
    batch.write_verified_batch(args, results, snapshot['configuration_sha256'])
    batch.save(batch.HERE / 'progress.json', {'verified': len(results), 'failed': [],
                                            'input_holds': sorted(held), 'total': 122})
    print(f'{len(results)} shots verified; {len(held)} input holds; corrected-verifier rechecks: {recovered}')


if __name__ == '__main__':
    main()
