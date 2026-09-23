#!/usr/bin/env python3
"""Record installed batch membership without claiming human visual review.

Run after run_batch.py publish: python audits/remaining122_rules_20260921/record_installation.py
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, fields, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    temporary.replace(path)


def main():
    publication = json.loads((HERE / 'publication.json').read_text())
    verification = json.loads((HERE / 'verification.json').read_text())
    snapshot = json.loads((HERE / 'run_inputs.json').read_text())
    assert publication['status'] == 'installed'
    holds = {r['shot']: r for r in json.loads((HERE / 'input_holds.json').read_text())}
    n_installed = verification['shots']
    assert n_installed + len(holds) == 122
    assert not (HERE / 'inventory_update.json').exists()
    selected = set(publication['installed'])
    assert selected == {r['shot'] for r in read(HERE / 'selection.csv')} - holds.keys()
    summaries = {r['shot']: r for r in read(HERE / 'shot_summary.csv')}
    for name, digest in verification['output_sha256'].items():
        assert sha(HERE / name) == digest, name
    previous = REPO / 'audits/processed40_20260915'
    for name in ('accepted_tae_modes.csv', 'shot_summary.csv'):
        path = previous / name
        assert sha(path) == snapshot['source_sha256'][str(path.relative_to(REPO))]
    inventory = REPO / 'audits/main_dataset_shots/shot_status.csv'
    assert sha(inventory) == snapshot['source_sha256'][str(inventory.relative_to(REPO))]
    updates = []
    for name in ('shot_status.csv', 'g_shot_status.csv'):
        path = inventory.with_name(name)
        before = path.read_bytes()
        reader = csv.DictReader(io.StringIO(before.decode()))
        fields = reader.fieldnames
        rows = list(reader)
        changes = []
        for r in rows:
            if r['shot'] in holds:
                old = dict(r)
                assert r['active_training_shot'] == r['post_training_checked'] == 'no'
                r['status'] = 'input_issue'
                hold = holds[r['shot']]
                bad_paths = ', '.join(v['mode_key'] for v in hold['invalid_files'])
                r['notes'] = (r['notes'].rstrip() + ' ' +
                    '2026-09-21: Rules batch found previously unflagged NaN metadata; '
                    f'whole shot held from publication: {bad_paths}. '
                    'Recalculation/input repair required; local diagnostic outputs retained. '
                    'See audits/remaining122_rules_20260921/input_holds.json.').strip()
                changes.append({'shot': r['shot'], 'before': old, 'after': dict(r)})
                continue
            if r['shot'] not in selected:
                continue
            old = dict(r)
            assert r['active_training_shot'] == 'no' and r['post_training_checked'] == 'no'
            s = summaries[r['shot']]
            r.update(post_training_checked='yes', checked_methods='rules',
                     status='sorted_rules_pending_review')
            note = (f"2026-09-21: Authorized remaining-122 production-v13 rules batch "
                    f"completed and installed; {s['input_modes']} inputs, nr=201, "
                    f"{s['good_before_dedup']} GOOD before deduplication, "
                    f"{s['selected_good']} selected GOOD, {s['bad']} BAD, "
                    f"{s['eae_like']} EAE-like, zero INVALID. Automated checks passed; "
                    "visual review pending. No RF-CNN comparison or new N1 alignment "
                    "clearance. See audits/remaining122_rules_20260921.")
            r['notes'] = (r['notes'].rstrip() + ' ' + note).strip()
            changes.append({'shot': r['shot'], 'before': old, 'after': dict(r)})
        assert len(changes) == (122 if name == 'shot_status.csv' else 2)
        if name == 'shot_status.csv':
            assert sum(r['post_training_checked'] == 'yes' for r in rows) == 40 + n_installed
            assert sum(r['active_training_shot'] == 'yes' for r in rows) == 14
            assert not any(r['active_training_shot'] == r['post_training_checked'] == 'yes' for r in rows)
        write(path, fields, rows)
        assert read(path) == rows
        updates.append({'path': str(path.relative_to(REPO)),
                        'before_sha256': hashlib.sha256(before).hexdigest(),
                        'after_sha256': sha(path), 'changes': changes})
    combined = read(previous / 'shot_summary.csv')
    for r in combined:
        r['review_status'] = 'prior_reviewed_cohort'
    fields = list(combined[0])
    for shot, s in sorted(summaries.items()):
        combined.append(dict(shot=shot, checked_methods='rules',
            rule_evaluated=str(int(s['tae_like']) + int(s['mixed'])),
            routed_eae=s['eae_like'], invalid=s['invalid'],
            final_good_before_dedup=s['good_before_dedup'], selected_good=s['selected_good'],
            final_bad=s['bad'], manual_overrides='0', review_status='not_visually_reviewed'))
    combined.sort(key=lambda r: r['shot'])
    assert len(combined) == len({r['shot'] for r in combined}) == 40 + n_installed
    combined_name = f'processed{len(combined)}_summary.csv'
    write(HERE / combined_name, fields, combined)
    totals = {k: sum(int(r[k]) for r in combined) for k in
              ('rule_evaluated', 'routed_eae', 'invalid', 'final_good_before_dedup',
               'selected_good', 'final_bad', 'manual_overrides')}
    result = {'status': 'complete', 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'new_rules_shots': n_installed, 'new_visually_reviewed_shots': 0,
              'new_input_holds': sorted(holds),
              'processed_post_training_shots': len(combined), 'active_training_shots': 14,
              'combined_disjoint_shots': len(combined) + 14,
              'remaining_held_or_empty_shots': 200 - len(combined) - 14,
              'combined_summary_csv': combined_name,
              'existing_reviewed_cohort': 40, 'rf_cnn_comparison_cohort': 39,
              'processed_totals': totals, 'inventories': updates}
    (HERE / 'inventory_update.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inventories'}, indent=2))


if __name__ == '__main__':
    main()
