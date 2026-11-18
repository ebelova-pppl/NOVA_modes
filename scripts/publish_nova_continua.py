#!/usr/bin/env python3
"""Back up and publish a validated NOVA reconstruction in two explicit steps.

Example (run from the repository root, after publication is authorized):
  python scripts/publish_nova_continua.py prepare --run-root /path/to/restored \
    --data-root /path/to/DiTw --archive-root /path/to/new/publication
  python scripts/publish_nova_continua.py apply --archive-root /path/to/publication

Prepare writes independent project backups and a frozen plan. Apply creates
and verifies every adjacent backup before atomically replacing any datconN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

from _repo_bootstrap import ensure_repo_src_on_path
ensure_repo_src_on_path()
from nova_continuum_run import sha256, utcnow, write_json


def metadata(path):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise ValueError(f'Expected a regular file with one hard link: {path}')
    return dict(device=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                uid=st.st_uid, gid=st.st_gid, mode=stat.S_IMODE(st.st_mode))


def acl(path):
    return subprocess.check_output(['getfacl', '-c', '-n', str(path)], text=True,
                                   stderr=subprocess.PIPE).strip() + '\n'


def replacement_acl(original, owner_uid):
    """Keep prior permissions when atomic replacement changes file ownership."""
    lines = original.strip().splitlines()
    if owner_uid != os.geteuid():
        owner = next(s.split(':')[2] for s in lines if s.startswith('user::'))
        mask = next((s.split(':')[2] for s in lines if s.startswith('mask::')), owner)
        if not (set(owner)-{'-'}) <= (set(mask)-{'-'}):
            raise ValueError('ACL mask would restrict the previous owner; review permissions first')
        lines = [s for s in lines if not s.startswith(f'user:{owner_uid}:')]
        lines.append(f'user:{owner_uid}:{owner}')
    return '\n'.join(lines) + '\n'


def copy_backup(source, dest, expected):
    """An independent, read-only copy; existing backups must match exactly."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        raise ValueError(f'Refusing backup symlink: {dest}')
    if dest.exists():
        if sha256(dest) != expected:
            raise ValueError(f'Existing backup differs: {dest}')
        return
    with source.open('rb') as src, dest.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    if sha256(dest) != expected:
        raise ValueError(f'Backup verification failed: {dest}')
    dest.chmod(0o444)


def verify_source_inputs(plan):
    for row in plan['inputs'].values():
        if sha256(row['source']) != row['sha256']:
            raise ValueError(f'Equilibrium, TRANSP input, or executable changed: {row["source"]}')


def prepare(run_root, data_root, archive_root):
    run, data, archive = (Path(p).resolve() for p in [run_root, data_root, archive_root])
    if archive.is_relative_to(data) or data.is_relative_to(archive):
        raise ValueError('Project archive must be outside the live data tree')
    if run.is_relative_to(data) or data.is_relative_to(run):
        raise ValueError('Reconstruction must be separate from the live data tree')
    if archive.exists():
        raise ValueError(f'Use a fresh archive directory: {archive}')
    batch = json.loads((run/'batch_manifest.json').read_text())
    validation = json.loads((run/'validation_summary.json').read_text())
    if validation['completed_tasks'] != len(batch['tasks']):
        raise ValueError('Historical validation does not cover this batch')
    for key in ['centroid_modes_unavailable', 'centroid_modes_above_1e_6',
                'crossing_points_unavailable', 'crossing_points_above_1e_6']:
        if validation[key]:
            raise ValueError(f'Unresolved historical comparison: {key}')
    plan = dict(schema='nova-continuum-publication-v1', created_utc=utcnow(),
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                run_root=str(run), data_root=str(data), archive_root=str(archive),
                batch_manifest_sha256=sha256(run/'batch_manifest.json'),
                validation_sha256=sha256(run/'validation_summary.json'),
                publisher_sha256=sha256(__file__), inputs={}, files=[])
    seen = set()
    for task in batch['tasks']:
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*/N[1-9][0-9]*', task['key']):
            raise ValueError(f'Unsafe scope: {task["key"]}')
        relative = f'{task["key"]}/datcon{task["settings"]["ntor"]}'
        if relative in seen:
            raise ValueError(f'Duplicate target: {relative}')
        seen.add(relative)
        source, target = run/'cases'/relative, data/relative
        if source.resolve() != source or target.resolve() != target:
            raise ValueError(f'Refusing redirected source/target: {relative}')
        receipt = json.loads((source.parent/'receipt.json').read_text())
        if receipt['status'] != 'success' or receipt['fingerprint'] != task['fingerprint']:
            raise ValueError(f'Invalid reconstruction receipt: {relative}')
        expected = receipt['outputs'][source.name]
        if sha256(source) != expected:
            raise ValueError(f'Reconstruction changed: {source}')
        original = metadata(target)
        original_acl = acl(target)
        original_hash = sha256(target)
        if metadata(target) != original:
            raise ValueError(f'Live file changed while reading: {target}')
        if not os.access(target.parent, os.W_OK | os.X_OK):
            raise PermissionError(f'Target directory is not writable: {target.parent}')
        alias = target.parent/'datcon'
        if not alias.is_symlink() or alias.resolve() != target:
            raise ValueError(f'Unexpected unnumbered datcon alias: {alias}')
        plan['files'].append(dict(relative=relative, source=str(source), target=str(target),
            old_sha256=original_hash, new_sha256=expected, original=original,
            original_acl=original_acl, replacement_acl=replacement_acl(original_acl, original['uid']),
            alias=os.readlink(alias), project_backup=str(archive/'before'/relative),
            adjacent_backup=str(target.parent/'continuum_backups'/plan['stamp']/target.name),
            settings=task['settings'], task_fingerprint=task['fingerprint']))
        for item in task['inputs'].values():
            plan['inputs'][item['source']] = item
    verify_source_inputs(plan)
    archive.mkdir(parents=True)
    for row in plan['files']:
        copy_backup(Path(row['target']), Path(row['project_backup']), row['old_sha256'])
        if metadata(Path(row['target'])) != row['original']:
            raise ValueError(f'Live file changed during backup: {row["target"]}')
    write_json(archive/'plan.json', plan)
    print(f'Prepared and verified {len(plan["files"])} project backups; live files unchanged.', flush=True)
    return plan


def check_live(row):
    target = Path(row['target'])
    if target.resolve() != target:
        raise ValueError(f'Redirected target: {target}')
    digest = sha256(target)
    if digest not in (row['old_sha256'], row['new_sha256']):
        raise ValueError(f'Unexpected live content; refusing overwrite: {target}')
    if digest == row['old_sha256'] and metadata(target) != row['original']:
        raise ValueError(f'Live metadata changed since preparation: {target}')
    alias = target.parent/'datcon'
    if not alias.is_symlink() or os.readlink(alias) != row['alias'] or alias.resolve() != target:
        raise ValueError(f'Live datcon alias changed: {alias}')
    return digest


def install_one(row, stamp):
    if check_live(row) == row['new_sha256']:
        return 'already_installed'
    target = Path(row['target'])
    descriptor, name = tempfile.mkstemp(prefix=f'.{target.name}.restore_{stamp}.', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as dst, Path(row['source']).open('rb') as src:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256(temporary) != row['new_sha256']:
            raise ValueError(f'Staged replacement changed: {temporary}')
        if temporary.stat().st_gid != row['original']['gid']:
            os.chown(temporary, -1, row['original']['gid'])
        temporary.chmod(row['original']['mode'])
        subprocess.run(['setfacl', '--set-file=-', str(temporary)],
                       input=row['replacement_acl'], text=True, check=True, capture_output=True)
        if check_live(row) != row['old_sha256']:
            raise ValueError(f'Target changed during staging: {target}')
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if sha256(target) != row['new_sha256']:
            raise ValueError(f'Published file hash mismatch: {target}')
        return 'installed'
    finally:
        temporary.unlink(missing_ok=True)


def apply(archive_root):
    archive = Path(archive_root).resolve()
    plan = json.loads((archive/'plan.json').read_text())
    if plan['archive_root'] != str(archive) or plan['publisher_sha256'] != sha256(__file__):
        raise ValueError('Publication archive or publisher code changed')
    run = Path(plan['run_root'])
    if (sha256(run/'batch_manifest.json') != plan['batch_manifest_sha256']
            or sha256(run/'validation_summary.json') != plan['validation_sha256']):
        raise ValueError('Reconstruction manifest or validation changed')
    verify_source_inputs(plan)
    # Verify the complete batch before beginning any publication.
    for row in plan['files']:
        check_live(row)
        if sha256(Path(row['source'])) != row['new_sha256']:
            raise ValueError(f'Reconstruction changed: {row["source"]}')
        if sha256(Path(row['project_backup'])) != row['old_sha256']:
            raise ValueError(f'Project backup changed: {row["project_backup"]}')
    for row in plan['files']:
        backup = Path(row['adjacent_backup'])
        if backup.resolve() != backup:
            raise ValueError(f'Redirected adjacent backup: {backup}')
        copy_backup(Path(row['project_backup']), backup, row['old_sha256'])
    print(f'All {len(plan["files"])} adjacent backups verified; publishing.', flush=True)
    records = []
    with (archive/'events.jsonl').open('a') as log:
        for row in plan['files']:
            status = install_one(row, plan['stamp'])
            record = dict(relative=row['relative'], status=status, utc=utcnow(),
                          sha256=sha256(Path(row['target'])))
            records.append(record)
            log.write(json.dumps(record)+'\n'); log.flush(); os.fsync(log.fileno())
            if len(records) % 100 == 0:
                print(f'Published {len(records)}/{len(plan["files"])}', flush=True)
    for row in plan['files']:
        if check_live(row) != row['new_sha256']:
            raise ValueError(f'Final verification failed: {row["target"]}')
        for key in ['project_backup', 'adjacent_backup']:
            if sha256(Path(row[key])) != row['old_sha256']:
                raise ValueError(f'Final backup verification failed: {row[key]}')
    summary = dict(status='complete', finished_utc=utcnow(), files=len(records),
        cases=len({r['relative'].split('/')[0] for r in records}),
        installed=sum(r['status']=='installed' for r in records),
        already_installed=sum(r['status']=='already_installed' for r in records),
        plan_sha256=sha256(archive/'plan.json'), records=records)
    write_json(archive/'publication_summary.json', summary)
    manifest = Path(plan['data_root'])/f'.continuum_restoration_{plan["stamp"]}.json'
    public_record = dict(plan=plan, publication=summary)
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing['plan'] != plan:
            raise ValueError(f'Existing publication manifest differs: {manifest}')
    else:
        with manifest.open('x') as stream:
            json.dump(public_record, stream, indent=2); stream.write('\n')
    print(f'Published and verified {len(records)} continua. Manifest: {manifest}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'apply'])
    parser.add_argument('--archive-root', required=True, type=Path, help='Fresh project backup/plan directory, or prepared archive for apply')
    parser.add_argument('--run-root', type=Path, help='Validated reconstruction batch (prepare only)')
    parser.add_argument('--data-root', type=Path, help='Live DiTw root (prepare only)')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.run_root is None or args.data_root is None:
            parser.error('prepare requires --run-root and --data-root')
        prepare(args.run_root, args.data_root, args.archive_root)
    else:
        apply(args.archive_root)


if __name__ == '__main__':
    main()
