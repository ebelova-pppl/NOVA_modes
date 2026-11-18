"""Publication must preserve independent backups and reject changed live data."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
spec = importlib.util.spec_from_file_location('publisher', REPO/'scripts/publish_nova_continua.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.fixture
def prepared(tmp_path):
    data, run, archive = (tmp_path/name for name in ['live', 'reconstructed', 'archive'])
    target = data/'case1/N1/datcon1'
    source = run/'cases/case1/N1/datcon1'
    target.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    target.write_bytes(b'old continuum\n')
    target.chmod(0o644)
    (target.parent/'datcon').symlink_to('datcon1')
    source.write_bytes(b'reconstructed continuum\n')
    equilibrium = data/'case1/mapdsk'; equilibrium.write_bytes(b'original equilibrium')
    task = dict(key='case1/N1',fingerprint='task-1',settings=dict(ntor=1),
                inputs={'mapdsk':dict(source=str(equilibrium),sha256=publisher.sha256(equilibrium))})
    publisher.write_json(run/'batch_manifest.json',dict(tasks=[task]))
    publisher.write_json(source.parent/'receipt.json',dict(status='success',fingerprint='task-1',
                         outputs={'datcon1':publisher.sha256(source)}))
    publisher.write_json(run/'validation_summary.json',dict(completed_tasks=1,
        centroid_modes_unavailable=0,centroid_modes_above_1e_6=0,
        crossing_points_unavailable=0,crossing_points_above_1e_6=0))
    plan = publisher.prepare(run, data, archive)
    return archive, plan, target, source


def test_backups_replacement_alias_and_idempotence(prepared):
    archive, plan, target, source = prepared
    assert target.read_bytes() == b'old continuum\n'
    result = publisher.apply(archive)
    assert result['installed'] == 1
    assert target.read_bytes() == source.read_bytes()
    assert (target.parent/'datcon').is_symlink()
    assert (target.parent/'datcon').read_bytes() == source.read_bytes()
    row = plan['files'][0]
    backups = [Path(row[k]) for k in ['project_backup', 'adjacent_backup']]
    for backup in backups:
        assert backup.read_bytes() == b'old continuum\n'
        assert backup.stat().st_ino != target.stat().st_ino
    assert backups[0].stat().st_ino != backups[1].stat().st_ino
    assert publisher.apply(archive)['already_installed'] == 1


def test_changed_live_file_prevents_publication(prepared):
    archive, plan, target, source = prepared
    target.write_bytes(b'new external calculation\n')
    with pytest.raises(ValueError, match='Unexpected live content'):
        publisher.apply(archive)
    assert target.read_bytes() == b'new external calculation\n'
    assert Path(plan['files'][0]['project_backup']).read_bytes() == b'old continuum\n'


def test_changed_input_prevents_publication(prepared):
    archive, plan, target, source = prepared
    Path(next(iter(plan['inputs']))).write_bytes(b'changed equilibrium')
    with pytest.raises(ValueError, match='Equilibrium'):
        publisher.apply(archive)
    assert target.read_bytes() == b'old continuum\n'


def test_bad_adjacent_backup_prevents_replacement(prepared):
    archive, plan, target, source = prepared
    backup = Path(plan['files'][0]['adjacent_backup'])
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b'wrong backup')
    with pytest.raises(ValueError, match='Existing backup differs'):
        publisher.apply(archive)
    assert target.read_bytes() == b'old continuum\n'


def test_interrupted_replace_keeps_backups_and_can_resume(prepared):
    archive, plan, target, source = prepared
    with patch.object(publisher.os, 'replace', side_effect=OSError('interrupted')):
        with pytest.raises(OSError, match='interrupted'):
            publisher.apply(archive)
    assert target.read_bytes() == b'old continuum\n'
    assert Path(plan['files'][0]['adjacent_backup']).read_bytes() == b'old continuum\n'
    assert publisher.apply(archive)['installed'] == 1


def test_previous_owner_retains_effective_write_access():
    old_uid = os.geteuid()+1
    original = 'user::rw-\nuser:12345:rw-\ngroup::r--\nmask::rw-\nother::r--\n'
    replacement = publisher.replacement_acl(original, old_uid)
    assert f'user:{old_uid}:rw-' in replacement
    assert 'group::r--' in replacement
    with pytest.raises(ValueError, match='ACL mask'):
        publisher.replacement_acl(original.replace('mask::rw-', 'mask::r-x'), old_uid)
