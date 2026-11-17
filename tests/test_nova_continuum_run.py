"""Exercise isolation, recovery, and producer failures without running NOVA."""
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import nova_continuum_run as runner


@pytest.fixture
def batch(tmp_path):
    data = tmp_path/'data'
    case = data/'case1'
    case.mkdir(parents=True)
    for name in ['mapdsk', 'mpout1', 'transp.dat']:
        (case/name).write_text('original input\n')
    binary = case/'gocont_tr'
    binary.write_text('''#!/usr/bin/env python3
import re, sys
from pathlib import Path
commands = sys.stdin.read().split('/')[0]
commands = re.sub(r'ns=(\\d+) (\\d+) (\\d+)', r'NS=\\1,\\2,\\3', commands)
Path('outcon').write_text(commands + ' /\\n')
Path('datcon').write_text('3 4\\n1.0 2.0 3.0\\n1.1 2.1 3.1\\n')
''')
    binary.chmod(0o755)
    manifest = tmp_path/'tasks.csv'
    manifest.write_text('shot,ntor,lmin,lmax\ncase1,1,-3,108\n')
    out = tmp_path/'results'
    return manifest, data, out


def initialize(batch):
    manifest, data, out = batch
    tasks = runner.prepare_tasks(manifest, data, out)
    runner.initialize_batch(out, tasks)
    return tasks[0]


def test_success_and_verified_resume(batch):
    task = initialize(batch)
    before = {name: runner.sha256(row['source']) for name, row in task['inputs'].items()}
    result = runner.execute_task(task)
    assert result['status'] == 'success', result
    assert {name: runner.sha256(row['source']) for name, row in task['inputs'].items()} == before
    assert runner.execute_task(task, resume=True)['resumed']
    assert len(list((Path(task['workdir'])/'attempts').iterdir())) == 1
    (Path(task['workdir'])/'datcon1').write_text('corrupt')
    with pytest.raises(ValueError, match='completed output changed'):
        runner.execute_task(task, resume=True)


def test_source_change_refuses_batch_resume(batch):
    initialize(batch)
    manifest, data, out = batch
    (data/'case1'/'transp.dat').write_text('changed')
    with pytest.raises(ValueError, match='Resume refused'):
        runner.initialize_batch(out, runner.prepare_tasks(manifest, data, out), resume=True)


def test_isolation_and_duplicates(batch):
    manifest, data, out = batch
    with pytest.raises(ValueError, match='separate'):
        runner.prepare_tasks(manifest, data, data/'new')
    with manifest.open('a') as stream:
        stream.write('case1,1,-3,108\n')
    with pytest.raises(ValueError, match='Duplicate'):
        runner.prepare_tasks(manifest, data, out)


def test_failed_producer_can_retry_without_overwrite(batch):
    manifest, data, out = batch
    binary = data/'case1'/'gocont_tr'
    # Simulate a transient zero-exit Fortran STOP with missing output.
    flag = data/'transient_failure'
    flag.touch()
    binary.write_text(binary.read_text().replace('commands = sys.stdin',
        f'if Path({str(flag)!r}).exists(): sys.exit(0)\ncommands = sys.stdin'))
    task = initialize(batch)
    first = runner.execute_task(task)
    assert first['status'] == 'failed'
    flag.unlink()
    second = runner.execute_task(task, resume=True)
    assert second['status'] == 'success', second
    assert second['attempt'] == 'attempts/0002'
    assert (Path(task['workdir'])/'attempts/0001/receipt.json').is_file()


def test_interrupted_publication_recovers_validated_attempt(batch):
    task = initialize(batch)
    original = runner.os.link
    count = 0

    def interrupted(source, dest):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('interrupted publication')
        return original(source, dest)

    with patch.object(runner.os, 'link', interrupted):
        result = runner.run_worker(task, 10, False)
    assert result['status'] == 'failed'
    result = runner.execute_task(task, resume=True)
    assert result['status'] == 'success'
    assert result['attempt'] == 'attempts/0001'


def test_runtime_settings_must_match(batch):
    manifest, data, out = batch
    binary = data/'case1'/'gocont_tr'
    binary.write_text(binary.read_text().replace("commands + ' /", "commands.replace('ntor=1', 'ntor=9') + ' /"))
    task = initialize(batch)
    result = runner.execute_task(task)
    assert result['status'] == 'failed'
    assert 'settings mismatch' in result['error']
    assert not (out/'cases/case1/N1/datcon1').exists()


def test_redirected_scope_refused(batch, tmp_path):
    task = initialize(batch)
    scope = Path(task['workdir'])
    scope.parent.mkdir(parents=True)
    scope.symlink_to(batch[1]/'case1')
    with pytest.raises(ValueError, match='redirected'):
        runner.execute_task(task, resume=True)
