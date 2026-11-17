"""Isolated serial NOVA continuum tasks, shared by MPI and serial execution.

The producer is gocont_tr. This module preserves its native datcon values;
scientific preprocessing and historical comparisons belong to existing loaders.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone

SCHEMA = 'nova-continuum-run-v1'
DEFAULTS = dict(binary='gocont_tr', gamma='1.66666666', iden='4', ksound='1',
                ns_first='3', ns_last='201', ns_step='1')
REQUIRED = {'shot', 'ntor', 'lmin', 'lmax'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def read_outcon_settings(path):
    """Read the first emitted namelist, without interpreting Fortran code."""
    text = Path(path).read_text(errors='replace').split(' /', 1)[0]
    result = {}
    for key in ['ntor', 'lmin', 'lmax', 'iden', 'ksound', 'gamma']:
        match = re.search(r'\b' + key + r'\s*=\s*([^,\s]+)', text, re.I)
        if not match:
            raise ValueError(f'{path}: missing {key} in output namelist')
        result[key] = float(match[1].replace('D', 'E')) if key == 'gamma' else int(match[1])
    match = re.search(r'\bNS\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', text, re.I)
    if not match:
        raise ValueError(f'{path}: missing NS in output namelist')
    result.update(zip(['ns_first', 'ns_last', 'ns_step'], map(int, match.groups())))
    return result


def prepare_tasks(csv_path, data_root, output_root):
    """Validate every task and fingerprint inputs before creating outputs."""
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(data_root) or data_root.is_relative_to(output_root):
        raise ValueError('Output root must be separate from the input data tree')
    with Path(csv_path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not REQUIRED <= fields or fields - REQUIRED - set(DEFAULTS):
            raise ValueError(f'Task CSV requires {sorted(REQUIRED)}; optional: {sorted(DEFAULTS)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Task CSV is empty')
    tasks, used, fingerprints = [], set(), {}
    for index, row in enumerate(rows):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'Malformed CSV row {index+2}')
        values = {**DEFAULTS, **{k: v.strip() for k, v in row.items() if v.strip()}}
        if not REQUIRED <= values.keys():
            raise ValueError(f'CSV row {index+2}: required values cannot be empty')
        shot = values['shot']
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', shot):
            raise ValueError(f'Invalid case identifier: {shot!r}')
        settings = {k: int(values[k]) for k in ['ntor', 'lmin', 'lmax', 'iden', 'ksound', 'ns_first', 'ns_last', 'ns_step']}
        settings['gamma'] = float(values['gamma'].replace('D', 'E'))
        if (settings['ntor'] <= 0 or settings['lmin'] > settings['lmax']
                or settings['iden'] != 4 or settings['ksound'] not in (0, 1)
                or not 1 <= settings['ns_first'] < settings['ns_last']
                or settings['ns_step'] != 1 or not math.isfinite(settings['gamma'])
                or settings['gamma'] <= 0):
            raise ValueError(f'Unsupported or invalid settings for {shot}: {settings}')
        key = f'{shot}/N{settings["ntor"]}'
        if key in used:
            raise ValueError(f'Duplicate task: {key}')
        used.add(key)
        binary = Path(values['binary'])
        if binary.is_absolute() or '..' in binary.parts:
            raise ValueError('binary must be a path relative to the case directory')
        case = data_root / shot
        inputs = {}
        for name, relative in [('mapdsk', 'mapdsk'), ('mpout1', 'mpout1'),
                               ('transp.dat', 'transp.dat'), ('gocont_tr', binary)]:
            source = (case / relative).resolve()
            if not source.is_file() or not source.stat().st_size:
                raise ValueError(f'Missing or empty input: {source}')
            if name == 'gocont_tr' and not os.access(source, os.X_OK):
                raise ValueError(f'Executable is not executable: {source}')
            if source.is_relative_to(output_root):
                raise ValueError(f'Input resolves inside output root: {source}')
            if source not in fingerprints:
                fingerprints[source] = dict(source=str(source), bytes=source.stat().st_size,
                                            sha256=sha256(source))
            inputs[name] = fingerprints[source]
        identity = dict(schema=SCHEMA, shot=shot, settings=settings, inputs=inputs)
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        tasks.append(dict(index=index, key=key, workdir=str(output_root/'cases'/key),
                          fingerprint=fingerprint, **identity))
    return tasks


def initialize_batch(output_root, tasks, resume=False):
    root = Path(output_root)
    identity = {t['key']: t['fingerprint'] for t in tasks}
    manifest = root/'batch_manifest.json'
    if root.exists():
        if not resume or not manifest.is_file():
            raise ValueError(f'Refusing existing output directory: {root}; use --resume for this batch')
        saved = json.loads(manifest.read_text())
        if (saved['schema'] != SCHEMA or saved['task_fingerprints'] != identity
                or saved['runner_sha256'] != sha256(__file__)):
            raise ValueError('Resume refused: task settings, source input hashes, or worker code changed')
    else:
        root.mkdir(parents=True)
        write_json(manifest, dict(schema=SCHEMA, created_utc=utcnow(), task_fingerprints=identity,
                                 runner_sha256=sha256(__file__), tasks=tasks))


def validate_datcon(path, n_radial):
    """Structural validation only; preserve sentinels and all raw columns."""
    with Path(path).open() as stream:
        fields = stream.readline().split()
        if len(fields) != 2:
            raise ValueError('datcon: expected two radial-index bounds')
        first, last = map(int, fields)
        if not 1 <= first <= last <= n_radial:
            raise ValueError(f'datcon: invalid radial bounds {first}, {last}')
        rows = [[float(v.replace('D', 'E')) for v in line.split()] for line in stream if line.strip()]
    if len(rows) != last-first+1 or any(len(r) != 3 for r in rows):
        raise ValueError('datcon: wrong row count or missing one of three producer columns')
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError('datcon: nonfinite output')
    if not any(0 < row[0] <= row[1] and row[0] != 1000 and row[1] != 1000 for row in rows):
        raise ValueError('datcon: no positive ordered nonsentinel boundary pair')
    return dict(first=first, last=last, rows=len(rows))


def resumed_result(task):
    scope = Path(task['workdir'])
    path = scope/'receipt.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt['fingerprint'] != task['fingerprint']:
        raise ValueError(f'{task["key"]}: resume fingerprint differs')
    if receipt['status'] != 'success':
        return None
    for name, digest in receipt['outputs'].items():
        if sha256(scope/name) != digest:
            raise ValueError(f'{task["key"]}: completed output changed: {name}')
    return dict(index=task['index'], key=task['key'], status='success', resumed=True,
                elapsed_seconds=receipt['elapsed_seconds'], attempt=receipt['attempt'],
                error='', datcon_sha256=receipt['outputs'][f'datcon{task["settings"]["ntor"]}'])


def publish_outputs(scope, work, receipt):
    """Finish a validated attempt, including recovery after interrupted publication."""
    names = {f'datcon{receipt["settings"]["ntor"]}': 'datcon',
             'outcon': 'outcon', 'commands': 'commands', 'stdout.log': 'stdout.log'}
    for name, digest in receipt['outputs'].items():
        source, dest = work/names[name], scope/name
        if sha256(source) != digest:
            raise ValueError(f'Validated attempt changed: {source}')
        if dest.is_symlink():
            raise ValueError(f'Refusing symlink at output: {dest}')
        if dest.exists():
            if sha256(dest) != digest:
                raise ValueError(f'Refusing to replace differing output: {dest}')
        else:
            os.link(source, dest)
    receipt.update(status='success', error='')
    write_json(work/'receipt.json', receipt)
    write_json(scope/'receipt.json', receipt)


def execute_task(task, timeout=1200, resume=False):
    """Run a serial producer in a private attempt directory; never touch source outputs."""
    scope = Path(task['workdir'])
    if scope.resolve() != scope or (scope/'attempts').resolve() != scope/'attempts':
        raise ValueError(f'Refusing redirected output directory: {scope}')
    if resume:
        result = resumed_result(task)
        if result is not None:
            return result
        # A completed calculation may have been interrupted while linking its
        # outputs. Recover that same validated attempt rather than rerunning it.
        for path in sorted((scope/'attempts').glob('*/receipt.json'), reverse=True):
            saved = json.loads(path.read_text())
            if saved['status'] == 'validated' and saved['fingerprint'] == task['fingerprint']:
                publish_outputs(scope, path.parent, saved)
                return resumed_result(task)
    elif scope.exists():
        raise ValueError(f'Refusing existing task directory: {scope}')
    scope.mkdir(parents=True, exist_ok=resume)
    attempts = scope/'attempts'
    attempts.mkdir(exist_ok=resume)
    number = 1
    while (attempts/f'{number:04d}').exists():
        number += 1
    work = attempts/f'{number:04d}'
    work.mkdir()
    settings = task['settings']
    receipt = dict(schema=SCHEMA, fingerprint=task['fingerprint'], status='running',
                   started_utc=utcnow(), host=socket.gethostname(), settings=settings,
                   inputs=task['inputs'], attempt=str(work.relative_to(scope)),
                   runner_sha256=sha256(__file__))
    write_json(work/'receipt.json', receipt)
    start = time.monotonic()
    try:
        for name, source in task['inputs'].items():
            shutil.copy2(source['source'], work/name)
            if sha256(work/name) != source['sha256']:
                raise ValueError(f'Input changed since preflight: {source["source"]}')
        commands = (' &inp ' + ' '.join(f'{key}={settings[key]}' for key in
                     ['iden', 'ntor', 'lmin', 'lmax', 'gamma', 'ksound'])
                    + f' plmax=500. 15. 5. ns={settings["ns_first"]} {settings["ns_last"]} 1 /\n2\n')
        (work/'commands').write_text(commands)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (work/'commands').open('rb') as stdin, (work/'stdout.log').open('wb') as stdout:
            process = subprocess.run(['./gocont_tr'], cwd=work, stdin=stdin, stdout=stdout,
                                     stderr=subprocess.STDOUT, timeout=timeout, env=env)
        receipt['returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'gocont_tr exited with {process.returncode}; see stdout.log')
        for name, source in task['inputs'].items():
            if sha256(work/name) != source['sha256']:
                raise ValueError(f'Producer modified its input copy: {name}')
        actual = read_outcon_settings(work/'outcon')
        if actual != settings:
            raise ValueError(f'Producer settings mismatch: expected {settings}, got {actual}')
        for name in ['outcon', 'stdout.log']:
            text = (work/name).read_text(errors='replace')
            if re.search(r'ier,\s*nsrf\s*=|error in subroutine', text, re.I):
                raise ValueError(f'Producer reported an eigenvalue error in {name}')
        receipt['datcon_structure'] = validate_datcon(work/'datcon', settings['ns_last'])
        outputs = {dst: sha256(work/src) for src, dst in
                   [('datcon', f'datcon{settings["ntor"]}'), ('outcon', 'outcon'),
                    ('commands', 'commands'), ('stdout.log', 'stdout.log')]}
        receipt.update(status='validated', outputs=outputs, error='',
                       elapsed_seconds=time.monotonic()-start, finished_utc=utcnow())
        write_json(work/'receipt.json', receipt)
        publish_outputs(scope, work, receipt)
    except Exception as exc:
        if receipt['status'] == 'validated':
            # Keep the recoverable attempt receipt intact if publication fails.
            failed = dict(receipt, status='failed', error=f'{type(exc).__name__}: {exc}')
            write_json(scope/'receipt.json', failed)
            raise
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    receipt.update(elapsed_seconds=time.monotonic()-start, finished_utc=utcnow())
    write_json(work/'receipt.json', receipt)
    write_json(scope/'receipt.json', receipt)
    return dict(index=task['index'], key=task['key'], status=receipt['status'], resumed=False,
                elapsed_seconds=receipt['elapsed_seconds'], attempt=receipt['attempt'],
                error=receipt['error'], datcon_sha256=receipt.get('outputs', {}).get(f'datcon{settings["ntor"]}', ''))


def run_worker(task, timeout, resume):
    """Convert per-task failures to data so independent tasks can finish."""
    try:
        return execute_task(task, timeout, resume)
    except Exception as exc:
        return dict(index=task['index'], key=task['key'], status='failed', resumed=False,
                    elapsed_seconds=0., attempt='', datcon_sha256='', error=f'{type(exc).__name__}: {exc}')
