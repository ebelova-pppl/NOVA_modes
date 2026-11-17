#!/usr/bin/env python3
"""Reconstruct native NOVA continua with independent serial producers under MPI.

Adapted from the scheduling pattern in FRC_AI/build_frcin_atlas_mpi.py.
Example: mpiexec -n 11 python scripts/run_nova_continuum_mpi.py \
  --tasks tasks.csv --data-root /path/to/DiTw --output-root /path/to/new/run
Use --dry-run for read-only preflight, --serial for a non-MPI reference run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import csv
import math
import os
from pathlib import Path
import sys
import time

from _repo_bootstrap import ensure_repo_src_on_path
ensure_repo_src_on_path()
from nova_continuum_run import initialize_batch, prepare_tasks, run_worker, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True, type=Path, help='CSV: shot,ntor,lmin,lmax; optional settings documented in scripts/README.md')
    p.add_argument('--data-root', required=True, type=Path, help='Read-only original case directories')
    p.add_argument('--output-root', required=True, type=Path, help='New directory outside the source data tree')
    p.add_argument('--resume', action='store_true', help='Reuse hash-verified completed tasks and retry failed tasks in new attempt directories')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print planned tasks without writing files or running NOVA')
    p.add_argument('--serial', action='store_true', help='Run the same worker sequentially without requiring mpi4py')
    p.add_argument('--timeout', type=float, default=1200., help='Maximum seconds per continuum calculation (default: 1200)')
    p.add_argument('--checkpoint-every', type=int, default=20, help='Write progress/results after this many tasks (default: 20)')
    args = p.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.checkpoint_every <= 0:
        p.error('--timeout and --checkpoint-every must be positive')
    return args


def save_results(root, records):
    path = root/'results.csv'
    temp = path.with_suffix('.tmp')
    rows = sorted(records, key=lambda r:r['index'])
    if rows:
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        temp.replace(path)


def run_root(args, executor=None, mpi_size=1):
    start = time.monotonic()
    try:
        tasks = prepare_tasks(args.tasks, args.data_root, args.output_root)
        if args.dry_run:
            print(f'Preflight passed: {len(tasks)} tasks across {len({t["shot"] for t in tasks})} cases; no files written.')
            print(f'Output: {args.output_root.resolve()}')
            return 0
        initialize_batch(args.output_root, tasks, args.resume)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    records = []
    root = args.output_root.resolve()
    print(f'Starting {len(tasks)} tasks with {max(1,mpi_size-1)} worker(s).', flush=True)
    if executor is None:
        results = (run_worker(t, args.timeout, args.resume) for t in tasks)
    else:
        futures = {executor.submit(run_worker, t, args.timeout, args.resume): t for t in tasks}

        def completed():
            for future in as_completed(futures):
                task = futures[future]
                try:
                    yield future.result()
                except Exception as exc:
                    yield dict(index=task['index'],key=task['key'],status='failed',resumed=False,
                               elapsed_seconds=0.,attempt='',datcon_sha256='',error=str(exc))
        results = completed()
    try:
        for result in results:
            records.append(result)
            if result['status'] != 'success':
                print(f'FAILED {result["key"]}: {result["error"]}', file=sys.stderr, flush=True)
            if len(records) % args.checkpoint_every == 0 or len(records) == len(tasks):
                save_results(root, records)
                print(f'Completed {len(records)}/{len(tasks)}; failed={sum(r["status"] != "success" for r in records)}', flush=True)
    finally:
        save_results(root, records)
    failed = sum(r['status'] != 'success' for r in records)
    summary = dict(tasks=len(tasks), completed=len(records), failed=failed,
                   succeeded=len(records)-failed, resumed=sum(r['resumed'] for r in records),
                   mpi_ranks=mpi_size, elapsed_seconds=time.monotonic()-start)
    write_json(root/'summary.json', summary)
    print(f'Finished in {summary["elapsed_seconds"]:.1f}s. Results: {root/"results.csv"}', flush=True)
    return int(failed > 0 or len(records) != len(tasks))


def main(argv=None):
    args = parse_args(argv)
    if args.serial or args.dry_run:
        if any(int(os.environ.get(key, '1')) > 1 for key in
               ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE']):
            print('ERROR: invoke --serial/--dry-run directly, without an MPI launcher.', file=sys.stderr)
            return 1
        return run_root(args)
    try:
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
    except (ImportError, RuntimeError) as exc:
        print(f'ERROR: load a matching MPI/mpi4py environment, or use --serial: {exc}', file=sys.stderr)
        return 1
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        if comm.Get_rank() == 0:
            print('ERROR: MPI requires at least two ranks (one coordinator and workers); use --serial otherwise.', file=sys.stderr)
        return 1
    code = None
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            code = run_root(args, executor, comm.Get_size())
    return comm.bcast(code, root=0)


if __name__ == '__main__':
    raise SystemExit(main())
