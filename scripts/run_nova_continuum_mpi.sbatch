#!/bin/bash
# Submit from the repository root; select account/partition/resources at submission.
# Example: sbatch --ntasks=11 --cpus-per-task=1 --time=00:10:00 \
#   scripts/run_nova_continuum_mpi.sbatch --tasks tasks.csv \
#   --data-root /path/to/DiTw --output-root /path/to/new/run
# Set NOVA_MPI_MODULE and NOVA_MPI_PYTHON for the site's matching MPI environment.
set -euo pipefail
if [[ -n "${NOVA_MPI_MODULE:-}" ]]; then
    module load "$NOVA_MPI_MODULE"
fi
: "${SLURM_NTASKS:?Submit with --ntasks (one coordinator plus workers)}"
cd "${NOVA_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec mpiexec -n "$SLURM_NTASKS" "${NOVA_MPI_PYTHON:-python}" \
    scripts/run_nova_continuum_mpi.py "$@"
