# configs/paths/nova_paths.flux.sh

# ----------------------------
# NOVA AE project paths on Flux
# ----------------------------

# Recommended Flux modules before running these scripts:
#   in /p/hym:
#   module load anaconda3
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   export CONDA_PKGS_DIRS="/p/hym/conda_pkgs"    # optional: shared conda pkgs directory
#   conda activate /p/hym/conda_envs/nova-perlmutter     # needed for CNN training / inference
#   cd /path/to/your/NOVA_modes

# Canonical git repo / worktree
# Resolve the repo root from this config file so the same file works when it is
# sourced from either main or a worktree.
_NOVA_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NOVA_REPO="$(cd "${_NOVA_CONFIG_DIR}/../.." && pwd)"
unset _NOVA_CONFIG_DIR

# Version-controlled models and labeled training list.
export NOVA_MODELS="$NOVA_REPO/models"

export NOVA_TRAIN_CSV="$NOVA_REPO/training_labels/tae_like_train.csv"

# Flux is CPU-only for this workflow. Override after sourcing only if needed.
export NOVA_TORCH_DEVICE=cpu

# CPU threading defaults for NumPy / SciPy / PyTorch. Increase
# NOVA_CPUS_PER_TASK before sourcing when running inside a larger allocation.
export NOVA_CPUS_PER_TASK="${NOVA_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NOVA_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NOVA_CPUS_PER_TASK}"

# Python imports from src/
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$NOVA_REPO/src"
else
    export PYTHONPATH="$NOVA_REPO/src:$PYTHONPATH"
fi

# Keep caches and user-level Python state out of the small home directory.
export XDG_CACHE_HOME="/p/hym/cache"
export XDG_CONFIG_HOME="/p/hym/config"
export XDG_DATA_HOME="/p/hym/local/share"
export XDG_STATE_HOME="/p/hym/local/state"
export PIP_CACHE_DIR="/p/hym/cache/pip"
export MPLCONFIGDIR="/p/hym/cache/matplotlib"
export PYTHONUSERBASE="/p/hym/local"

# -----------------------------
# Helper functions
# -----------------------------

nova_env() {
    echo "NOVA_REPO      = $NOVA_REPO"
    echo "NOVA_MODELS    = $NOVA_MODELS"
    echo "NOVA_RUN_ROOT  = ${NOVA_RUN_ROOT:-<unset>}"
    echo "NOVA_TRAIN_CSV = $NOVA_TRAIN_CSV"
    echo "NOVA_TORCH_DEVICE = $NOVA_TORCH_DEVICE"
    echo "NOVA_CPUS_PER_TASK = $NOVA_CPUS_PER_TASK"
    echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
    echo "MKL_NUM_THREADS = $MKL_NUM_THREADS"
    echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "PYTHONPATH     = $PYTHONPATH"
}

nova_cdrepo() {
    cd "$NOVA_REPO"
}

nova_run_sort() {
    python "$NOVA_REPO/scripts/sort_shot.py" "$@"
}

nova_run_cpu() {
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
        echo "nova_run_cpu: job=$SLURM_JOB_ID cpus_per_task=$NOVA_CPUS_PER_TASK command=$*"
        srun --nodes 1 --ntasks 1 --cpus-per-task "$NOVA_CPUS_PER_TASK" --kill-on-bad-exit=1 "$@"
    else
        echo "nova_run_cpu: no active Slurm allocation detected; running directly on CPU."
        "$@"
    fi
}

nova_cpu_smoke() {
    nova_run_cpu python -u "$NOVA_REPO/scripts/torch_runtime.py" --device cpu --smoke "$@"
}

nova_run_cnn_raw() {
    nova_run_cpu python -u "$NOVA_REPO/scripts/cnn_raw.py" --device cpu "$@"
}

nova_run_cnn_straightened() {
    nova_run_cpu python -u "$NOVA_REPO/scripts/cnn_straightened.py" "$@"
}

nova_run_cnn_hybrid() {
    nova_run_cpu python -u "$NOVA_REPO/scripts/cnn_hybrid.py" "$@"
}
