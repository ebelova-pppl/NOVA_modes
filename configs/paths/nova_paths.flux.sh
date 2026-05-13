# ~/src_nova/NOVA_modes/configs/paths/nova_paths.flux.sh

# ----------------------------
# NOVA AE project paths on Flux
# ----------------------------

# Recommended Flux modules before running these scripts:
#   module load anaconda3
#   conda activate <your torch env>      # needed for CNN training / inference

# Canonical git repo / worktree
# Resolve the repo root from this config file so the same file works when it is
# sourced from either main or a worktree.
_NOVA_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NOVA_REPO="$(cd "${_NOVA_CONFIG_DIR}/../.." && pwd)"
unset _NOVA_CONFIG_DIR

# Persistent data / models / saved results. On Flux, HOME is expected to be
# /u/ebelova, so these defaults expand to the requested /u/ebelova/... layout.
_NOVA_FLUX_HOME="${NOVA_FLUX_HOME:-$HOME}"
_NOVA_FLUX_WORK_ROOT="${NOVA_FLUX_WORK_ROOT:-${_NOVA_FLUX_HOME}/src_nova}"
export NOVA_DATA_TAE="${_NOVA_FLUX_HOME}/NOVA/data_tae"          # old TAE-only dataset / legacy train_master.csv
export NOVA_DATA_MIXED="${_NOVA_FLUX_WORK_ROOT}/data_mixed"      # mixed TAE+EAE training set
export NOVA_DATA="$NOVA_DATA_MIXED"                              # default to mixed data, since that is the active workflow
export NOVA_MODELS="${_NOVA_FLUX_WORK_ROOT}/models"
export NOVA_RESULTS="${_NOVA_FLUX_WORK_ROOT}/results"

# Active run areas. Override these before sourcing if a different Flux scratch
# or work directory is preferred.
export NOVA_RUN_ROOT="${NOVA_RUN_ROOT:-${_NOVA_FLUX_WORK_ROOT}/runs}"
export NOVA_RUN_RF="$NOVA_RUN_ROOT/nova_rf"
export NOVA_RUN_CNN="$NOVA_RUN_ROOT/nova_cnn"
unset _NOVA_FLUX_HOME _NOVA_FLUX_WORK_ROOT

# Version-controlled labeled training lists.
export NOVA_TRAIN_CSV="$NOVA_REPO/training_labels/tae_like.csv"
export NOVA_TRAIN_CSV_TAE="$NOVA_REPO/training_labels/tae_like.csv"
export NOVA_TRAIN_CSV_MIXED="$NOVA_REPO/training_labels/all_modes.csv"

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

# -----------------------------
# Helper functions
# -----------------------------

nova_env() {
    echo "NOVA_REPO      = $NOVA_REPO"
    echo "NOVA_DATA      = $NOVA_DATA"
    echo "NOVA_DATA_TAE  = $NOVA_DATA_TAE"
    echo "NOVA_DATA_MIXED = $NOVA_DATA_MIXED"
    echo "NOVA_MODELS    = $NOVA_MODELS"
    echo "NOVA_RESULTS   = $NOVA_RESULTS"
    echo "NOVA_RUN_ROOT  = $NOVA_RUN_ROOT"
    echo "NOVA_TRAIN_CSV = $NOVA_TRAIN_CSV"
    echo "NOVA_TRAIN_CSV_TAE = $NOVA_TRAIN_CSV_TAE"
    echo "NOVA_TRAIN_CSV_MIXED = $NOVA_TRAIN_CSV_MIXED"
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
