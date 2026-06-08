# ~/src_nova/configs/paths/nova_paths.nersc.sh

# -----------------------------
# NOVA AE project paths at NERSC
# -----------------------------

# Canonical git repo / worktree
# Resolve the repo root from this config file so the same file works when it is
# sourced from either main or a worktree.
_NOVA_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NOVA_REPO="$(cd "${_NOVA_CONFIG_DIR}/../.." && pwd)"
unset _NOVA_CONFIG_DIR

# Persistent data / models / saved results for TAE-only work or for mixed TAE+EAE data sets.

export NOVA_DATA_TAE="/global/cfs/cdirs/m314/nova/data"      # old TAE-only dataset / legacy train_master.csv (for training TAE-only models, in main_branch)
export NOVA_DATA_MIXED="/global/cfs/cdirs/m314/nova2/data"   # Mixed TAE+EAE data (for training TAE+EAE models, in main and mixed_branch)
export NOVA_DATA=$NOVA_DATA_MIXED                            # Default to mixed data, since that's the working branch.
export NOVA_MODELS="/global/cfs/cdirs/m314/nova2/models"
export NOVA_RESULTS="/global/cfs/cdirs/m314/nova2/results"

# Active run area in scratch
export NOVA_RUN_RF="$SCRATCH/nova_rf"    # older version of RF
export NOVA_RUN_CNN="$SCRATCH/nova_cnn"  # older versions of CNN

# Version-controlled labeled training list
export NOVA_TRAIN_CSV="$NOVA_REPO/training_labels/tae_like_train.csv"       # "train_master.csv" was the legacy TAE-only training default.
export NOVA_TRAIN_CSV_TAE="$NOVA_REPO/training_labels/tae_like_train.csv"   # New TAE training set (for training TAE-only models)
export NOVA_TRAIN_CSV_MIXED="$NOVA_REPO/training_labels/all_modes.csv" # Includes TAE+EAE data (for now used fro splitting TAEs vs EAEs, in future for training TAE+EAE models)

# Optional Torch device override for CNN scripts:
#   export NOVA_TORCH_DEVICE=cpu     # diagnose without GPU memory
#   export NOVA_TORCH_DEVICE=cuda    # force CUDA when available
# Leave unset for the scripts' automatic cuda/cpu choice.

# Python imports from src/
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$NOVA_REPO/src"
else
    export PYTHONPATH="$NOVA_REPO/src:$PYTHONPATH"
fi

# -----------------------------
# Convenience locations
# -----------------------------
#export NOVA_RUN_RF="$NOVA_RUN_ROOT/rf"
#export NOVA_RUN_CNN="$NOVA_RUN_ROOT/cnn"
#export NOVA_TMP="$NOVA_RUN_ROOT/tmp"

# -----------------------------
# Helper functions
# -----------------------------

nova_env() {
    echo "NOVA_REPO      = $NOVA_REPO"
    echo "NOVA_DATA      = $NOVA_DATA"
    echo "NOVA_MODELS    = $NOVA_MODELS"
    echo "NOVA_RESULTS   = $NOVA_RESULTS"
    echo "NOVA_TRAIN_CSV = $NOVA_TRAIN_CSV"
    echo "NOVA_TRAIN_CSV_TAE = $NOVA_TRAIN_CSV_TAE"
    echo "NOVA_TRAIN_CSV_MIXED = $NOVA_TRAIN_CSV_MIXED"
    echo "NOVA_TORCH_DEVICE = ${NOVA_TORCH_DEVICE:-<auto>}"
    echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "PYTHONPATH     = $PYTHONPATH"
}

nova_cdrepo() {
    cd "$NOVA_REPO"
}

nova_run_sort() {
    python "$NOVA_REPO/scripts/sort_shot.py" "$@"
}

nova_srun_gpu() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "nova_srun_gpu: no active Slurm allocation detected; run salloc first."
        return 2
    fi
    echo "nova_srun_gpu: job=$SLURM_JOB_ID cpus_per_task=${NOVA_CPUS_PER_TASK:-1} command=$*"
    srun --nodes 1 --ntasks 1 --cpus-per-task "${NOVA_CPUS_PER_TASK:-1}" --gpus-per-task 1 --gpu-bind=none --kill-on-bad-exit=1 "$@"
}

nova_gpu_smoke() {
    nova_srun_gpu python -u "$NOVA_REPO/scripts/torch_runtime.py" --smoke "$@"
}

nova_run_cnn_raw() {
    nova_srun_gpu python -u "$NOVA_REPO/scripts/cnn_raw.py" "$@"
}

nova_run_cnn_straightened() {
    nova_srun_gpu python -u "$NOVA_REPO/scripts/cnn_straightened.py" "$@"
}

nova_run_cnn_hybrid() {
    nova_srun_gpu python -u "$NOVA_REPO/scripts/cnn_hybrid.py" "$@"
}
