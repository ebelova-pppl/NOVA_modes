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
export NOVA_RUN_RF="$SCRATCH/nova_rf"
export NOVA_RUN_CNN="$SCRATCH/nova_cnn"

# Version-controlled labeled training list
export NOVA_TRAIN_CSV="$NOVA_REPO/training_labels/train_master.csv"   # Keep the legacy TAE-only training default for now.
export NOVA_TRAIN_CSV_TAE="$NOVA_REPO/training_labels/train_tae.csv"
export NOVA_TRAIN_CSV_MIXED="$NOVA_REPO/training_labels/all_modes.csv"

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
    echo "PYTHONPATH     = $PYTHONPATH"
}

nova_cdrepo() {
    cd "$NOVA_REPO"
}

nova_run_sort() {
    python "$NOVA_REPO/scripts/sort_shot.py" "$@"
}
