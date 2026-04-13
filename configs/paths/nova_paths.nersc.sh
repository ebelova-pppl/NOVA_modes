# ~/src_nova/configs/paths/nova_paths.nersc.sh

# -----------------------------
# NOVA AE project paths at NERSC
# -----------------------------

# Canonical git repo
export NOVA_REPO_TAE="$HOME/src_nova"                            # Main branch with TAE-only work.
export NOVA_REPO_MIXED="$HOME/src_nova.worktrees/mixed_branch"   # Mixed TAE+EAE work.
export NOVA_REPO=$NOVA_REPO_MIXED   # Default to mixed branch, since that's the working branch. Switch to main branch if working on TAE-only.

# Persistent data / models / saved results for TAE-only work or for mixed TAE+EAE data sets.

export NOVA_DATA_TAE="/global/cfs/cdirs/m314/nova/data"      # TAE-only data (in main branch)
export NOVA_DATA_MIXED="/global/cfs/cdirs/m314/nova2/data"   # Mixed TAE+EAE data (for training TAE+EAE models, in mixed_branch)
export NOVA_DATA=$NOVA_DATA_MIXED    # Default to mixed data, since that's the working branch. Switch to TAE-only if working on main_branch.
export NOVA_MODELS="/global/cfs/cdirs/m314/nova2/models"
export NOVA_RESULTS="/global/cfs/cdirs/m314/nova2/results"

# Active run area in scratch
export NOVA_RUN_RF="$SCRATCH/nova_rf"
export NOVA_RUN_CNN="$SCRATCH/nova_cnn"

# Version-controlled labeled training list
export NOVA_TRAIN_CSV="$NOVA_REPO/training_labels/train_master.csv"                  # TAE-only training list (in main branch)
export NOVA_TRAIN_CSV_TAE="$NOVA_REPO/training_labels/train_master_tae.csv"
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
#export NOVA_LOGS="$NOVA_RUN_ROOT/logs"
#export NOVA_TMP="$NOVA_RUN_ROOT/tmp"

# -----------------------------
# Helper functions
# -----------------------------

nova_env() {
    echo "NOVA_REPO      = $NOVA_REPO"
    echo "NOVA_DATA      = $NOVA_DATA"
    echo "NOVA_MODELS    = $NOVA_MODELS"
    echo "NOVA_RESULTS   = $NOVA_RESULTS"
    echo "NOVA_RUN_ROOT  = $NOVA_RUN_ROOT"
    echo "NOVA_TRAIN_CSV = $NOVA_TRAIN_CSV"
    echo "PYTHONPATH     = $PYTHONPATH"
}

nova_cdrepo() {
    cd "$NOVA_REPO"
}

nova_run_sort() {
    python "$NOVA_REPO/scripts/sort_shot.py" "$@"
}