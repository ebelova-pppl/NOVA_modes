# ~/src_nova/configs/paths/nova_paths.nersc.sh

# -----------------------------
# NOVA AE project paths at NERSC
# -----------------------------

# Canonical git repo
export NOVA_REPO="$HOME/src_nova"

# Persistent data / models / saved results
export NOVA_DATA="/global/cfs/cdirs/m314/nova/data"
export NOVA_MODELS="/global/cfs/cdirs/m314/nova/models"
export NOVA_RESULTS="/global/cfs/cdirs/m314/nova/results"

# Active run area in scratch
export NOVA_RUN_ROOT="$SCRATCH/nova_s"

# Version-controlled labeled training list
export NOVA_TRAIN_CSV="$NOVA_REPO/training_labels/train_master.csv"

# Python imports from src/
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$NOVA_REPO/src"
else
    export PYTHONPATH="$NOVA_REPO/src:$PYTHONPATH"
fi

# -----------------------------
# Convenience locations
# -----------------------------
export NOVA_RUN_RF="$NOVA_RUN_ROOT/rf"
export NOVA_RUN_CNN="$NOVA_RUN_ROOT/cnn"
export NOVA_LOGS="$NOVA_RUN_ROOT/logs"
export NOVA_TMP="$NOVA_RUN_ROOT/tmp"

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