# ~/src_nova/NOVA_modes/configs/paths/nova_paths.flux.csh

# ----------------------------
# NOVA AE project paths on Flux
# ----------------------------

# Recommended Flux modules before running these scripts:
#   module load anaconda3
#   conda activate /p/hym/conda_envs/torch-hym     # needed for CNN training / inference

# tcsh does not expose a sourced file path like bash's BASH_SOURCE. The default
# below matches the Flux clone layout; setenv NOVA_REPO before sourcing this
# file if the repo lives somewhere else.
if (! $?NOVA_REPO) then
    setenv NOVA_REPO "${HOME}/src_nova/NOVA_modes"
endif

# Persistent data / models / saved results. On Flux, HOME is expected to be
# /u/ebelova, so these defaults expand to the requested /u/ebelova/... layout.
if ($?NOVA_FLUX_HOME) then
    set _NOVA_FLUX_HOME = "$NOVA_FLUX_HOME"
else
    set _NOVA_FLUX_HOME = "$HOME"
endif

if ($?NOVA_FLUX_WORK_ROOT) then
    set _NOVA_FLUX_WORK_ROOT = "$NOVA_FLUX_WORK_ROOT"
else
    set _NOVA_FLUX_WORK_ROOT = "${_NOVA_FLUX_HOME}/src_nova"
endif

setenv NOVA_DATA_TAE "${_NOVA_FLUX_HOME}/NOVA/data_tae"
setenv NOVA_DATA_MIXED "${_NOVA_FLUX_WORK_ROOT}/data_mixed"
setenv NOVA_DATA "$NOVA_DATA_MIXED"
setenv NOVA_MODELS "${_NOVA_FLUX_WORK_ROOT}/models"
setenv NOVA_RESULTS "${_NOVA_FLUX_WORK_ROOT}/results"

# Active run areas. Override NOVA_RUN_ROOT before sourcing if a different Flux
# scratch or work directory is preferred.
if (! $?NOVA_RUN_ROOT) then
    setenv NOVA_RUN_ROOT "${_NOVA_FLUX_WORK_ROOT}/runs"
endif
setenv NOVA_RUN_RF "${NOVA_RUN_ROOT}/nova_rf"
setenv NOVA_RUN_CNN "${NOVA_RUN_ROOT}/nova_cnn"

unset _NOVA_FLUX_HOME
unset _NOVA_FLUX_WORK_ROOT

# Version-controlled labeled training lists.
setenv NOVA_TRAIN_CSV "${NOVA_REPO}/training_labels/tae_like.csv"
setenv NOVA_TRAIN_CSV_TAE "${NOVA_REPO}/training_labels/tae_like.csv"
setenv NOVA_TRAIN_CSV_MIXED "${NOVA_REPO}/training_labels/all_modes.csv"

# Flux is CPU-only for this workflow. Override after sourcing only if needed.
setenv NOVA_TORCH_DEVICE cpu

# CPU threading defaults for NumPy / SciPy / PyTorch. Increase
# NOVA_CPUS_PER_TASK before sourcing when running inside a larger allocation.
if (! $?NOVA_CPUS_PER_TASK) then
    setenv NOVA_CPUS_PER_TASK 1
endif
if (! $?OMP_NUM_THREADS) then
    setenv OMP_NUM_THREADS "$NOVA_CPUS_PER_TASK"
endif
if (! $?MKL_NUM_THREADS) then
    setenv MKL_NUM_THREADS "$NOVA_CPUS_PER_TASK"
endif

# Python imports from src/
if (! $?PYTHONPATH) then
    setenv PYTHONPATH "${NOVA_REPO}/src"
else
    setenv PYTHONPATH "${NOVA_REPO}/src:${PYTHONPATH}"
endif

if ($?CUDA_VISIBLE_DEVICES) then
    setenv NOVA_CUDA_VISIBLE_DEVICES_STATUS "$CUDA_VISIBLE_DEVICES"
else
    setenv NOVA_CUDA_VISIBLE_DEVICES_STATUS "<unset>"
endif

# -----------------------------
# Helper aliases
# -----------------------------

alias nova_env 'echo "NOVA_REPO      = $NOVA_REPO"; echo "NOVA_DATA      = $NOVA_DATA"; echo "NOVA_DATA_TAE  = $NOVA_DATA_TAE"; echo "NOVA_DATA_MIXED = $NOVA_DATA_MIXED"; echo "NOVA_MODELS    = $NOVA_MODELS"; echo "NOVA_RESULTS   = $NOVA_RESULTS"; echo "NOVA_RUN_ROOT  = $NOVA_RUN_ROOT"; echo "NOVA_TRAIN_CSV = $NOVA_TRAIN_CSV"; echo "NOVA_TRAIN_CSV_TAE = $NOVA_TRAIN_CSV_TAE"; echo "NOVA_TRAIN_CSV_MIXED = $NOVA_TRAIN_CSV_MIXED"; echo "NOVA_TORCH_DEVICE = $NOVA_TORCH_DEVICE"; echo "NOVA_CPUS_PER_TASK = $NOVA_CPUS_PER_TASK"; echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"; echo "MKL_NUM_THREADS = $MKL_NUM_THREADS"; echo "CUDA_VISIBLE_DEVICES = $NOVA_CUDA_VISIBLE_DEVICES_STATUS"; echo "PYTHONPATH     = $PYTHONPATH"'

alias nova_cdrepo 'cd "$NOVA_REPO"'
alias nova_run_sort 'python "$NOVA_REPO/scripts/sort_shot.py" \!*'
alias nova_run_cpu 'echo "nova_run_cpu: running directly on CPU."; \!*'
alias nova_cpu_smoke 'python -u "$NOVA_REPO/scripts/torch_runtime.py" --device cpu --smoke \!*'
alias nova_run_cnn_raw 'python -u "$NOVA_REPO/scripts/cnn_raw.py" --device cpu \!*'
alias nova_run_cnn_straightened 'python -u "$NOVA_REPO/scripts/cnn_straightened.py" \!*'
alias nova_run_cnn_hybrid 'python -u "$NOVA_REPO/scripts/cnn_hybrid.py" \!*'
