# configs/paths/nova_paths.flux.csh

# ----------------------------
# NOVA AE project paths on Flux
# ----------------------------
set _NOVA_PATHS_CONFIG_OK = 1

# Recommended Flux modules before running these scripts:
#   in /p/hym:
#   module load anaconda3
#   source `conda info --base`/etc/profile.d/conda.csh
#   setenv CONDA_PKGS_DIRS "/p/hym/conda_pkgs"    # optional: shared conda pkgs directory
#   conda activate /p/hym/conda_envs/nova-perlmutter     # needed for CNN training / inference
#   cd /path/to/your/NOVA_modes
#   source configs/paths/nova_paths.flux.csh

# tcsh does not expose a sourced file path like bash's BASH_SOURCE. Source this
# file from inside the intended Git checkout, or setenv NOVA_REPO first if you
# need to source it from somewhere else.
if (! $?NOVA_REPO) then
    git rev-parse --show-toplevel >& /dev/null
    if ($status == 0) then
        set _NOVA_REPO_FROM_GIT = `git rev-parse --show-toplevel`
        setenv NOVA_REPO "$_NOVA_REPO_FROM_GIT"
    else
        echo "ERROR: Could not determine NOVA_REPO for tcsh."
        echo "       cd into your NOVA_modes Git checkout before sourcing this file,"
        echo "       or run: setenv NOVA_REPO /path/to/your/NOVA_modes"
        set _NOVA_PATHS_CONFIG_OK = 0
        goto nova_paths_flux_done
    endif
    unset _NOVA_REPO_FROM_GIT
endif

# Persistent data / models / saved results.
if ($?NOVA_FLUX_WORK_ROOT) then
    set _NOVA_FLUX_WORK_ROOT = "$NOVA_FLUX_WORK_ROOT"
else if ($?USER) then
    set _NOVA_FLUX_WORK_ROOT = "/p/hym/${USER}/NOVA"
else
    set _NOVA_FLUX_WORK_ROOT = "/p/hym/NOVA"
endif

setenv NOVA_DATA_TAE "/u/ebelova/NOVA_old/data_tae"          # old TAE-only set, used for initial CNN training
setenv NOVA_DATA_MIXED "${_NOVA_FLUX_WORK_ROOT}/data_mixed"  # new mixed set TAEs+EAEs
setenv NOVA_DATA "$NOVA_DATA_MIXED"
setenv NOVA_MODELS "${_NOVA_FLUX_WORK_ROOT}/models_flux"
setenv NOVA_RESULTS "${_NOVA_FLUX_WORK_ROOT}/results"

# Active run areas. Override NOVA_RUN_ROOT before sourcing if a different Flux
# scratch or work directory is preferred.
if (! $?NOVA_RUN_ROOT) then
    setenv NOVA_RUN_ROOT "${_NOVA_FLUX_WORK_ROOT}/runs"
endif

unset _NOVA_FLUX_WORK_ROOT

# Version-controlled labeled training lists.
setenv NOVA_TRAIN_CSV "${NOVA_REPO}/training_labels/tae_like_train.csv"
setenv NOVA_TRAIN_CSV_TAE "${NOVA_REPO}/training_labels/tae_like_train.csv"
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
#if (! $?PYTHONPATH) then
    setenv PYTHONPATH "${NOVA_REPO}/src"
#else
#    setenv PYTHONPATH "${NOVA_REPO}/src:${PYTHONPATH}"
#endif

if ($?CUDA_VISIBLE_DEVICES) then
    setenv NOVA_CUDA_VISIBLE_DEVICES_STATUS "$CUDA_VISIBLE_DEVICES"
else
    setenv NOVA_CUDA_VISIBLE_DEVICES_STATUS "<unset>"
endif

# Keep caches and user-level Python state out of the small home directory.
setenv XDG_CACHE_HOME /p/hym/cache
setenv XDG_CONFIG_HOME /p/hym/config
setenv XDG_DATA_HOME /p/hym/local/share
setenv XDG_STATE_HOME /p/hym/local/state
setenv PIP_CACHE_DIR /p/hym/cache/pip
setenv MPLCONFIGDIR /p/hym/cache/matplotlib
setenv PYTHONUSERBASE /p/hym/local

# -----------------------------
# Helper aliases
# -----------------------------

alias set_paths 'source "$NOVA_REPO/configs/paths/nova_paths.flux.csh"'

alias set_nova_env 'module load anaconda3; source `conda info --base`/etc/profile.d/conda.csh; setenv CONDA_PKGS_DIRS "/p/hym/conda_pkgs"; conda activate /p/hym/conda_envs/nova-perlmutter' 

alias nova_env 'echo "NOVA_REPO      = $NOVA_REPO"; echo "NOVA_DATA      = $NOVA_DATA"; echo "NOVA_DATA_TAE  = $NOVA_DATA_TAE"; echo "NOVA_DATA_MIXED = $NOVA_DATA_MIXED"; echo "NOVA_MODELS    = $NOVA_MODELS"; echo "NOVA_RESULTS   = $NOVA_RESULTS"; echo "NOVA_RUN_ROOT  = $NOVA_RUN_ROOT"; echo "NOVA_TRAIN_CSV = $NOVA_TRAIN_CSV"; echo "NOVA_TRAIN_CSV_TAE = $NOVA_TRAIN_CSV_TAE"; echo "NOVA_TRAIN_CSV_MIXED = $NOVA_TRAIN_CSV_MIXED"; echo "NOVA_TORCH_DEVICE = $NOVA_TORCH_DEVICE"; echo "NOVA_CPUS_PER_TASK = $NOVA_CPUS_PER_TASK"; echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"; echo "MKL_NUM_THREADS = $MKL_NUM_THREADS"; echo "CUDA_VISIBLE_DEVICES = $NOVA_CUDA_VISIBLE_DEVICES_STATUS"; echo "PYTHONPATH     = $PYTHONPATH"'

alias nova_cdrepo 'cd "$NOVA_REPO"'
alias nova_run_sort 'python "$NOVA_REPO/scripts/sort_shot.py" \!*'
alias nova_run_cpu 'echo "nova_run_cpu: running directly on CPU."; \!*'
alias nova_cpu_smoke 'python -u "$NOVA_REPO/scripts/torch_runtime.py" --device cpu --smoke \!*'
alias nova_run_cnn_raw 'python -u "$NOVA_REPO/scripts/cnn_raw.py" --device cpu \!*'
alias nova_run_cnn_straightened 'python -u "$NOVA_REPO/scripts/cnn_straightened.py" \!*'
alias nova_run_cnn_hybrid 'python -u "$NOVA_REPO/scripts/cnn_hybrid.py" \!*'

nova_paths_flux_done:
unset _NOVA_PATHS_CONFIG_OK
