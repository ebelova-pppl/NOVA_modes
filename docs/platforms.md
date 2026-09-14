# Environments: Flux and Perlmutter

Choose the environment for the **method you intend to run**. The current
production sorter, `sort_shot_mixed.py --method rules`, does not load RF or CNN
checkpoints. A Perlmutter-like AI environment on Flux is needed when using the
Perlmutter-trained AI models, not for current deterministic sorting.

| Workflow | Python packages | Model environment / hardware |
|---|---|---|
| Current production rules and severity-based duplicate ranking | NumPy 2.x and SciPy | No scikit-learn, PyTorch, checkpoint, or GPU required. |
| Plotting and visual review without RF guidance | Rules dependencies plus Matplotlib | Use `label_modes_fast.py --no-rf` to disable optional RF guidance. |
| RF inference or RF-based duplicate ranking in older presets | NumPy, SciPy, joblib, and the checkpoint-compatible scikit-learn environment | Current RF checkpoint was saved with scikit-learn 1.9.0; the recorded environment also includes Narwhals. CPU is sufficient. |
| `--method rf-cnn`, CNN inference, or CNN training | AI environment including PyTorch; Matplotlib for plotting | Both RF and CNN checkpoints are required by `--method rf-cnn`. Flux uses CPU for this workflow; Perlmutter can use allocated GPUs. |

NumPy 2.x matters even for rules: the current rule engine uses
`numpy.trapezoid`. An older system Python with NumPy 1.x may import some
scripts but is not sufficient for full rule evaluation. The previously used
NumPy 2.1.2 environment is suitable; the AI-specific version matching is not
a reason to install PyTorch or scikit-learn for rules.

Frozen production v5-v10 presets retain RF-based duplicate ranking. The
v11-and-later presets, including the current v13 default, use rule severity.
Reproducing the older RF ranking requires its compatible RF checkpoint and
environment. See the [configuration reference](../configs/rules/README.md).

The root [requirements.txt](../requirements.txt) is an environment-notes file,
not a pip requirements specification; do not use `pip install -r
requirements.txt`. This page separates the instructions that were previously
combined there. No environment changes are necessary if the selected Python
already has the packages for the desired workflow.

## PPPL Flux: tcsh

Flux commands below use the user's usual interactive shell, `tcsh`.

### Rules only

Use an existing Python environment with NumPy 2.x and SciPy. On the checked
Flux node (2026-09-14), `module load anaconda3` alone supplies NumPy 1.26.4
and SciPy 1.13.1, so it does **not** satisfy the rules requirement. The existing
shared `nova-perlmutter` environment does. Activate it with the separate
`source` and `conda activate` commands in the
[shared-environment setup](#ai-models-existing-perlmutter-like-environment-on-flux)
below, then check the selected interpreter:

```tcsh
cd /path/to/your/NOVA_modes
setenv NOVA_REPO "$cwd"
python -c "import sys, numpy, scipy; print(sys.executable); print('numpy', numpy.__version__); print('scipy', scipy.__version__); assert hasattr(numpy, 'trapezoid'), 'Rules require NumPy 2.x'"
python "$NOVA_REPO/scripts/sort_shot_mixed.py" --help
```

This reuses the installed environment without installing packages. Its AI
packages are present but are not imported by the current rules workflow.
If you already have another compatible environment, use it directly. For a
separate minimal rules environment in a writable project location, start
after `module load anaconda3` and use:

```tcsh
source `conda info --base`/etc/profile.d/conda.csh
setenv CONDA_PKGS_DIRS /path/to/writable/conda_pkgs
conda create --prefix /path/to/writable/conda_envs/nova-rules python=3.11 "numpy>=2,<3" scipy
conda activate /path/to/writable/conda_envs/nova-rules
```

The `create` command is a one-time installation. Replace the placeholder
paths first; on later sessions, initialize Conda and activate the environment.
Add Matplotlib if using the plot viewers. This minimal environment is not an
AI-checkpoint environment.

The sorter can run with explicit input and output paths; sourcing a path
helper is optional. Continue with [the user instructions](getting_started.md).

### Optional project paths and helpers

The repository includes an existing PPPL project configuration:

```tcsh
cd /path/to/your/NOVA_modes
setenv NOVA_REPO "$cwd"
source configs/paths/nova_paths.flux.csh
nova_env
```

Review [the Flux tcsh configuration](../configs/paths/nova_paths.flux.csh)
before adopting it for another user or project: it contains site-specific
data and cache locations. Setting `NOVA_REPO` explicitly is useful when
switching checkouts or worktrees; otherwise, the helper infers it from the
current Git checkout only when it is unset.

The helper defines:

- `NOVA_MODELS` and the version-controlled training-list paths;
- `NOVA_DATA`, the rebuilt training-data root, and `NOVA_DITW_ROOT`, the live
  shot database used for new-shot sorting;
- `NOVA_TORCH_DEVICE=cpu`, which matters only when running CNNs;
- `NOVA_CPUS_PER_TASK=1` by default and corresponding default
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS` values;
- `PYTHONPATH` for interactive imports from `src/`;
- cache and user-state locations under `/p/hym` (`XDG_*`, `PIP_CACHE_DIR`,
  `MPLCONFIGDIR`, and `PYTHONUSERBASE`) to avoid filling the small home quota.

For another dataset, override data/output variables **after** sourcing the
helper. `NOVA_DATA` and `NOVA_DITW_ROOT` have different roles and can contain
different payloads for the same shot name. Pass the intended `--shot_dir`
explicitly when sorting.

`nova_env` prints environment variables; it does not verify installed Python
package versions. Use the Python checks on this page for that. The
`set_nova_env` alias targets the shared **AI** environment. In a fresh `tcsh`
with Conda's shell hook uninitialized, its combined command can fail with
`Run 'conda init' before 'conda activate'`. Use the separate initialization
and activation lines below; no persistent `conda init` change is needed.
Also, the existing `nova_run_sort` helper invokes the
older single-model `sort_shot.py`; use `sort_shot_mixed.py` explicitly for the
current production workflow.

### AI models: existing Perlmutter-like environment on Flux

Use this section for the RF/CNN workflows, optional RF guidance in the viewer,
or reproduction of older RF-ranked rules presets. It is also fine to run
current rules from this environment if it is already available.

The existing shared environment and package-cache locations are preserved
below. They are PPPL project locations, not requirements for other users:

```tcsh
module load anaconda3
source `conda info --base`/etc/profile.d/conda.csh
setenv CONDA_PKGS_DIRS /p/hym/conda_pkgs
conda activate /p/hym/conda_envs/nova-perlmutter

cd /path/to/your/NOVA_modes
setenv NOVA_REPO "$cwd"
source configs/paths/nova_paths.flux.csh
nova_env
```

The Perlmutter column records the training environment; the Flux column was
checked against the installed shared environment on 2026-09-14:

| Package | Perlmutter (recorded) | Flux AI environment (checked) |
|---|---|---|
| Python | Not recorded here | 3.11.15 |
| NumPy | 2.1.2 | 2.1.2 |
| SciPy | Not recorded here | 1.17.1 |
| scikit-learn | 1.9.0 | 1.9.0 |
| Narwhals | >=2.0.1 | 2.25.0 |
| PyTorch | 2.8.0+cu129 | 2.8.0+cu128 |
| Matplotlib | Not recorded here | 3.10.6 |
| joblib | Not recorded here | 1.5.3 |

The Flux dependency check (`python -m pip check`) passed. Perlmutter's live
modules and GPU execution were not rechecked from this Flux session. The
table does not describe default site modules. The CUDA build suffix differs
in the existing Flux environment because this workflow runs inference there
on CPU. Previously
recorded portability checks found identical RF/raw/straightened/hybrid CNN
outputs for the checked modes; that result does not establish compatibility
with arbitrary future package versions. Checkpoint provenance is in the
[model documentation](../models/README.md).

Check the active interpreter and installed packages:

```tcsh
which python
python -m pip check
python -c "import sys, numpy, sklearn, narwhals, torch; print(sys.executable); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('narwhals', narwhals.__version__); print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

If maintaining this AI environment and it still has an older scikit-learn,
the existing upgrade instruction is:

```tcsh
python -m pip install --upgrade "scikit-learn==1.9.0" "narwhals>=2.0.1"
```

Do not use `--no-deps`. To repair an earlier no-dependency upgrade that omitted
Narwhals:

```tcsh
python -m pip install "narwhals>=2.0.1"
python -m pip check
python -c "import sys, sklearn, narwhals; print(sys.executable); print('sklearn', sklearn.__version__); print('narwhals', narwhals.__version__)"
```

These commands persistently change the activated environment, so first check
that `which python` names the intended environment. They are troubleshooting
instructions for AI use, not prerequisites for rules sorting.

After sourcing the Flux configuration, the existing CPU helpers remain:

```tcsh
nova_cpu_smoke
nova_run_cnn_raw --batch_size 32 --cache_data
```

The second command trains a CNN; it is not a sorting smoke test. The shared
configuration also provides `nova_run_cnn_straightened` and
`nova_run_cnn_hybrid`. See [script instructions](../scripts/README.md) for
training options and data requirements.

## Flux: optional Bash alternative

Use Bash syntax only when the interactive shell is actually Bash. For a
rules-only environment, initialize and activate the selected rules environment
as follows; omit activation if the current Python already passes the NumPy /
SciPy check above:

```bash
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /path/to/writable/conda_envs/nova-rules
cd /path/to/your/NOVA_modes
export NOVA_REPO="$PWD"
```

For the existing AI environment, replace the activation line with:

```bash
export CONDA_PKGS_DIRS=/p/hym/conda_pkgs
conda activate /p/hym/conda_envs/nova-perlmutter
```

The optional Bash path helper is:

```bash
source configs/paths/nova_paths.flux.sh
nova_env
```

The [Bash helper](../configs/paths/nova_paths.flux.sh) locates the checkout
relative to its own file. Unlike the Flux tcsh helper, it currently sets
`NOVA_DITW_ROOT` but does **not** set `NOVA_DATA`; for relative-path training or
inspection workflows, set `export NOVA_DATA=/path/to/training_database`
yourself. Data/cache paths are site-specific in this helper too.

## NERSC Perlmutter: Bash

### Rules only

Use CPU Python with NumPy 2.x and SciPy; no PyTorch module or GPU is required:

```bash
module load python
cd /path/to/your/NOVA_modes
export NOVA_REPO="$PWD"
python -c "import sys, numpy, scipy; print(sys.executable); print('numpy', numpy.__version__); print('scipy', scipy.__version__); assert hasattr(numpy, 'trapezoid'), 'Rules require NumPy 2.x'"
python "$NOVA_REPO/scripts/sort_shot_mixed.py" --help
```

If the loaded module does not supply those packages, use a compatible Python
environment. Do not load PyTorch solely to satisfy a rules-only workflow.
Run substantial sorting batches using the appropriate site compute allocation.

Optionally load the existing NERSC project paths:

```bash
source configs/paths/nova_paths.nersc.sh
nova_env
```

The [NERSC configuration](../configs/paths/nova_paths.nersc.sh) defines the
legacy TAE-only and mixed-data roots, sets `NOVA_DATA` to the mixed root,
locates models and training lists in the checkout, and sets project results
and scratch-run paths. Those are existing project-specific locations; other
projects should use their own explicit paths or override variables after
sourcing. This helper does not define the Flux `NOVA_DITW_ROOT` variable.

### AI models and GPU execution

The existing Perlmutter AI setup is:

```bash
module load python
module load pytorch
cd /path/to/your/NOVA_modes
source configs/paths/nova_paths.nersc.sh
nova_env
```

Verify package versions against the recorded AI environment above before
loading existing checkpoints. RF runs on CPU. CNN scripts choose CUDA when
available unless `--device` or `NOVA_TORCH_DEVICE` selects another device.

For interactive GPU work, request an allocation and then launch the Python
process with `srun` so it runs on the allocated node. Replace
`YOUR_GPU_ACCOUNT` with the project allocation (the existing project example
used `m314_g`):

```bash
salloc --nodes 1 --qos interactive --time 1:00:00 --constraint gpu --gpus 1 --account YOUR_GPU_ACCOUNT
srun --nodes 1 --ntasks 1 --cpus-per-task 1 --gpus-per-task 1 python -u "$NOVA_REPO/scripts/cnn_raw.py" --batch_size 32
```

That Python command trains a CNN. For inference, use the corresponding
classification command under `srun` instead. The path helper also provides
`nova_gpu_smoke`, which runs a small Torch allocation and reports device,
allocation, matrix-multiplication, and CPU-copy timings through `srun`.
`nova_run_cnn_raw`, `nova_run_cnn_straightened`, and `nova_run_cnn_hybrid`
launch training through the same GPU path. The helpers default to one CPU
per task; if setting `NOVA_CPUS_PER_TASK` higher, request matching CPUs in
the allocation.

If CUDA reports out-of-memory, inspect the trainers' printed free/total GPU
memory and try `--batch_size 8` or `--batch_size 4`. Diagnose without GPU memory
using `--device cpu` or `export NOVA_TORCH_DEVICE=cpu`; to explicitly select a
GPU, use `export NOVA_TORCH_DEVICE=cuda` inside the GPU allocation.

## Shared command-line and model notes

- Current `scripts/` entry points locate this checkout's `src/` directory
  relative to their own files. An inherited `PYTHONPATH` is not required for
  direct CLI commands; the path helpers remain useful for interactive imports
  and test discovery.
- `--device` and `--make_plots` belong to `sort_shot_mixed.py --method rf-cnn`.
  The rules method rejects those options. Use the separate viewers to inspect
  rule results.
- For older CNN checkpoints without model-type metadata, `cnn_classify.py`
  can infer raw, straightened, or hybrid from the checkpoint filename. For a
  generic filename, specify the kind explicitly, for example:

  ```bash
  python "$NOVA_REPO/scripts/cnn_classify.py" \
    --model /path/to/checkpoint.pt \
    --model_kind cnn_raw \
    --path /path/to/mode
  ```

- For reproducible comparisons, retain the code revision, frozen rule
  configuration or checkpoint identity, input fingerprints, and package
  versions. The rules method removes the AI-checkpoint dependency; it does
  not make numerical-library versions irrelevant to reproduction.
