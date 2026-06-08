#!/bin/bash
#SBATCH -J nova_loso10
#SBATCH -A m314_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00
#SBATCH -o slurm-nova-loso10-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load pytorch
source configs/paths/nova_paths.nersc.sh

python -u scripts/run_loso_10.py \
  --steps all \
  --out_root outputs/loso_10 \
  --work_root "$SCRATCH/nova_s/loso_10" \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_batch_size 8 \
  --cnn_cache_data \
  "$@"
