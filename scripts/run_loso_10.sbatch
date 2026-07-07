#!/bin/bash
#SBATCH -J nova_loso
#SBATCH -A m314_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00
#SBATCH -o slurm-nova-loso-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load pytorch
source configs/paths/nova_paths.nersc.sh

CNN_M_TARGET="${CNN_M_TARGET:-54}"
LOSO_TAG="${LOSO_TAG:-loso_13_M${CNN_M_TARGET}}"

python -u scripts/run_loso_10.py \
  --steps all \
  --out_root "outputs/${LOSO_TAG}" \
  --work_root "$SCRATCH/nova_s/${LOSO_TAG}" \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_batch_size 8 \
  --cnn_m_target "$CNN_M_TARGET" \
  --cnn_cache_data \
  "$@"
