#!/bin/bash
#SBATCH --job-name=ddpm_eval
#SBATCH --output=slurm_eval_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# Usage: sbatch --job-name=ddpm_eval_linear paper_eval.sbatch linear [extra eval args]
# Evaluates the current checkpoint in runs/<schedule>/ckpt without training further.
SCHEDULE=${1:?usage: sbatch paper_eval.sbatch <linear|cosine> [extra args]}
shift

echo "Job $SLURM_JOB_ID (eval $SCHEDULE) on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/diffusion-3

export XLA_FLAGS="--xla_gpu_enable_command_buffer="

python -u ddpm_paper.py eval --schedule "$SCHEDULE" --n-samples 10000 --sampler ddpm "$@"
