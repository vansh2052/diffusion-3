#!/bin/bash
#SBATCH --job-name=ddpm_paper
#SBATCH --output=slurm_paper_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Usage: sbatch --job-name=ddpm_linear paper_train.sbatch linear 50000 [extra train args]
# Training resumes from runs/<schedule>/ckpt, so re-submitting with a larger step
# count continues the same run.
SCHEDULE=${1:?usage: sbatch paper_train.sbatch <linear|cosine> <total_steps> [extra args]}
STEPS=${2:?usage: sbatch paper_train.sbatch <linear|cosine> <total_steps> [extra args]}
shift 2

echo "Job $SLURM_JOB_ID ($SCHEDULE, $STEPS steps) on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/diffusion-3

python -u ddpm_paper.py train --schedule "$SCHEDULE" --steps "$STEPS" "$@"
