#!/bin/bash
#SBATCH --job-name=fid_stats
#SBATCH --output=slurm_stats_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00

# Usage: sbatch paper_stats.sbatch [keras|paper|both]
# Computes (and caches) real-CIFAR-10 reference statistics for the chosen FID
# network, plus calibration numbers: FID of test vs train, FID of two halves of
# train, and the Inception Score of the real data.
METRIC=${1:-paper}

echo "Job $SLURM_JOB_ID (stats: $METRIC) on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/diffusion-3

python -u ddpm_paper.py stats --metric "$METRIC"
