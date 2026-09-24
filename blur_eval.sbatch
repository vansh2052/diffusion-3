#!/bin/bash
#SBATCH --job-name=blur_eval
#SBATCH --output=slurm_blur_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00

# Usage: sbatch --job-name=blur_coupled blur_eval.sbatch <run_tag> [NAME=VALUE ...]
#
# Runs diffusion-3.ipynb headless. The training cells resume from their checkpoints
# (a no-op when already at the target epoch, or a fresh run when the checkpoint
# directory is new), then the evaluation cell scores every sampler and writes
# runs_blur/<run_tag>.json. Extra NAME=VALUE arguments become environment variables:
#   BLUR_SCHEDULE (coupled|linear_t)  SIGMA_MAX  KERNEL_SIZE  BLUR_CKPT_DIR  EVAL_N
#
#   before: sbatch --job-name=blur_coupled blur_eval.sbatch coupled
#   after:  sbatch --job-name=blur_fixed   blur_eval.sbatch linear_t \
#               BLUR_SCHEDULE=linear_t SIGMA_MAX=6.0 KERNEL_SIZE=33 BLUR_CKPT_DIR=checkpoints_blur_v2
TAG=${1:?usage: sbatch blur_eval.sbatch <run_tag> [NAME=VALUE ...]}
shift
for kv in "$@"; do
    export "$kv"
done
export BLUR_RUN_TAG="$TAG"

echo "Job $SLURM_JOB_ID (blur eval '$TAG') on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
echo "BLUR_SCHEDULE=${BLUR_SCHEDULE:-coupled} SIGMA_MAX=${SIGMA_MAX:-2.5} KERNEL_SIZE=${KERNEL_SIZE:-17} BLUR_CKPT_DIR=${BLUR_CKPT_DIR:-checkpoints_blur}"
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/diffusion-3

jupyter nbconvert --to notebook --execute diffusion-3.ipynb \
    --output "diffusion-3-blur-${TAG}.ipynb" \
    --ExecutePreprocessor.timeout=-1
