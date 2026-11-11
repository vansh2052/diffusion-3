#!/bin/bash
#SBATCH --job-name=arch_bench
#SBATCH --output=/home/asrivastava/arch_bench/slurm_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# One process per diffusion kind; each trains all models round-robin (identical step counts).
echo "Job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, start $(date)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/arch_bench
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" TF_CPP_MIN_LOG_LEVEL=2
M=original_unet,unet_v1,ladder_v1,unet_v2,ladder_v2,ladder_rev_v2,routed_dense,routed_fixed,routed_learned
python -u arch_bench.py train_multi --models $M --kind gauss --steps 100000 > logs/train_gauss.log 2>&1 &
python -u arch_bench.py train_multi --models $M --kind blur  --steps 100000 > logs/train_blur.log 2>&1 &
wait
echo "training done $(date)"
python -u arch_bench.py eval_multi --models $M --kind gauss,blur > logs/eval.log 2>&1
python arch_bench.py report | tee report.txt
echo "all done $(date)"
