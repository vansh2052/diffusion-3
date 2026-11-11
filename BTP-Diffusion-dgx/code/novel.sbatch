#!/bin/bash
#SBATCH --job-name=novel_bench
#SBATCH --output=/home/asrivastava/arch_bench/slurm_novel_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
echo "Job $SLURM_JOB_ID start $(date)"
cd ~/arch_bench
bash novel_train.sh
echo "novel training done $(date)"
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" TF_CPP_MIN_LOG_LEVEL=2
python -u novel.py eval gauss    swl                         > logs/novel_eval_gauss.log 2>&1 &
python -u novel.py eval spec_eq  original_unet,ladder_v2,swl > logs/novel_eval_spec_eq.log 2>&1 &
python -u novel.py eval spec_c2f original_unet,ladder_v2,swl > logs/novel_eval_spec_c2f.log 2>&1 &
wait
python arch_bench.py report > report_all.txt 2>/dev/null
echo "all done $(date)"
