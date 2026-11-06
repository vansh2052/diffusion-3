#!/bin/bash
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/arch_bench
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" TF_CPP_MIN_LOG_LEVEL=2
python -u novel.py train gauss swl_long 8000 >> logs/improve_train_b.log 2>&1
