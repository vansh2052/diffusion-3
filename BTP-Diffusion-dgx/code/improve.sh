#!/bin/bash
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/arch_bench
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" TF_CPP_MIN_LOG_LEVEL=2
python -u novel.py train gauss swl_wide,swl_p3 4000 >> logs/improve_train_a.log 2>&1 &
python -u novel.py train gauss swl_long        8000 >> logs/improve_train_b.log 2>&1 &
wait
python -u novel.py eval gauss swl_wide > logs/improve_eval_wide.log 2>&1 &
python -u novel.py eval gauss swl_p3   > logs/improve_eval_p3.log 2>&1 &
python -u novel.py eval gauss swl_long > logs/improve_eval_long.log 2>&1 &
wait
