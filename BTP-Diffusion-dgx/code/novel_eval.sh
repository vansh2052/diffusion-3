#!/bin/bash
source /raid/money/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd ~/arch_bench
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" TF_CPP_MIN_LOG_LEVEL=2
python -u novel.py eval gauss    swl                         > logs/novel_eval_gauss.log 2>&1 &
python -u novel.py eval spec_eq  swl,original_unet,ladder_v2 > logs/novel_eval_spec_eq.log 2>&1 &
python -u novel.py eval spec_c2f swl,original_unet,ladder_v2 > logs/novel_eval_spec_c2f.log 2>&1 &
wait
