# dgx1: code for the 2026-11-06 DGX run

These are exactly the files that ran on the DGX (`~/arch_bench/`). The results are in `../results/` (`gauss`, `blur`, `spec_eq`, `spec_c2f`, `report_all.txt`, `samples_top.png`).

| File | What it does |
|---|---|
| `arch_bench.py` | Everything for the main comparison: data stream (identical for every model), original UNet, path U-Nets (v1 = Colab bricks, v2 = fixed bricks), noise-routed U-Net, training (`train_multi`), FID/IS evaluation, `report` |
| `novel.py` | The new pipeline: Haar-wavelet spectral schedules (`spec_eq`, `spec_c2f`) and the **SWL** architecture (+ `swl_wide`, `swl_p3`, `swl_long`) |
| `run_all.sbatch` | Job 1: trains 9 models × {gauss, blur} for 4,000 steps, then evaluates them |
| `novel.sbatch` | Job 2: runs `novel_train.sh`, then evaluates the spectral runs |
| `novel_train.sh` / `novel_eval.sh` | Train / evaluate SWL and the spectral-schedule runs |
| `improve.sh` / `improve_b.sh` | Improvement round: `swl_wide`, `swl_p3` (4k steps) and `swl_long` (8k steps) |

## How to run (on the DGX)
```bash
ssh dgx
mkdir -p ~/arch_bench/logs && cd ~/arch_bench     # copy these files here
# data: ~/diffusion-3/cifar10_airplanes.npz (override with BENCH_DATA=...)
echo 4000 > target_steps                          # shared step count (read live during training)
sbatch run_all.sbatch                             # main comparison
# single runs, inside a GPU job:
python arch_bench.py train_multi --models original_unet,unet_v2 --kind gauss --steps 4000
python novel.py train gauss swl 4000
python novel.py eval  gauss swl
python arch_bench.py report
```
SLURM rules on this machine: partition `gpu`, `--gres=gpu:1`, `--mem=128G` (required, and the maximum).
Conda env: `source /raid/money/miniconda3/etc/profile.d/conda.sh && conda activate diffusion`.

Env overrides: `BENCH_ROOT` (output folder, default `~/arch_bench`), `BENCH_DATA`, `BENCH_N_EVAL` (default 5000), `BENCH_EVAL_STEPS` (default 100).
