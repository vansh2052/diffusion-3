# Context: BTP Diffusion Architectures

Everything needed to understand the project: goal, data, setup, every model and pipeline, the code, the
infrastructure, and how the work developed. Results are in **`eval.md`**; the reasoning behind each choice is in **`decision.md`**.

---

## 1. Goal

Find a denoising network that beats a standard U-Net for diffusion models, starting from the idea of a **"ladder"**
U-Net, and test new noise schedules. The work runs at small scale (CIFAR-10 airplanes, 32×32) so many ideas can be
compared fairly on a Colab T4 and the department DGX.

---

## 2. Data and diffusion setup

| Item | Value |
|---|---|
| Dataset | CIFAR-10, **airplane class only**: 5,000 training images, 1,000 test images, 32×32×3 |
| Scaling | pixels mapped to **[-1, 1]** (so the forward process ends at N(0, I)) |
| Timesteps | `T = 1000`, 0-indexed t = 0..999 |
| Schedule | cosine (Nichol & Dhariwal), `s = 0.008`, betas clipped to [1e-4, 0.999] |
| Timestep input | sinusoidal embedding, 128-d (precomputed `PE[t]` for the original U-Net; computed inside the model for the others) |
| Optimizer | Adam, learning rate 2e-4 |
| EMA | decay 0.999 with warm-up `min(0.999, (1+n)/(10+n))`; **all sampling and FID use EMA weights** |
| Augmentation | random horizontal flip |
| Batch | 128 → 39 steps per epoch; **100 epochs = 3,900 steps** (Colab), **4,000 steps** (DGX) |

### Pipelines (kinds of diffusion)
| Name | Forward process | Model predicts | Sampler |
|---|---|---|---|
| **gauss** | `x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε` | the noise ε | DDIM (η = 0), 100 steps |
| **blur** | blur x0 with a Gaussian kernel whose σ rises with t (σ_max 2.5, 17×17 kernel), then mix with noise using ᾱ_t (Cold-Diffusion style) | the clean image x0 | cold-diffusion "restart" sampler, 100 steps, reuses the same noise |
| **spec_eq** (new) | Gaussian diffusion in an orthonormal 2-level Haar wavelet domain; each band has its own ᾱ: all bands cross SNR = 1 together | ε | DDIM per wavelet coefficient |
| **spec_c2f** (new) | as above, but exaggerated coarse-to-fine: fine detail is destroyed earlier, coarse layout survives longer | ε | DDIM per wavelet coefficient |

Spectral band shifts measured on real airplanes (log-SNR units; bands = coarse LL2 8×8, level-2 detail, level-1 detail):
- `spec_eq`: [−2.61, +0.68, +2.61]; at t = 500, ᾱ per band = [0.066, 0.658, 0.93]
- `spec_c2f`: [+1.31, −0.34, −1.31]; at t = 500, ᾱ per band = [0.781, 0.408, 0.208]
- The coarse band carries ≈13× the mean power.

---

## 3. Vocabulary

- **Level:** a resolution. 0 = 32×32, 1 = 16×16, 2 = 8×8, 3 = 4×4.
- **Route / path:** the list of levels a network visits, e.g. `[0, 1, 2, 1, 0]`.
- **Stepping:** moving one level at a time. **Jumping:** moving several levels in one layer.
- **Ladder:** several down-and-up passes, each going deeper. **Reverse ladder:** deepest pass first.
- **Skip connection (path models):** when a route returns to a level it visited before, the last feature map from that level is concatenated in.
- **Bricks:** the building blocks a network is made of (ResBlock, down/up layers, attention).
- **base / mults:** channel width; channels at level k = `base × mults[k]` (must be divisible by 8 for GroupNorm(8)).
- **Identical stream:** every model sees exactly the same images, flips, timesteps and noise at each training step.
- **FID:** Fréchet distance between Inception-V3 features of generated and real images (lower is better).
- **IS:** Inception Score (higher is better; real airplanes score 4.91).

---

## 4. "Original UNet" vs "path unet" (a common confusion)

There are two different U-Nets that both go down and back up:

| | **Original `UNet`** | **Path `unet`** (`unet_v1`) |
|---|---|---|
| What it is | one hand-written class from the Colab notebook | `PathUNet([0,1,2,3,2,1,0])`: a route given to a builder class |
| Bricks | `ResBlock`, `DownBlock`, `UpBlock`, `AttentionBlock` | `PResBlock`, `Resample`, `PAttentionBlock` |
| Levels | 32 → 16 → **8** | 32 → 16 → 8 → **4** |
| Width at 32×32 | up to **64** ch | 32 ch |
| Downsampling | **3×3** stride 2 (overlapping) | **2×2** stride 2 (non-overlapping) |
| Upsampling | transposed conv | nearest + 3×3 conv |
| Attention | **8×8** | 4×4 only |
| Params | 1.66M | 1.54M |

Analogy: same floor plan, different bricks. Ladders vs path `unet` compares floor plans fairly (same bricks); path
`unet` vs the original compares the bricks. The later **v2** bricks fix the path builder's weaknesses.

---

## 5. All architectures

### 5.1 Original U-Net
`original_unet`: input conv (32 ch) → DownBlock (ResBlock 32→64, 3×3 stride-2) → DownBlock (64→128) → bottleneck at 8×8
(ResBlock, attention, ResBlock) → UpBlock ×2 (transposed conv, concat skip, ResBlock) → output conv. **1,663,235 params** at `base=32`.

### 5.2 Path U-Nets (one class, `PathUNet`)
Every number in the route = "go to that level and run one ResBlock there". Moves use `Resample`.

| Name | Route | Bricks | Params |
|---|---|---|---|
| `unet` / `unet_v1` | `[0,1,2,3,2,1,0]` | v1 | 1,544,675 |
| `ladder_step` / `ladder_v1` | `[0,1,0,1,2,1,0,1,2,3,2,1,0]` | v1 | 2,353,059 |
| `ladder_jump` | `[0,1,0,2,0,3,0]` | v1 | 1,523,491 |
| `ladder_reverse` | `[0,3,0,2,0,1,0]` | v1 | 1,523,491 |
| `jump_down_step_up` | `[0,1,0,2,1,0,3,2,1,0]` | v1 | 2,109,283 |
| `w_net` | `[0,1,2,3,2,1,0,1,2,3,2,1,0]` | v1 | 2,994,883 |
| `ladder_step_b24` | ladder_step route, base 24 | v1 | 1,543,035 |
| `unet_b40` | unet route, base 40 | v1 | 2,154,907 |
| `unet_v2` | `[0,1,2,1,0]` | v2 | 2,365,251 |
| `ladder_v2` | `[0,1,0,1,2,1,0]` | v2 | 3,207,043 |
| `ladder_rev_v2` | `[0,1,2,1,0,1,0]` | v2 | 3,207,043 |
| `wnet_v2` | `[0,1,2,1,0,1,2,1,0]` | v2 | 4,652,419 (built, not run) |

- **v1 bricks** (Colab): channels 32/64/64/128 (`mults=(1,2,2,4)`), down k levels = conv with kernel = stride = 2^k, attention at 4×4.
- **v2 bricks** (DGX): channels 64/128/128 (`base=32, mults=(2,4,4)`), down one level = 3×3 stride-2 conv (avg-pool + conv for jumps), up = nearest + 3×3 conv, 3 levels, attention at 8×8.

Compute (v1, MFLOPs per image): unet 314, ladder_step 742, ladder_jump/reverse 456, jump_down_step_up 603, w_net 621.

### 5.3 Noise-routed U-Net (`RoutedUNet`)
Four levels (32/16/8/4) with **two blocks (a, b) per level going down and two going up**, a 4×4 bottleneck
(ResBlock + attention + ResBlock), attention inside the 8×8 blocks, 3×3 stride-2 down, a 1×1 `fuse` conv merging skips.
Which blocks run depends on t:

| t range | Blocks used (down, mirrored up) | Share of block compute |
|---|---|---|
| 0–100 (low noise) | L0a, L0b, L1a, L1b | 78% |
| 100–250 | L1a, L2a, L3a | 31% |
| 250–400 | L2a, L2b, L3a | 18% |
| 400–1000 (high noise) | L2a, L3a, L3b | 16% |

Modes: `routed_dense` (all blocks always), `routed_fixed` (the table), `routed_learned` (a gate g(t) per block, compute
penalty 0.01, blocks with gate < 0.1 skipped when sampling). 3.94M params.

### 5.4 SWL: Shared-Weight Wavelet Ladder (new)
1. **Haar resampling:** down = orthonormal Haar space-to-depth + 1×1 conv; up = 1×1 conv + inverse Haar (lossless).
2. **Two passes sharing weights:** pass 1 = 32→16→32, pass 2 = 32→16→8→16→32, using the **same** encoder/decoder block per level, plus a learned pass embedding added to the time embedding.
3. **Pass gates:** `h ← h + g_p(t)·(pass_out − h)`, `g_p(t) = 1 + tanh(Dense(t_emb))` (zero-init, so it starts at 1).
4. **adaLN-Zero ResBlocks:** parameter-free GroupNorm, scale/shift from the conditioning (zero-init), zero-init output conv.
5. Attention at 8×8; channels 64/128/128.

| Variant | Change | Params |
|---|---|---|
| `swl` | as above | 2,723,333 |
| `swl_wide` | channels 80/160/160 | 3,738,757 |
| `swl_p3` | passes (1, 2, 2) | 2,724,358 |
| `swl_long` | `swl` continued from its 4k checkpoint to 8k steps | 2,723,333 |

---

## 6. Code and files

### This folder (`Desktop/BTP-Diffusion-dgx/`)
| Path | Content |
|---|---|
| `README.md` | overview |
| `context.md` / `eval.md` / `decision.md` | these three documents |
| `RESEARCH_NOTES.md` | the running project log |
| `code/dgx1.ipynb` | all DGX code in one readable notebook (`RUN = False` by default) |
| `code/arch_bench.py` | data stream, original U-Net, PathUNet, RoutedUNet, `train` / `train_multi`, FID/IS, `report` |
| `code/novel.py` | Haar transforms, spectral schedules, SWL, `train_multi` for the new kinds, DDIM per coefficient |
| `code/run_all.sbatch`, `code/novel.sbatch` | the two SLURM jobs |
| `code/*.sh` | helper scripts the jobs ran (`novel_train.sh`, `novel_eval.sh`, `improve.sh`, `improve_b.sh`) |
| `results/<kind>/<model>/` | `eval.json` (FID, IS, params, sampling time), `loss.csv` (mean loss per 500 steps), `train_done.json`, `samples_*.npy` (32 samples) |
| `results/report_all.txt`, `results/samples_top.png`, `results/logs/` | table, sample grid, logs |
| `colab/diffusionRedo.ipynb` | the Colab notebook (83 cells) |

### Elsewhere
- `Desktop/diffusion/`: the original project (Colab notebooks, papers, `dgx_results/`).
- DGX: `~/arch_bench/` (code + checkpoints, 1.3 GB), `~/diffusion-3/` (older work, `cifar10_airplanes.npz`).
- Google Drive (Colab): `MyDrive/diffusion_cache/` with `checkpoints/`, `checkpoints_blur/`, `checkpoints_path_unet/`, `checkpoints_path_unet_blur/`, `checkpoints_routed_unet/`, `fid_reference_airplanes.npz`. **These paths must not change.**

---

## 7. Infrastructure

### Colab
T4 GPU, TensorFlow/Keras 3, Python 3.13. Dataset and checkpoints cached on Drive; every training cell resumes from checkpoints.

### Department DGX A100
| Item | Value |
|---|---|
| Address | `10.206.5.228`, SSH only, campus network |
| Login | `ssh dgx` (alias in `~/.ssh/config`, user `asrivastava`, key `~/.ssh/id_ed25519` loaded from the macOS Keychain) |
| Hardware | 8 × A100 40 GB, 2 × AMD EPYC 7742 (128 cores), 1 TB RAM |
| Scheduler | SLURM, partition `gpu`; **`#SBATCH --mem=128G` is required and is also the maximum** |
| Environment | `source /raid/money/miniconda3/etc/profile.d/conda.sh && conda activate diffusion` (TF 2.21, Keras 3.12, Python 3.10) |
| Disk | `/raid` 100% full, home ~78 GB free: keep outputs small |
| Sharing | usually only 1 GPU free; other users may run on "your" GPU outside SLURM |

---

## 8. How the work developed

1. **Colab baseline:** the original U-Net on airplanes, Gaussian and blur pipelines, base width reduced 64 → 32 for the T4.
2. **Ladder idea:** the route-based `PathUNet` and 6 shapes (unet, ladder_step, ladder_jump, ladder_reverse, jump_down_step_up, w_net), trained and scored with FID/IS.
3. **Size-matched tests:** `ladder_step_b24` (equal params) and `unet_b40` (≈ ladder size).
4. **Brick problem found:** every path model lost by ~85 FID to the original U-Net in Gaussian diffusion and showed blocky samples, but only ~8 FID in blur. Hypothesis: 2×2 downsampling + thin full-resolution layers hurt noise prediction.
5. **Blur run for all architectures** (Colab, identical stream) and the **noise-routed U-Net** design.
6. **DGX run (2026-11-06):** one benchmark script, identical stream, v2 bricks, routed U-Net, Gaussian + blur for 9 models.
7. **New pipeline and architecture:** spectral wavelet schedules and SWL.
8. **Improvement round:** `swl_wide`, `swl_p3`, `swl_long`.

### Problems hit (and fixes)
| Problem | Cause | Fix |
|---|---|---|
| Checkpoint shape mismatch on Colab | old base-64 checkpoints on Drive | moved to `checkpoints_base64/` |
| `train_step` returned `None` / Adam has no `_momentums` | Keras 3 builds the optimizer lazily, failing inside `tf.function` | `optimizer.build(...)` up front; `@tf.function(autograph=False)` |
| `property 'path' ... has no setter` | Keras reserves `.path` | renamed to `self.levels` |
| Inception OOM (batch 250 at 299×299) | GPU memory | Inception batch 50 (Colab) / 25–100 (DGX) |
| Models saw different batches (first Colab path runs) | a shared data iterator | step-seeded streams (`stream_batch(step)`) |
| DGX job 9915: 19 of 20 processes OOM | another user's process (39 GB) on the GPU SLURM gave us | one process per kind, all models round-robin |
| SSH denied from Claude's shell | username typo (`asrivastva`) and a passphrase-protected key | fixed the username; `ssh-add --apple-use-keychain` |
