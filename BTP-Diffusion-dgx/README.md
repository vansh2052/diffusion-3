# BTP: Diffusion Architectures on the DGX A100

**Goal:** find a denoising network that beats a standard U-Net for diffusion models on CIFAR-10 airplanes
(5,000 images, 32×32), and test new noise schedules. Run on the department DGX A100 on 2026-11-06.

**Headline result:** the new **SWL (Shared-Weight Wavelet Ladder)** architecture reaches **FID 92.8**, against **128.5** for the
original U-Net after the same 4,000 training steps (**−28%**), with 2.7M parameters. Trained for 8,000 steps it reaches
**FID 72.0, IS 4.42** (real airplanes: IS 4.91).

---

## Folder layout

```
BTP-Diffusion-dgx/
├── README.md               ← this file
├── context.md              ← goal, data, setup, every model and pipeline, code, infrastructure, history
├── eval.md                 ← evaluation protocol and every result (DGX + Colab)
├── decision.md             ← every decision, why it was made, and how it turned out; next steps
├── RESEARCH_NOTES.md       ← full project log: Colab experiments, every result, bugs, next steps
├── code/
│   ├── dgx1.ipynb          ← all DGX code in one readable notebook (with explanations and results)
│   ├── arch_bench.py       ← main benchmark (data stream, all architectures, training, FID/IS, report)
│   ├── novel.py            ← new pipeline: spectral wavelet schedules + SWL architecture
│   ├── run_all.sbatch      ← SLURM job: main comparison (9 models × Gaussian/blur)
│   ├── novel.sbatch        ← SLURM job: new pipeline + improvement round
│   ├── novel_train.sh, novel_eval.sh, improve.sh, improve_b.sh   ← helper scripts the jobs ran
│   └── README.md           ← how to run on the DGX
├── results/
│   ├── report_all.txt      ← the full results table
│   ├── samples_top.png     ← samples from the best models (same starting noise in every row)
│   ├── gauss/  blur/  spec_eq/  spec_c2f/      ← one folder per pipeline
│   │   └── <model>/eval.json, loss.csv, train_done.json, samples_*.npy (32 samples, uint8)
│   └── logs/               ← training and evaluation logs
└── colab/
    └── diffusionRedo.ipynb ← the earlier Colab (T4) notebook these experiments grew out of
```

---

## Setup

| | |
|---|---|
| Data | CIFAR-10, airplane class only (5,000 training images), scaled to [-1, 1] |
| Diffusion | 1,000 steps, cosine schedule |
| Training | batch 128, Adam 2e-4, EMA 0.999, random flips, **4,000 steps** (≈100 epochs) |
| Fairness | **every model sees the identical training stream**: images, flips, timesteps and noise depend only on the step number |
| Evaluation | **5,000 samples per model**, same starting noise (seed 1234), DDIM 100 steps (cold sampler for blur); FID and IS with Keras InceptionV3 |
| Hardware | 1 × A100 40 GB (shared DGX), SLURM; all models trained round-robin in one process (identical step counts) |

**Pipelines**
- **gauss:** standard Gaussian diffusion, the model predicts the noise (ε)
- **blur:** blur + noise degradation (cold diffusion), the model predicts the clean image (x0)
- **spec_eq / spec_c2f (new):** per-frequency-band noise schedules in an orthonormal Haar wavelet domain (see below)

---

## Architectures

| Name | Description | Params |
|---|---|---|
| `original_unet` | The hand-written U-Net from the Colab notebook: 32→16→8, attention at 8×8 | 1.66M |
| `unet_v1`, `ladder_v1` | `PathUNet` with the Colab "bricks" (2×2 non-overlapping downsampling, 4 levels, attention only at 4×4): plain U-Net and the stepping ladder | 1.54M / 2.35M |
| `unet_v2`, `ladder_v2`, `ladder_rev_v2` | `PathUNet` with **fixed bricks** (3×3 stride-2 downsampling, 64/128/128 ch, stop at 8×8 with attention): plain, ladder `[0,1,0,1,2,1,0]`, reverse ladder `[0,1,2,1,0,1,0]` | 2.37M / 3.21M / 3.21M |
| `routed_dense / fixed / learned` | Noise-routed U-Net: two blocks per level, **which blocks run depends on the noise level** (all / hand-designed table / learned gates) | 3.94M |
| **`swl`** (new) | **Shared-Weight Wavelet Ladder**, see below | **2.72M** |
| `swl_wide`, `swl_p3`, `swl_long` | SWL wider (80/160/160 ch), with a third shared pass, or trained to 8k steps | 3.74M / 2.72M / 2.72M |

### SWL: Shared-Weight Wavelet Ladder
1. **Lossless wavelet resampling.** Down = orthonormal Haar transform (2×2 → 4 channels) + 1×1 conv; up = 1×1 conv + inverse Haar. Nothing is lost when changing resolution, unlike strided convs.
2. **Ladder passes with shared weights.** Pass 1 goes 32→16→32, pass 2 goes 32→16→8→16→32, and both **reuse the same per-level blocks** (with a learned pass embedding). You get the ladder's extra refinement without its extra parameters.
3. **Noise-dependent pass gates.** Each pass's update is scaled by `g_p(t) = 1 + tanh(Dense(t_emb))`, starting at 1. The network learns how much each pass matters at each noise level.
4. **adaLN-Zero conditioning.** GroupNorm, then scale/shift from the timestep (+ pass) embedding, with a zero-initialised output conv (as in DiT).
5. Attention at 8×8.

### Spectral wavelet schedules (new)
A 2-level Haar transform splits each image into a coarse 8×8 band, a mid-detail band and a fine-detail band. Each band gets
the cosine log-SNR **plus a shift computed from the band's power on real airplanes** (the coarse band has ≈13× the mean power):
- `spec_eq`: all bands cross signal = noise at the same time
- `spec_c2f`: exaggerated coarse-to-fine (fine detail destroyed even earlier, coarse layout survives longer)

Haar is orthonormal, so the noise stays white in pixel space and DDIM works per coefficient.

---

## Results (FID on 5,000 samples; lower is better)

### Gaussian diffusion
| Model | Params | Final loss | **FID** | IS |
|---|---|---|---|---|
| **swl_long** (8k steps) | 2.72M | 0.0474 | **72.0** | **4.42** |
| swl_wide | 3.74M | 0.0501 | 87.4 | 3.90 |
| **swl** | 2.72M | 0.0513 | **92.8** | 3.74 |
| swl_p3 | 2.72M | 0.0522 | 100.5 | 3.47 |
| original_unet | 1.66M | 0.0559 | 128.5 | 3.17 |
| ladder_v2 | 3.21M | 0.0557 | 128.9 | 2.93 |
| ladder_rev_v2 | 3.21M | 0.0569 | 146.5 | 2.80 |
| unet_v2 | 2.37M | 0.0577 | 149.9 | 2.78 |
| routed_learned (all blocks) | 3.95M | 0.0602 | 162.7 | 2.58 |
| ladder_v1 | 2.35M | 0.0612 | 176.3 | 2.50 |
| routed_dense | 3.94M | 0.0624 | 192.8 | 2.38 |
| unet_v1 | 1.54M | 0.0645 | 209.4 | 2.25 |
| routed_fixed (12% block compute) | 3.94M | 0.0632 | 228.3 | 2.11 |
| routed_learned, pruned (38% compute) | 3.95M | 0.0602 | 292.9 | 2.02 |

### Blur + noise diffusion
| Model | FID | IS |
|---|---|---|
| **ladder_rev_v2** | **136.6** | 2.76 |
| ladder_v2 | 141.3 | 2.78 |
| unet_v2 | 151.8 | 2.88 |
| ladder_v1 | 154.7 | 2.68 |
| original_unet | 160.4 | 2.68 |
| unet_v1 | 162.6 | 2.60 |
| routed_dense | 165.2 | 2.71 |
| routed_learned (pruned / all) | 179.0 / 180.2 | 2.55 / 2.53 |
| routed_fixed | 224.1 | 2.16 |

### Spectral schedules
| Model | standard | spec_eq | spec_c2f |
|---|---|---|---|
| swl | **92.8** | 111.7 | 113.2 |
| original_unet | 128.5 | 149.9 | 182.0 |
| ladder_v2 | 128.9 | 142.4 | 196.3 |

Sample images from the best models: `results/samples_top.png`.

---

## Conclusions
1. **SWL is the best architecture**, 28% lower FID than the original U-Net at equal training steps, with a mid-sized parameter count. Longer training helps it most (FID 72 at 8k steps, still improving). Its samples are visibly smoother and more coherent.
2. **The downsampling ("bricks") hypothesis is confirmed.** Replacing 2×2 non-overlapping downsampling with 3×3 stride-2 and widening the full-resolution layers took the plain U-Net from FID 209 to 150, and the ladder from 176 to 129.
3. **The ladder helps on good bricks** (150 → 129 in Gaussian). The reverse ladder is the best model in blur diffusion (136.6).
4. **Widening SWL helps a little** (87.4); **a third shared pass hurts** (100.5).
5. **The spectral wavelet schedules hurt every architecture** at this training length. SWL is the most robust to them.
6. **Routing blocks by noise level did not work** at this scale. Pruning learned gates collapsed quality.

## Caveats
- One training seed per model. Differences of a few FID points are within noise; the SWL gap (~36 points) is not.
- `swl_long` trained for 8k steps, the others for 4k. Plain `swl` at 4k already beats every other model.
- SWL is about 2–3× slower to sample than the original U-Net.
- All models are still undertrained (real-data FID would be ~0); the ranking may change with much longer training.

## Next steps
- Train SWL and the original U-Net equally long (16k–30k steps), with 2–3 seeds.
- Ablate SWL's components one at a time (shared weights, Haar resampling, adaLN-Zero, gates) to find what gives the gain.
- Run SWL on blur + noise and with v-prediction; compare at equal sampling compute.

## How to reproduce
See `code/README.md`. In short, on the DGX: copy `code/` to `~/arch_bench`, `echo 4000 > target_steps`, then `sbatch run_all.sbatch` and `sbatch novel.sbatch`. Or open `code/dgx1.ipynb`, set `RUN = True` and run it in a GPU session.
