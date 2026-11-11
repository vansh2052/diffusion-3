# Evaluation: protocol and every result

How models were scored, and every number from both the Colab and DGX runs. Setup and definitions: **`context.md`**.
What the numbers led to: **`decision.md`**.

---

## 1. Evaluation protocol

| Item | Value |
|---|---|
| Metric 1 | **FID** (Fréchet Inception Distance), lower is better. Distance between the mean/covariance of Inception-V3 pool features (2048-d) of generated vs real images |
| Metric 2 | **IS** (Inception Score), higher is better; 10 splits, mean ± std |
| Feature network | Keras `InceptionV3(weights="imagenet")`, images resized bilinearly to 299×299, scaled to [-1, 1] |
| Reference | all 5,000 real training airplanes (cached as `fid_reference_airplanes.npz`); **real airplanes: IS 4.91 ± 0.25** |
| Samples | **5,000 per model**, the **same starting noise** for every model (seed 1234) |
| Sampler | DDIM η = 0, 100 evenly spaced steps, x0 clipped to [-1, 1] (gauss); cold-diffusion restart sampler, 100 steps (blur); DDIM per Haar coefficient (spec_*) |
| Weights | EMA weights |
| Training loss | mean noise-prediction MSE (or x0 MSE for blur) over each 500-step block |

### How to read the numbers
- **FID is not linear.** Points at FID 200 are "cheaper" than points at FID 30. Only compare numbers with the same sample count (5,000 here).
- **IS on a single class tops out low** (real data = 4.91), so FID is the main metric.
- **Loss and FID mostly agree but not always** (e.g. `ladder_reverse` tied the U-Net on loss but was worst on FID).
- **Every model is undertrained** (FID 70–230 vs ~0 for real data), so rankings show which designs learn fastest at this budget.
- **One seed per model.** Gaps of a few FID points can be noise; the large gaps (≥ 15) are reliable.
- **Absolute values differ from papers** (Keras InceptionV3, single class, 5k samples); compare within this project only.

### Consistency check (DGX vs Colab, same models and protocol)
| Model | Colab FID | DGX FID |
|---|---|---|
| original_unet (Gaussian) | 124.5 | 128.5 |
| unet_v1 (Gaussian) | 209.4 | 209.4 |

---

## 2. DGX results (2026-11-06): identical stream, 4,000 steps

### 2.1 Gaussian diffusion
| Model | Params | ms/step* | Final loss | **FID** | IS | Block compute | Sampling time (5k) |
|---|---|---|---|---|---|---|---|
| **swl_long** (8k steps) | 2,723,333 | – | 0.0474 | **72.00** | **4.42** | – | 259 s |
| swl_wide | 3,738,757 | 85 | 0.0501 | 87.42 | 3.90 | – | 318 s |
| **swl** | 2,723,333 | 85 | 0.0513 | **92.84** | 3.74 | – | 372 s |
| swl_p3 | 2,724,358 | 90 | 0.0522 | 100.48 | 3.47 | – | 352 s |
| original_unet | 1,663,235 | 36 | 0.0559 | 128.45 | 3.17 | – | 73 s |
| ladder_v2 | 3,207,043 | 51 | 0.0557 | 128.91 | 2.93 | – | 239 s |
| ladder_rev_v2 | 3,207,043 | 50 | 0.0569 | 146.49 | 2.80 | – | 152 s |
| unet_v2 | 2,365,251 | 36 | 0.0577 | 149.86 | 2.78 | – | 185 s |
| routed_learned (all blocks) | 3,945,843 | 52 | 0.0602 | 162.68 | 2.58 | 100% | 64 s |
| ladder_v1 | 2,353,059 | 49 | 0.0612 | 176.33 | 2.50 | – | 103 s |
| routed_dense | 3,937,635 | 50 | 0.0624 | 192.77 | 2.38 | 100% | 205 s |
| unet_v1 | 1,544,675 | 30 | 0.0645 | 209.42 | 2.25 | – | 44 s |
| routed_fixed | 3,937,635 | 128 | 0.0632 | 228.28 | 2.11 | 12% | 69 s |
| routed_learned (pruned, gate < 0.1) | 3,945,843 | 52 | 0.0602 | 292.90 | 2.02 | 38% | 39 s |

\* Training time per step while sharing one A100 with other models (relative cost only). Sampling times also vary with GPU contention.

### 2.2 Blur + noise diffusion
| Model | Params | Final loss | **FID** | IS | Block compute |
|---|---|---|---|---|---|
| **ladder_rev_v2** | 3,207,043 | 0.0561 | **136.56** | 2.76 | – |
| ladder_v2 | 3,207,043 | 0.0561 | 141.31 | 2.78 | – |
| unet_v2 | 2,365,251 | 0.0565 | 151.79 | 2.88 | – |
| ladder_v1 | 2,353,059 | 0.0566 | 154.66 | 2.68 | – |
| original_unet | 1,663,235 | 0.0560 | 160.39 | 2.68 | – |
| unet_v1 | 1,544,675 | 0.0573 | 162.60 | 2.60 | – |
| routed_dense | 3,937,635 | 0.0574 | 165.16 | 2.71 | 100% |
| routed_learned (pruned) | 3,945,843 | 0.0577 | 178.95 | 2.55 | 21% |
| routed_learned (all blocks) | 3,945,843 | 0.0577 | 180.20 | 2.53 | 100% |
| routed_fixed | 3,937,635 | 0.0583 | 224.13 | 2.16 | 12% |

### 2.3 Spectral wavelet schedules (FID / IS)
| Model | gauss (standard) | spec_eq | spec_c2f |
|---|---|---|---|
| swl | **92.84** / 3.74 | 111.70 / 3.18 | 113.19 / 3.32 |
| original_unet | 128.45 / 3.17 | 149.89 / 2.67 | 181.99 / 2.64 |
| ladder_v2 | 128.91 / 2.93 | 142.44 / 2.75 | 196.27 / 2.45 |

Final training losses (not comparable across schedules, since the task changes): spec_eq swl 0.0972, ladder_v2 0.1047,
original 0.1094; spec_c2f swl 0.0447, original 0.0486, ladder_v2 0.0488.

### 2.4 Training curves (loss every 500 steps, Gaussian)
Per-model `loss.csv` in `results/gauss/<model>/`. Early (step 500) vs late:

| Model | step 500 | step 3,500 | step 4,000 |
|---|---|---|---|
| original_unet | 0.1855 | 0.0568 | 0.0559 |
| unet_v1 | 0.2435 | 0.0662 | 0.0645 |
| ladder_v1 | 0.2602 | 0.0629 | 0.0612 |
| unet_v2 | 0.2082 | 0.0590 | 0.0577 |
| ladder_v2 | 0.2138 | 0.0571 | 0.0557 |
| ladder_rev_v2 | 0.2204 | 0.0582 | 0.0569 |
| routed_dense | 0.2805 | 0.0643 | 0.0624 |
| routed_fixed | 0.3376 | 0.0648 | 0.0632 |
| routed_learned | 0.2739 | 0.0616 | 0.0602 |
| swl | – | 0.0517 | 0.0513 |
| swl_long | – | – | 0.0513 → 0.0474 at 8k |

### 2.5 Samples
`results/samples_top.png`: rows = original_unet, ladder_v2, swl, swl_wide, swl_long (Gaussian), ladder_rev_v2 (blur);
columns = the same starting noise. SWL rows are smoother, with stronger contrast and more coherent aircraft shapes; the
original U-Net and ladder rows have blotchier texture; blur samples are soft (typical of x0-prediction).

---

## 3. Colab results (T4, earlier)

### 3.1 Gaussian, path models with v1 bricks, 3,900 steps (non-identical batches)
| Model | Params | Final loss | vs unet | FID | IS |
|---|---|---|---|---|---|
| **original UNet** | 1,663,235 | – | – | **124.53** | 3.14 |
| ladder_step | 2,353,059 | 0.0605 | −5.8% | 178.86 | 2.47 |
| w_net | 2,994,883 | 0.0623 | −3.0% | 191.13 | 2.36 |
| jump_down_step_up | 2,109,283 | 0.0621 | −3.3% | 194.10 | 2.31 |
| unet_b40 | 2,154,907 | 0.0633 | −1.4% | 203.55 | 2.22 |
| ladder_jump | 1,523,491 | 0.0652 | +1.6% | 209.04 | 2.16 |
| unet (path) | 1,544,675 | 0.0642 | – | 209.35 | 2.20 |
| ladder_step_b24 | 1,543,035 | 0.0657 | +2.3% | 227.72 | 2.09 |
| ladder_reverse | 1,523,491 | 0.0642 | 0.0% | 234.93 | 2.07 |

Training speed on the T4 (ms/step): unet 122, ladder_step 256, ladder_jump 170, ladder_reverse 170, jump_down_step_up 212,
w_net 218, ladder_step_b24 200, unet_b40 160.

### 3.2 Blur + noise, 150 epochs, identical stream (partial run)
| Model | FID | IS |
|---|---|---|
| ladder_step | 141.06 | 2.87 |
| original_unet | 141.18 | 2.83 |
| unet (path) | 149.27 | 2.75 |

### 3.3 What the Colab samples showed
- Gaussian: every path model (v1 bricks) had a **blocky mosaic texture**; the original U-Net did not.
- Blur: **no blockiness in any model** (same bricks) → the bricks hurt ε-prediction specifically.

---

## 4. Summary of findings
1. **SWL wins clearly:** FID 92.8 vs 128.5 (−28%) at equal steps; 72.0 at 8k steps; IS 4.42 vs real 4.91.
2. **The bricks matter more than the route:** v1 → v2 bricks: plain U-Net 209 → 150, ladder 176 → 129.
3. **The ladder helps on good bricks** (Gaussian 150 → 129; blur 152 → 141, reverse 137).
4. **At equal params (v1), stepping lost** (b24: 228 vs 209) and **jumping tied** (209 vs 209).
5. **Spectral schedules hurt** every model (+15 to +67 FID); SWL suffered least.
6. **Noise routing failed** (193–293 FID); pruning learned gates collapsed quality.
7. **SWL scaling:** wider +, third pass −, longer training ++.

## 5. Not yet evaluated
- SWL on blur + noise; SWL with v-prediction
- Several seeds per model; equal-length long training (16k–30k steps) for SWL vs the original U-Net
- Equal sampling-compute comparisons
- `wnet_v2` (built, not run) and `ladder_step_reverse` on v1 bricks
