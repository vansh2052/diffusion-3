# Evaluation: blur-diffusion prototypes (16 Nov 2026)

Setup and every formula: **`context.md`**. Conclusions: **`decision.md`**. Next steps: **`TODO_NEXT.md`**.

**Protocol (micro):** 16×16 CIFAR-10 airplanes, original `UNet` at base 8 (224k params), 3,000 steps, batch 64, identical
training stream for every run; 1,000 samples with the same starting noise, 40 sampling steps;
**KID×1000** (main, lower is better) and **FID75** (secondary) on Inception-V3 features at 75×75 vs the 5,000 real images.
Reference points: real vs real KID 0.22 · real blurred by 1 px KID 107 · uniform noise KID 854.
One seed per run. KID differences of a few points are within noise; treat gaps of ≥ 10 as meaningful.

---

## 1. All results so far (sorted by KID)

| # | Run | Process | Target | Loss / extra | Sampler | **KID×1000** | FID75 |
|---|---|---|---|---|---|---|---|
| 1 | `lin1.25_x0` | exact DCT blur | x0 | – | **η = 1** (stochastic) | **137.1** | 291.7 |
| 2 | `lin1.25_x0` | exact DCT blur | x0 | – | **quadratic steps** (η = 0) | **138.3** | 292.2 |
| 3 | `lin1.25_x0` | exact DCT blur | x0 | – | η = 0.5 | 149.9 | 305.0 |
| 4 | `lin1.25_x0` | exact DCT blur | x0 | – | η = 0 (DDIM) | 153.9 | 309.7 |
| 5 | `old_x0` | old pixel blur | x0 | – | **noise re-estimation** ("ddim") | 154.2 | 309.6 |
| 6 | `wien0.5_x0` | Wiener blur γ=0.5 | x0 | – | η = 0 | 156.6 | 314.2 |
| 7 | `lin1.25_x0_suw` | exact DCT blur | x0 | spectral uncertainty weighting | η = 0 | 173.1 | 325.4 |
| 8 | `lin1.25_u_suw` | exact DCT blur | u | spectral uncertainty weighting | η = 0 | 187.8 | 341.9 |
| 9 | `lin1.25_u` | exact DCT blur | u | – | η = 0 | 201.7 | 355.6 |
| 10 | `gauss_eps` | Gaussian | eps | – | η = 0 | 208.7 | 357.6 |
| 11 | `old_x0` | old pixel blur | x0 | – | **cold** (original) | 219.6 | 379.6 |
| 12 | `lin1.25_u0.5` | exact DCT blur | u^0.5 | – | η = 0 | 276.1 | 417.3 |

Still running when this file was written (see `done_micro.txt` / `report` for the final table): `lin1.25_x0_hf0.01`
(additive spectral boost), `lin1.25_x0` with η = 1 + quadratic steps, `gauss_x0`, `she1.25_x0`, `lin1.25_x0_sc`.

### 32×32 check (first attempt, before the laptop got too hot)
Old pipeline, base-16 U-Net, 3,000 steps, 2,000 samples, 50 steps, standard Inception FID:
**cold sampler FID 193.4 → noise-re-estimating sampler FID 183.5** (same model). Same direction as the micro result.

---

## 2. Effect of each change (micro, KID×1000)

| Change | From → to | Effect |
|---|---|---|
| Cold sampler → noise re-estimation (same old model) | 219.6 → 154.2 | **−30%** |
| Old zero-padded pixel blur → exact DCT blur (x0, new sampler) | 154.2 → 153.9 | none |
| Deterministic DDIM → stochastic η = 1 | 153.9 → 137.1 | **−11%** |
| Uniform → quadratic step spacing (η = 0) | 153.9 → 138.3 | **−10%** |
| Gaussian blur width → data-adaptive Wiener blur (x0) | 153.9 → 156.6 | none (tie, but no width to tune) |
| x0 target → u target (exact blur) | 153.9 → 201.7 | +31% (worse) |
| u → u + spectral uncertainty weighting | 201.7 → 187.8 | −7% |
| x0 → x0 + spectral uncertainty weighting | 153.9 → 173.1 | +12% (worse) |
| u → u^0.5 | 201.7 → 276.1 | +37% (worse) |
| **Gaussian eps (DDIM) → best blur formula (exact blur, x0, η = 1)** | **208.7 → 137.1** | **−34%** |
| **Old pipeline as it was (cold) → best blur formula** | **219.6 → 137.1** | **−38%** |

---

## 3. What the samples look like
Grids in this folder; every column starts from the same noise.

- `micro_grid_1.png`: **Gaussian eps** = rough layouts covered in coloured high-frequency speckle (the tiny model cannot
  remove fine noise). **Old cold** = washed-out, grey, very blurry, horizontal streaks. **Old + noise re-estimation** = clearly
  more contrast and detail, same streaks.
- `micro_grid_2.png`: **u target** = less speckle than Gaussian but strong colour casts (green / purple / pink).
- `micro_grid_3.png`: **old vs exact DCT blur** (both x0, new sampler) = almost identical images, streaks in both
  (so the streaks are the small x0 model averaging horizontal structures, not the zero padding).
- `micro_grid_4.png`: **x0** (natural colours, smooth, blurry) vs **u + SUW**, **u** (colour casts) and **u^0.5** (muted colours
  but grainy, blotchy texture).

---

## 4. Learned spectral weights (from `suw_weights.json`)

Relative weight per radial DCT band (|k| = 0, 1, 2–3, 4–7, 8–15, 16+), mean = 1:

**u target + SUW** (learned a coarse-to-fine sweep):
| t | DC | 1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.70 | 0.91 | 1.06 | 1.28 | 1.31 |
| 250 | 0.64 | 0.94 | 1.25 | **1.41** | 0.98 | 0.77 |
| 500 | 0.83 | 1.18 | **1.32** | 1.08 | 0.81 | 0.78 |
| 950 | 1.06 | 1.00 | 1.01 | 0.97 | 0.97 | 0.98 |

**x0 target + SUW** (switched the colour bands off):
| t | DC | 1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.06 | 0.15 | 0.50 | 1.45 | 3.84 |
| 500 | 0.01 | 0.05 | 0.11 | 0.27 | 1.09 | 4.48 |
| 950 | 0.00 | 0.01 | 0.03 | 0.13 | 0.95 | 4.88 |

With the u target, the network discovered on its own that the important frequencies move from coarse (high noise) to fine
(low noise). With the x0 target, uncertainty weighting gives up on the bands with the largest absolute error, the DC/colour
bands, which is why its samples got worse.

---

## 5. Caveats
- **Tiny scale.** 224k params, 16×16, 3,000 steps: every model is far from converged. Formulas that need more capacity or
  training (u, eps) may look worse here than they would on the DGX.
- **One seed per run, 1,000 samples.** Gaps under about 10 KID points need a second seed before trusting them.
- **KID/FID75 are not comparable with the DGX FIDs** (different resolution, feature size and sample count).
- **The sampler gains are measured on one model** (`lin1.25_x0`). They should be re-checked on the other models.
