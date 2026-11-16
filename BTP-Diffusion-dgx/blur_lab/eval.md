# Evaluation: blur-diffusion prototypes (16 Nov 2026)

Setup and every formula: **`context.md`**. Conclusions: **`decision.md`**. Next steps: **`TODO_NEXT.md`**.

**Protocol (micro):** 16×16 CIFAR-10 airplanes, original `UNet` at base 8 (224k params), 3,000 steps, batch 64, identical
training stream for every run; 1,000 samples with the same starting noise, 40 sampling steps;
**KID×1000** (main, lower is better) and **FID75** (secondary) on Inception-V3 features at 75×75 vs the 5,000 real images.
Reference points: real vs real KID **0.22** · real blurred by 1 px **107** · uniform noise **854**.
One seed per run: KID differences under ~10 points are within noise.

Samplers: `cold`, `ddim` (old pixel process only) · `eta0` = per-frequency DDIM · `eta1` = stochastic (ancestral) ·
`eta0.5` · `…q2` / `q3` / `q1.5` = step spacing `t_i = T·(i/n)^p` (more steps at low noise; uniform is p = 1).

---

## 1. All results (sorted by KID)

| Run | Process | Target / loss | Sampler | **KID×1000** | FID75 |
|---|---|---|---|---|---|
| **`she1.25_x0_hf0.01`** | **stochastic heat eq.** | **x0 + boost β=0.01** | **η=1, q2** | **97.5** | **237.4** |
| `she1.25_x0` | stochastic heat eq. | x0 | η=1, q2 | 103.2 | 247.9 |
| `gauss_x0_hf0.01` | Gaussian | x0 + boost β=0.01 | η=1, q2 | 103.2 | 248.3 |
| `lin1.25_x0_hf0.01` | exact blur | x0 + boost β=0.01 | η=1, q3 | 105.8 | 254.6 |
| `lin1.25_x0_hf0.01` | exact blur | x0 + boost β=0.01 | η=1, q2 | 106.2 | 254.7 |
| `gauss_x0` | Gaussian | x0 | η=1, q2 | 108.9 | 257.3 |
| `lin1.25_x0` | exact blur | x0 | η=1, q2 | 110.0 | 259.4 |
| `lin2.5_x0_hf0.01` | exact blur, 2× width | x0 + boost | η=1, q2 | 110.5 | 260.4 |
| `lin1.25_x0_hf0.01` | exact blur | x0 + boost | η=1, q1.5 | 110.8 | 258.4 |
| `wien0.5_x0` | Wiener blur | x0 | η=1, q2 | 111.7 | 263.4 |
| `lin1.25_x0_hf0.03` | exact blur | x0 + boost β=0.03 | η=1, q2 | 127.7 | 277.7 |
| `lin1.25_x0_hf0.01` | exact blur | x0 + boost | η=1 | 129.6 | 281.3 |
| `lin1.25_x0` | exact blur | x0 | η=1 | 137.1 | 291.7 |
| `lin1.25_x0` | exact blur | x0 | η=0, q2 | 138.3 | 292.2 |
| `gauss_x0_hf0.01` | Gaussian | x0 + boost | η=0 | 146.2 | 299.9 |
| `lin1.25_x0_hf0.01` | exact blur | x0 + boost | η=0 | 147.6 | 301.9 |
| `she1.25_x0_hf0.01` | stochastic heat eq. | x0 + boost | η=0 | 150.5 | 301.6 |
| `lin1.25_x0` | exact blur | x0 | η=0.5 | 149.9 | 305.0 |
| `gauss_x0` | Gaussian | x0 | η=0 | 151.8 | 307.4 |
| `lin1.25_x0` | exact blur | x0 | η=0 | 153.9 | 309.7 |
| `lin2.5_x0_hf0.01` | exact blur, 2× width | x0 + boost | η=0 | 154.0 | 308.8 |
| `old_x0` | old pixel blur | x0 | **noise re-estimation** | 154.2 | 309.6 |
| `she1.25_x0` | stochastic heat eq. | x0 | η=0 | 156.3 | 311.4 |
| `wien0.5_x0` | Wiener blur | x0 | η=0 | 156.6 | 314.2 |
| `lin1.25_x0_hf0.03` | exact blur | x0 + boost β=0.03 | η=0 | 162.5 | 321.8 |
| `lin1.25_x0_suw` | exact blur | x0 + spectral uncertainty weighting | η=0 | 173.1 | 325.4 |
| `lin1.25_u_suw` | exact blur | u + spectral uncertainty weighting | η=0 | 187.8 | 341.9 |
| `lin1.25_u` | exact blur | u | η=0 | 201.7 | 355.6 |
| `gauss_eps` | Gaussian | eps | η=0 | 208.7 | 357.6 |
| `old_x0` | old pixel blur | x0 | **cold** (original) | 219.6 | 379.6 |
| `gauss_eps` | Gaussian | eps | η=1 | 252.0 | 396.0 |
| `gauss_eps` | Gaussian | eps | η=1, q2 | 258.5 | 402.4 |
| `lin1.25_u0.5` | exact blur | u^0.5 | η=0 | 276.1 | 417.3 |

Full machine-generated table: `micro_report.txt` (`python blur_micro.py report`).

### 32×32 check (first attempt, before the laptop got too hot)
Old pipeline, base-16 U-Net, 3,000 steps, 2,000 samples, 50 steps, standard Inception FID:
**cold sampler FID 193.4 → noise-re-estimating sampler FID 183.5** (same model). Same direction as the micro result.

---

## 2. Effect of each change (KID×1000)

| Change | From → to | Effect |
|---|---|---|
| **Cold sampler → noise re-estimation** (same old model) | 219.6 → 154.2 | **−30%** |
| Old zero-padded pixel blur → exact DCT blur (x0, deterministic) | 154.2 → 153.9 | none |
| **Deterministic → stochastic η = 1 + quadratic steps** (exact blur, x0) | 153.9 → 110.0 | **−29%** |
| … η = 1 alone / quadratic steps alone | 153.9 → 137.1 / 138.3 | −11% / −10% |
| Quadratic → cubic / 1.5-power spacing (best model) | 106.2 → 105.8 / 110.8 | plateau at p ≈ 2–3 |
| **Additive spectral boost β = 0.01** | exact blur: 153.9 → 147.6 · 137.1 → 129.6 · 110.0 → 106.2; Gaussian: 151.8 → 146.2 · 108.9 → 103.2; heat eq.: 156.3 → 150.5 · 103.2 → 97.5 | **−3 to −6% in all 7 pairs** |
| Boost β = 0.01 → 0.03 | 106.2 → 127.7 | too strong (worse) |
| Gaussian blur width → Wiener blur (x0) | 153.9 → 156.6 · 110.0 → 111.7 | tie (and nothing to tune) |
| Exact blur → stochastic heat equation (x0) | 153.9 → 156.3 · 110.0 → **103.2** | tie deterministic, best with η = 1 |
| **Mild blur vs no blur** (x0): Gaussian → exact blur | 151.8 → 153.9 · 108.9 → 110.0 | **none: blur at this strength is neutral** |
| Same sampler on Gaussian **eps** | 208.7 → 252.0 (η=1) → 258.5 (η=1, q2) | **hurts eps models** |
| Gaussian eps → Gaussian x0 (deterministic) | 208.7 → 151.8 | −27% (x0 is more robust for a tiny model) |
| x0 → u target (exact blur) | 153.9 → 201.7 | worse (colour casts) |
| u → u + spectral uncertainty weighting | 201.7 → 187.8 | −7% |
| x0 → x0 + spectral uncertainty weighting | 153.9 → 173.1 | worse (colour bands switched off) |
| u → u^0.5 | 201.7 → 276.1 | worse (grainy) |
| Exact blur → stronger blur (2× width), both + boost | 106.2 → 110.5 | no gain |
| **Gaussian → stochastic heat equation** (x0, η=1, q2) | 108.9 → 103.2 · with boost 103.2 → 97.5 | **−5% in both pairs** (only blur-type process that beats no blur) |
| **Old pipeline as it was → best formula** | **219.6 → 97.5** | **−56%** |
| **Gaussian eps (DDIM) → best formula** | **208.7 → 97.5** | **−53%** |

---

## 3. What the samples look like
Every column starts from the same noise (for η = 1 the path then differs, so columns don't line up with deterministic rows).

- `micro_grid_best.png`: old cold · Gaussian eps · Gaussian x0 + boost · exact blur + boost · **stochastic heat eq. + boost** (all three η=1, q2).
  The three bottom rows show **recognisable airplane-like scenes** (light/dark aircraft shapes on sky and ground, natural colours);
  the old cold row is washed-out grey with streaks; Gaussian eps is covered in coloured speckle.
- `micro_grid_5.png`: one model (`lin1.25_x0`) with 5 samplers: cold-like grey → sharper with η=1 / q2 → sharpest with η=1 + q2.
- `micro_grid_1.png`: Gaussian eps vs old blur (cold) vs old blur (noise re-estimation).
- `micro_grid_2.png`: u target = colour casts.  `micro_grid_3.png`: old vs exact DCT blur = nearly identical.
- `micro_grid_4.png`: x0 vs u + SUW vs u vs u^0.5 (grainy).

---

## 4. Learned spectral weights (`suw_weights.json`)

Relative weight per radial DCT band (|k| = 0, 1, 2–3, 4–7, 8–15, 16+), mean = 1.

**u target + SUW**: the network learned a **coarse-to-fine sweep** by itself:
| t | DC | 1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.70 | 0.91 | 1.06 | 1.28 | 1.31 |
| 250 | 0.64 | 0.94 | 1.25 | **1.41** | 0.98 | 0.77 |
| 500 | 0.83 | 1.18 | **1.32** | 1.08 | 0.81 | 0.78 |
| 950 | 1.06 | 1.00 | 1.01 | 0.97 | 0.97 | 0.98 |

**x0 target + SUW**: it switched the colour bands off:
| t | DC | 1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.06 | 0.15 | 0.50 | 1.45 | 3.84 |
| 500 | 0.01 | 0.05 | 0.11 | 0.27 | 1.09 | 4.48 |
| 950 | 0.00 | 0.01 | 0.03 | 0.13 | 0.95 | 4.88 |

---

## 5. Caveats
- **Tiny scale:** 224k params, 16×16, 3,000 steps; every model is far from converged. Targets that need more capacity (eps, u)
  probably look worse here than they would at full scale. The x0 advantage is likely a small-model effect.
- **One seed, 1,000 samples:** the top 4 formulas (103–110) are within noise of each other; a second seed is needed to rank them.
- **KID/FID75 are not comparable with the DGX FIDs.**
- **The blur here is mild** (σ ≤ 1.25 px at 16×16, the old schedule); stronger blur is tested in §6.

---

## 6. Last runs of the day (controls)

| Question | Runs | Answer |
|---|---|---|
| Is the boost blur-specific? | `gauss_x0` vs `gauss_x0_hf0.01`: 151.8 → 146.2 (η=0), 108.9 → 103.2 (η=1,q2) | **No, it helps any x0 model** |
| Does stronger blur help once the sampler is right? | `lin1.25_x0_hf0.01` 106.2 vs `lin2.5_x0_hf0.01` 110.5 | **No** |
| Does the heat equation keep its edge with the boost? | `she1.25_x0` 103.2 → `she1.25_x0_hf0.01` **97.5**; Gaussian + boost 103.2 | **Yes, best overall, ~5.5 KID ahead of Gaussian in both pairs** |
