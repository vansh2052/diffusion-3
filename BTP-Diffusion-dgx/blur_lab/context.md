# Context: blur-diffusion formulas (prototypes, 16 Nov 2026)

What this folder is, how the experiments are set up, and every formula that was tried. Results: **`eval.md`**.
What was concluded and why: **`decision.md`**.

---

## 1. Goal

Find a better **formula for blur diffusion**, meaning a better forward process (how images are blurred and noised), training
target, loss and sampler, starting from the pipeline used so far in the project:

- **Old pipeline ("cold" blur diffusion):** blur `x0` with a Gaussian of width `σ_b(t) = σ_max (1 − √ᾱ_t)` (17×17 kernel, zero
  padding), then mix with noise: `x_t = √ᾱ_t · Blur_t(x0) + √(1 − ᾱ_t) · z`. The network predicts `x0`. Sampling uses the Cold
  Diffusion update `x_{t−1} = x_t − D(x̂0, t) + D(x̂0, t−1)`, which reuses the **initial** noise `z` at every step.
- On the DGX this pipeline was clearly worse than Gaussian diffusion (FID 160 vs 128 for the same U-Net) and gave soft, blurry samples.

These are **prototypes** on a laptop (MacBook Air M2, CPU only). The large runs are planned for the DGX later.

---

## 2. Prototype setup ("micro" protocol)

The first attempt (32×32, base-16 U-Net, 8 threads) made the laptop too hot, so everything was scaled down:

| Item | Value |
|---|---|
| Data | CIFAR-10 airplanes (5,000 train / 1,000 test), **2×2 average-pooled to 16×16**, scaled to [-1, 1] |
| Model | the project's original `UNet` class at **base width 8** (8/16/32 channels, attention at the 4×4 bottleneck), **224,003 params** |
| Training | batch 64, Adam 4e-4, EMA 0.998, **3,000 steps** (≈ 38 epochs), random flips |
| Identical stream | every run sees the same images, flips, timesteps (stratified uniform) and noise at every step (seeded by step) |
| Diffusion | T = 1000, cosine schedule (as in the main project) |
| Sampling | 40 steps, the same starting noise for every model (seed 1234), x0 clipped to [-1, 1] |
| Metric | **KID×1000** (Kernel Inception Distance, unbiased, cubic kernel) and **FID75**, both on Inception-V3 pool features with the images resized to 75×75, 1,000 samples vs the 5,000 real training images |
| CPU limits | macOS efficiency cores only (`taskpolicy -b`), 4 TF threads, about 2.7 cores busy, no thermal warnings |

**Metric sanity checks (KID×1000):** real test vs real train **0.22**; real images blurred by 1 px **107**; uniform noise **854**.
FID75 with 1,000 samples has a floor of about 74 (real vs real), so **KID is the main number** and FID75 is secondary.
These numbers are only comparable within this protocol, not with the DGX FIDs.

---

## 3. The unifying framework

Every process tested is a **linear Gaussian diffusion that is diagonal in the 2-D DCT basis** (orthonormal DCT-II, 16×16):

```
x_t = IDCT( a[t,k] · DCT(x0) + s[t,k] · DCT(ε) ),   ε ~ N(0, I) in pixel space,   k = 2-D DCT frequency
```

- **Isotropic Gaussian diffusion:** `a = √ᾱ_t`, `s = √(1 − ᾱ_t)` at every frequency.
- **Blur:** a Gaussian blur of width σ with reflective boundaries multiplies frequency k by `exp(−λ_k σ²/2)`, where
  `λ_k = (π k_x/N)² + (π k_y/N)²`. So "blur, then add noise" is `a[t,k] = √ᾱ_t · exp(−λ_k τ_t)` with heat time `τ_t = σ_b(t)²/2`.
  (Checked: the DCT blur matches a real reflective pixel blur exactly for σ ≥ 1.5 px at 32×32.)

Because the process is Gaussian and diagonal, **the reverse step has a closed form**. That is DDIM generalised per frequency
(η = 0 deterministic, η > 0 stochastic):

```
x̂0, ε̂  ← decode(network output)               (then clip x̂0 to [-1,1] in pixels and set ε̂ = (X_t − a·X̂0)/s)
X_{t'} = a_{t'}·X̂0 + √(s_{t'}² − σ²)·ε̂ + σ·z,   σ² = η² · s_{t'}² · (s_t² − r² s_{t'}²)/s_t²,   r = a_t / a_{t'}
```

**Correctness test:** given a "perfect" network that outputs the true target, the sampler reconstructs the clean image to about
1e-6 for every process, target and η.

---

## 4. Everything that was varied

### 4.1 Forward processes
| Name | Formula | Notes |
|---|---|---|
| `gauss` | `a = √ᾱ`, `s = √(1−ᾱ)` | standard diffusion |
| `old` | pixel conv blur (9×9 kernel at 16 px, **zero padding**), `σ_b = 1.25 (1 − √ᾱ)` | the previous pipeline (2.5 px at 32×32 = 1.25 px at 16×16) |
| `lin<σ>` | exact DCT blur, `σ_b = σ_max (1 − √ᾱ)` | same schedule as `old`, reflective boundary, exact sampler |
| `sin<σ>` | `σ_b = σ_max sin(πt/2T)`, floor `d_min = 0.001` | Hoogeboom & Salimans (2022)-style schedule |
| `wien<γ>` (new) | `d_k(t) = (SNR_k / (1 + SNR_k))^γ`, `SNR_k = ᾱ P_k / (1 − ᾱ)` | **data-adaptive "Wiener blur"**: every frequency is attenuated by its own Wiener gain at the current noise level, so the blur removes what the noise already hides. No blur-width parameter |
| `she<σ>` (new) | `dX = (Δ − β/2) X dt + √β dW` per frequency | **stochastic heat equation**: blur and noise happen together, so injected noise is blurred too (coloured noise). Same signal `a` as `lin`, smaller high-frequency noise in the middle of the process |

### 4.2 What the network predicts
`P_k` = average power of the real images at DCT frequency k. At 16×16 it spans **29 (DC) to 0.0045 (finest)**: coarse
frequencies carry far more energy than fine ones (about 4×10⁵× at 32×32).

| Target | Definition | Implied weight on x0 error at frequency k |
|---|---|---|
| `eps` | the noise ε | `a²/s²` |
| `x0` | the clean image | `1` (fine detail almost ignored → blurry) |
| `v` | per-frequency v: `(a ε − s x0)/√(a² + s²)` | `(a² + s²)/s²` |
| `u` (new) | **per-frequency EDM preconditioning:** `(a√P ε − s x0/√P)/√(a²P + s²)`, so `x̂0 = Wiener(x_t) + c_out·F` | `a²/s² + 1/P_k` |
| `u<γ>` (new) | `u` with `P_k^γ` instead of `P_k` (γ = 1 → u, γ = 0 → v) | `a²/s² + P_k^(−γ)` |

### 4.3 Loss weighting
| Name | Loss | Idea |
|---|---|---|
| plain | mean squared error of the target | |
| `suw` (new) | `Σ_b frac_b · (mse_b · e^{−u_b(t)} + u_b(t))` over 6 radial DCT bands, `u_b(t)` from a small MLP on t | **spectral uncertainty weighting**: learns per band and noise level how predictable each band is (EDM2-style uncertainty weighting extended to frequency bands) |
| `hf<β>` (new) | `mean over k of (1 + β/P_k) · (DCT error)²` | **additive spectral boost**: never reduces the low-frequency weight, adds weight to fine frequencies in inverse proportion to their energy. β = 0.01 → DC ×1.0003, (8,8) ×1.46, finest ×3.2 |

### 4.4 Other options
- `sc`: **self-conditioning** (Chen et al. 2022): the network also sees its previous x0 estimate (50% of training samples).
- Samplers: `cold` and `ddim` (old process only); `eta<η>` (per-frequency DDIM) and `eta<η>q2` (quadratic step spacing, more steps at low noise).

---

## 5. Files

| File | What it does |
|---|---|
| `blur_micro.py` | the prototype framework: data, DCT, all processes, targets, losses, training, samplers, KID/FID75, `report` |
| `run_micro.sh`, `queue_micro.txt`, `done_micro.txt` | the low-power queue runner and the queue of runs |
| `grid.py` | sample grids (same starting noise in every column) |
| `micro_runs/<run>/` | `eval.json` (KID, FID75), `loss.csv`, `train_done.json`, `suw_weights.json` (learned band weights), samples |
| `micro_grid_*.png` | sample grids used in `eval.md` |
| `blur_proto.py` | the same framework at 32×32 with a base-16 U-Net and Inception FID/IS, for the DGX runs (only partly run here) |
| `arch_bench.py` | helper from the DGX benchmark (original `UNet`, timestep encoding, cosine schedule) |

Run one prototype: `taskpolicy -b python blur_micro.py run <process> 3000 0`, re-score with other samplers: `python blur_micro.py reeval <process> eta0q2,eta1`,
table: `python blur_micro.py report`.
