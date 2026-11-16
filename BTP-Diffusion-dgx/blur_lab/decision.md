# Decisions: blur-diffusion prototypes (16 Nov 2026)

What was concluded, why, and how sure we are. Setup: **`context.md`**. Numbers: **`eval.md`**. Next: **`TODO_NEXT.md`**.

---

## 0. Short version

The old blur pipeline's problems were mostly **its sampler, not the blur**. Three changes, each tested on its own, take it
from **KID 219.6 to 97.5 (−56%)** and from worse than Gaussian diffusion to under half its KID (Gaussian eps: 208.7):

1. **Re-estimate the noise every step** instead of reusing the initial noise (fixes Cold Diffusion's update): −30%.
2. **Stochastic sampling (η = 1) with quadratic step spacing** for x0-predicting models: −29% on top.
3. **Stochastic heat equation** as the forward process plus the **additive spectral boost** loss: the best result (97.5).
   The boost helps in all 7 paired tests (−3 to −6%), and the heat equation beats no-blur by ~5.5 KID in both paired tests.

Honest caveat: **at this scale a plain blur is neutral.** A Gaussian x0 model with the same sampler scores 108.9, about the
same as mild blur (110.0), and stronger blur is slightly worse (110.5 vs 106.2 with the boost). Only the **stochastic heat
equation** (blur + coloured noise) beats no-blur, by a small but repeated margin (103.2 vs 108.9; 97.5 vs 103.2). The new
sampler also **hurts** Gaussian eps models. So most of the gain comes from the x0 target + sampler + spectral boost; whether
blur helps at full scale is the main open question for the DGX.

---

## 1. The three solutions

### Solution 1: exact spectral blur diffusion with a noise-re-estimating sampler ("blur-DDIM")
```
Forward (DCT domain, k = frequency):
    X_t(k) = a_t(k) · X_0(k) + s_t · E(k),      a_t(k) = √ᾱ_t · exp(−λ_k τ_t),   s_t = √(1 − ᾱ_t)
    λ_k = (π k_x / N)² + (π k_y / N)²,   τ_t = σ_b(t)² / 2,   σ_b(t) = σ_max (1 − √ᾱ_t)
Network predicts x0.  Each sampling step t → t':
    Ê = (X_t − a_t · X̂0) / s_t                        ← re-estimate the noise from the current state
    X_t' = a_t'(k) · X̂0 + s_t' · Ê                    (deterministic version)
```
- **Evidence:** same old model: cold 219.6 → re-estimation 154.2 (−30%); 32×32 check: FID 193.4 → 183.5.
- **Why:** the Cold Diffusion update `x_t − D(x̂0,t) + D(x̂0,t−1)` assumes the noise inside `x_t` is still the initial `z`.
  With imperfect predictions that is false, and the error accumulates. Re-estimating `Ê` from the current state is the DDIM
  step for a blur + noise Gaussian process.
- Writing the blur as `exp(−λ_k τ)` in the DCT basis makes the whole process an exactly solvable Gaussian diffusion (checked:
  perfect predictions reconstruct the image to 1e-6). The exact boundary handling itself didn't change the score (154.2 → 153.9).

### Solution 2: posterior sampling for x0 models (stochastic η = 1 + quadratic step spacing)
```
X_t' = a_t' X̂0 + √(s_t'² − σ²) Ê + σ z,     σ² = η² s_t'² (s_t² − r² s_t'²) / s_t²,   r = a_t / a_t',   η = 1
sampling times  t_i = T · (i / n)²            (more steps at low noise, where blur hides detail until the end)
```
- **Evidence:** exact blur x0: 153.9 → 110.0 (η = 1 alone 137.1, quadratic alone 138.3); cubic ≈ quadratic (105.8 vs 106.2);
  Gaussian x0: 151.8 → 108.9; Wiener blur: 156.6 → 111.7; stochastic heat eq.: 156.3 → 103.2.
- **Why:** an x0 predictor outputs the **average** of all plausible images, which is blurry. Deterministic DDIM follows that
  average all the way down; η = 1 adds fresh noise each step, and the low-noise steps (which quadratic spacing concentrates)
  turn it back into detail. That samples from the posterior instead of following its mean.
- **Not general:** on a Gaussian **eps** model the same sampler makes things worse (208.7 → 252.0 / 258.5). Use it for x0 models.

### Solution 3: spectrum-aware design: stochastic heat equation + additive spectral boost
```
Stochastic heat equation (blur and noise at the same time, per DCT frequency):
    dX(k) = −(λ_k τ'(t) + β_t / 2) X(k) dt + √β_t dW
  → a_t(k) = √ᾱ_t · exp(−λ_k τ_t)   (same as Solution 1),   s_t(k)² = Σ_u β_u · exp(−2 ∫_u^t …)   (coloured noise)

Additive spectral boost on the x0 loss (P_k = mean power of the training images at frequency k):
    L = mean_k (1 + β / P_k) · |DCT(x̂0 − x0)(k)|²,      β = 0.01
```
- **Evidence:** stochastic heat eq. + x0 + Solution 2: **103.2**, the best result (FID75 also best, 247.9). The boost improves the
  exact-blur model with every sampler (153.9 → 147.6, 137.1 → 129.6, 110.0 → 106.2), the Gaussian x0 model (151.8 → 146.2,
  108.9 → 103.2) and the heat-equation model (156.3 → 150.5, **103.2 → 97.5 = best overall**); β = 0.03 is too strong (127.7).
- **Why the heat equation:** noise injected along the way is itself blurred by the later heat flow, so the middle of the
  process has less high-frequency noise; with stochastic sampling, the fresh noise added each step is coloured the same way.
- **Why the boost:** natural images have ~10⁴–10⁵× more energy at coarse frequencies, so an x0 loss barely sees fine detail.
  Earlier attempts that **re**-weighted frequencies (u target, u^γ, learned band weights) took weight away from the colour/DC
  frequencies and made samples worse. The boost only **adds** weight to fine frequencies and never lowers DC's (weight
  1.0003 at DC, 3.2× at the finest frequency). Per-frequency constant weights don't change the optimal predictor, only
  how a small model spends its capacity.

### Also found: data-adaptive Wiener blur (no blur width to tune)
```
a_t(k) = √ᾱ_t · ( SNR_k(t) / (1 + SNR_k(t)) )^γ,    SNR_k(t) = ᾱ_t P_k / (1 − ᾱ_t),    γ = 0.5
```
It ties the hand-tuned Gaussian blur (156.6 vs 153.9; 111.7 vs 110.0) **without choosing σ_max**: a principled default when
moving to new datasets or resolutions.

---

## 2. Decisions and reasons

| # | Decision | Why | Outcome |
|---|---|---|---|
| D1 | Write blur + noise as a **Gaussian diffusion diagonal in the DCT basis** | a reflective Gaussian blur is diagonal in the DCT, so one exact framework covers Gaussian, blur, Wiener and heat-equation processes | ✓ exact samplers, perfect-prediction test passes (1e-6) |
| D2 | Replace the cold sampler with **noise re-estimation** | the cold update assumes the noise is still `z` | ✓✓ −30% |
| D3 | **η = 1 + quadratic spacing** | x0 predictions are posterior means (blurry); fresh noise + more low-noise steps restore detail | ✓✓ −29% more (x0 models only) |
| D4 | Exact DCT blur instead of zero-padded pixel blur | remove the boundary artefact | ≈ no change (streaks came from the small x0 model, not padding) |
| D5 | **u target** (per-frequency EDM preconditioning with the data spectrum) | x0 loss ignores fine detail; single-σ_data EDM is mis-scaled across a 10⁵ spectrum | ✗ colour casts: DC/colour gets 1/256 of the weight |
| D6 | **Spectral uncertainty weighting** (learned per-band weights) | learn which bands are predictable when | u: ✓ −7% and learned a coarse-to-fine sweep · x0: ✗ switched colour bands off |
| D7 | **u^γ**, γ = 0.5 | balance colour and detail | ✗ worst (grainy, target no longer unit-variance) |
| D8 | **Additive spectral boost** `1 + β/P_k` | only add weight to fine frequencies, never remove colour weight | ✓ −3 to −5% with every sampler; β = 0.01 ≫ 0.03 |
| D9 | **Gaussian x0 control** | is the gain from blur or from the target + sampler? | blur is **neutral** at this strength (151.8 vs 153.9; 108.9 vs 110.0) |
| D10 | **Stronger blur** (2× width) with the best setup | does more blur help once the sampler is right? | no (110.5 vs 106.2) |
| D11b | **Heat equation + boost** as the final candidate | best process × the only consistently helpful loss change | ✓ best result, 97.5 |
| D11 | Dropped self-conditioning from the queue | slowest run, generic trick; the Gaussian and stronger-blur controls answered more important questions | – (in `TODO_NEXT.md`) |
| D12 | Prototypes at **16×16, base 8, efficiency cores, KID** | the laptop overheated at 32×32; FID on 1,000 samples has a floor of 74, KID is unbiased | ✓ ~10 min per formula, cool laptop |
| D13 | Identical training stream, init and sampling noise for every run | differences come from the formula only | ✓ |

## 3. What we learned
1. **In blur diffusion, the sampler matters more than the blur.** The old pipeline's gap to Gaussian was the cold sampler.
2. **x0-prediction + stochastic, low-noise-heavy sampling** is the strongest combination for small models.
3. **Never take loss weight away from the low frequencies** at this scale: colour and layout dominate KID.
4. **The data spectrum is a useful design input:** it gives a tuning-free blur (Wiener) and a safe fine-detail boost.
5. **Learned spectral weights rediscover coarse-to-fine generation** (u + SUW), a nice sanity check on the theory.
6. **Plain blur (mild or strong) does not beat no blur** at 16×16 with a 224k-param model; the **stochastic heat equation**
   does, by ~5%. Both claims need a full-scale test and a second seed.

## 4. Dropped (at this scale)
u and u^γ targets; x0 + spectral uncertainty weighting; boost β ≥ 0.03; η = 0.5; stronger blur. They may behave differently
with a 2–3M-parameter model trained longer, so u and SUW are listed for a DGX re-test in `TODO_NEXT.md`.
