# Decisions: blur-diffusion prototypes (16 Nov 2026)

What was decided, why, and what the evidence said. Setup: **`context.md`**. Numbers: **`eval.md`**. Next: **`TODO_NEXT.md`**.

---

## 1. The recommended blur-diffusion formula (so far)

**Solution 1: exact spectral blur diffusion with a noise-re-estimating sampler** (best: KID 137.1 vs 219.6 for the old
pipeline, −38%; vs 208.7 for Gaussian eps, −34%)

```
Forward:   X_t(k) = √ᾱ_t · e^{−λ_k τ_t} · X_0(k) + √(1−ᾱ_t) · E(k)        X = DCT(x),  λ_k = (πk_x/N)² + (πk_y/N)²,  τ_t = σ_b(t)²/2
           σ_b(t) = σ_max (1 − √ᾱ_t)     (σ_max = 2.5 px at 32×32, 1.25 px at 16×16; the old schedule)
Network:   predicts x0
Sampler:   per-frequency DDIM / ancestral step (η = 1 best, or η = 0 with quadratic step spacing):
           Ê = (X_t − a_t X̂0)/s_t                         ← re-estimate the noise from the current state
           X_{t'} = a_{t'} X̂0 + √(s_{t'}² − σ²) Ê + σ z,   σ² = η² s_{t'}² (s_t² − (a_t/a_{t'})² s_{t'}²) / s_t²
```

The biggest single win is the **sampler**: the Cold Diffusion update reuses the initial noise `z` at every step, but the
correct move for blur + noise is to re-estimate the noise from the current state each step (DDIM generalised to blur). That
alone cut KID by 30% on the same trained model, and the 32×32 check agreed (FID 193 → 184). Stochasticity (η = 1) or more
steps at low noise (quadratic spacing) gave another 10–11%.

**Solution 2: data-adaptive "Wiener blur"** (KID 156.6, a tie with the tuned Gaussian blur at η = 0)

```
d_k(t) = ( SNR_k(t) / (1 + SNR_k(t)) )^γ,    SNR_k(t) = ᾱ_t P_k / (1 − ᾱ_t),    γ = 0.5
X_t(k) = √ᾱ_t · d_k(t) · X_0(k) + √(1−ᾱ_t) · E(k)
```

Each frequency is attenuated by its own Wiener gain at the current noise level, using the measured power spectrum `P_k` of
the training images. It blurs exactly what the noise already hides and leaves visible detail alone. It matches the
hand-tuned blur with **no blur width to choose**, which makes it a principled default for new datasets and resolutions.

**Solution 3 (pending):** *additive spectral boost* `w_k = 1 + β/P_k` on the x0 loss, and/or *self-conditioning*. Results
come in after this file was written; see the final table (`python blur_micro.py report`) and `TODO_NEXT.md`.

---

## 2. Decisions and reasons

| # | Decision | Why | Outcome |
|---|---|---|---|
| D1 | Write blur + noise as a **Gaussian diffusion diagonal in the DCT basis** | a Gaussian blur with reflective boundaries is diagonal in the DCT, so blur + noise is a Gaussian process with a closed-form reverse step; one framework then covers Gaussian, blur, Wiener and heat-equation processes | ✓ exact samplers, verified with a perfect-prediction test (error ~1e-6) |
| D2 | Replace the cold sampler with **noise re-estimation** (DDIM for blur) | the cold update assumes the noise in `x_t` is still the initial `z`, which stops being true once predictions are imperfect | ✓✓ −30% KID |
| D3 | Try **η > 0 and quadratic step spacing** | blur hides fine detail until the last, low-noise steps; fresh noise or more steps there should help | ✓ −10 to −11% |
| D4 | Exact DCT blur instead of the zero-padded pixel blur | remove the dark-border artefact of zero padding | ≈ no change in score or images (the streaks were not caused by padding) |
| D5 | **u target** (per-frequency EDM preconditioning with the data spectrum) | natural images have ~10⁵× more energy at coarse frequencies; x0 loss ignores fine detail, single-σ_data EDM is mis-scaled | ✗ in this tiny regime: colour casts (DC/colour gets 1/256 of the loss weight) |
| D6 | **Spectral uncertainty weighting** (learned per-band, per-t weights) | let the model learn which bands are predictable when | ✓ on u (−7%, and it learned a coarse-to-fine sweep), ✗ on x0 (+12%: it turned the colour bands off) |
| D7 | **u^γ** (partial spectral preconditioning) | balance colour and detail | ✗ γ = 0.5 was worst (grainy texture, target no longer unit-variance) |
| D8 | **Additive spectral boost** `1 + β/P_k` | after D5–D7: never reduce the low-frequency (colour) weight, only add weight to fine frequencies | pending |
| D9 | Switch the prototypes from 32×32/base 16/8 threads to **16×16/base 8/efficiency cores** and **KID** instead of FID | the laptop overheated; FID with 1,000 samples has a floor of 74 for real vs real, KID is unbiased (0.22) | ✓ cool laptop, ~10 min per formula |
| D10 | Same training stream, init and sampling noise for every run | so differences come from the formula only | ✓ |

## 3. What we learned
1. **In blur diffusion the sampler matters more than the blur.** Most of the old pipeline's gap came from the cold sampler.
2. **At small scale, low frequencies (colour, layout) decide KID.** Every change that took weight away from them hurt (u, u^0.5,
   x0 + SUW). Fine detail can only be added on top, not traded for.
3. **x0-prediction is the most robust target for small, short-trained blur models;** eps and u need more capacity to remove
   the last fine noise.
4. **The data spectrum is a useful design input:** it gave a blur schedule with no tuning (Wiener blur) and explains why x0
   samples are soft.
5. **Learned spectral weights reproduce the coarse-to-fine structure of diffusion** without being told (u + SUW).

## 4. Dropped (for now)
`u` and `u^γ` targets at this scale, x0 + SUW. They may behave differently with a bigger model and longer training; that
is a DGX question (see `TODO_NEXT.md`).
