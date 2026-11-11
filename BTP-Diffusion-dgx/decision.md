# Decisions: what was chosen, and why

A log of every significant decision in the project: the reasoning, the alternatives, and what the evidence later said.
Setup: **`context.md`**. Numbers: **`eval.md`**.

---

## 1. Experimental setup

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| D1 | **Airplanes only, 32×32** | many architectures can be compared quickly on a T4/A100 | full CIFAR-10 (kept for later via `ddpm_paper.py`) |
| D2 | **Width base 32 (down from 64)** | fit the Colab T4; keep all models "small" (1.5–4.7M params) | base 16 (even faster, too weak) |
| D3 | **Cosine schedule, T = 1000** | the linear schedule destroys signal too early at 32×32 | linear (Ho et al.) |
| D4 | **EMA weights for all sampling and FID** | the single biggest visual-quality gain in diffusion training | live weights |
| D5 | **FID on 5,000 samples, same seed and deterministic sampler for every model** | the user couldn't rank blurry samples by eye; fixed noise makes each column the "same" sample across models | 2,000 samples (faster, noisier); 10k (too slow) |
| D6 | **Keras InceptionV3** for FID/IS | weights were already cached on Colab and the DGX; same numbers on both | the paper-standard 2015 graph (`fid_paper.py`); fine for absolute values, not needed for relative ranking |
| D7 | **FID as the main metric, IS secondary** | IS saturates at 4.91 on one class | IS-first |
| D8 | **Identical training stream** (images, flips, t, noise depend only on the step) | the first Colab path runs shared one iterator, so models saw different data; differences < 2% were unreadable | per-model random streams |
| D9 | **Stratified timesteps 13/19/19/77 per batch** (DGX) | the routed U-Net needs fixed-size slices per route for XLA; applied to every model so the stream stays identical | uniform t + boolean masks (dynamic shapes, slow) |
| D10 | **4,000 steps (≈100 epochs)** per model | fits the 2-hour DGX budget and matches the Colab runs | longer (planned for the next round) |

## 2. Architecture decisions

| # | Decision | Why | Outcome |
|---|---|---|---|
| A1 | **Route-based `PathUNet`**: one class, any list of levels; skip = last feature map at a revisited level | test the ladder idea and many shapes with identical bricks; `[0,1,2,3,2,1,0]` reproduces a plain U-Net | ✓ made fair comparisons cheap |
| A2 | Test **both stepping and jumping** readings of the ladder idea | the user's description was ambiguous | stepping won; jumping tied the U-Net |
| A3 | **Size-matched controls** `ladder_step_b24`, `unet_b40` | separate "better shape" from "more parameters" | stepping lost at equal params; a bigger U-Net gained little → the ladder uses extra compute well, not params |
| A4 | **Diagnose the 85-FID gap to the original U-Net as a brick problem** (2×2 non-overlapping down, 32 ch at 32×32, attention at 4×4) | blocky texture only in path models, only in ε-prediction; the gap shrank to 8 in blur | ✓ confirmed on the DGX (v1 → v2: 209 → 150) |
| A5 | **v2 bricks:** 3×3 stride-2 down, 64/128/128 ch, stop at 8×8, attention at 8×8 | copy what the original U-Net does right | ✓ |
| A6 | **Noise-routed U-Net** (the user's "different layers for different t" idea), with fixed and learned routing plus a dense control | could make sampling 3–8× cheaper at high noise | ✗ quality much worse (193–293 FID); pruning gates collapsed it |
| A7 | Routed **way up mirrors the way down** with its own blocks | encoder and decoder do different jobs (standard U-Net practice) | – |
| A8 | **SWL (new):** Haar resampling + ladder passes with **shared weights** + pass gates g(t) + adaLN-Zero | each piece targets a known U-Net weakness: aliasing (Haar), the ladder's parameter cost (sharing), fixed compute per noise level (gates), weak time conditioning (adaLN-Zero) | ✓✓ best model: 92.8 vs 128.5 |
| A9 | Gates initialised to 1, adaLN/output convs zero-initialised | the slow start seen in `ladder_reverse` (worst for 3,000 steps); zero-init makes every block start as an identity | – (not ablated yet) |
| A10 | **Improvement round on SWL:** wider, third pass, longer training | push the best model further cheaply | wider + (87.4), third pass − (100.5), longer ++ (72.0) |
| A11 | Drop `w_net`/`wnet_v2` and `jump_down_step_up` from the DGX run | the largest/slowest; on Colab they tied each other and didn't isolate a question of their own | – |

## 3. Pipeline / scheduler decisions

| # | Decision | Why | Outcome |
|---|---|---|---|
| P1 | Keep the **blur + noise pipeline (x0-prediction)** alongside Gaussian | a second kind of diffusion shows whether rankings generalise | rankings largely held; the reverse ladder is best in blur |
| P2 | **Spectral wavelet schedules** (`spec_eq`, `spec_c2f`), shifts computed from real band power | test the "global info matters more" intuition directly in the noise process; Haar is orthonormal, so DDIM still works per coefficient | ✗ hurt every model at 4k steps; SWL was the most robust |
| P3 | Don't use I-JEPA as the generator; consider **REPA-style alignment** later (DINOv2 / I-JEPA encoder, **never Inception**) | I-JEPA doesn't generate images; aligning to Inception would game FID | deferred until the architecture question is settled |

## 4. Infrastructure decisions

| # | Decision | Why |
|---|---|---|
| I1 | **Colab: everything resumable from Drive checkpoints; never change Drive paths** | sessions drop; earlier checkpoints must stay loadable |
| I2 | **`optimizer.build(...)` up front, `@tf.function(autograph=False)`** | Keras 3 lazy optimizer state failed inside `tf.function` on Colab |
| I3 | **SSH key in the macOS Keychain + `ssh dgx` alias**; the password never passes through Claude | security; non-interactive access for running jobs |
| I4 | **All models round-robin in one process per kind** (`train_multi`) | the SLURM GPU was shared with an out-of-SLURM process (39 of 40 GB used), so 20 processes OOM'd; one process also guarantees identical step counts |
| I5 | **Step target read live from a file** (`target_steps`) | choose the step count after measuring real speed, without restarting |
| I6 | **Extra runs inside our own allocation** (`srun --overlap`) + a dependent follow-up job | use the idle GPU memory during evaluation; resume from checkpoints if the first job ends |
| I7 | Respect others on the shared DGX | never touch other users' processes; 1 GPU, `--mem=128G` |
| I8 | Keep outputs small, in `~/arch_bench` | `/raid` was 100% full |

## 5. How the user's hypotheses fared

| Hypothesis | Verdict |
|---|---|
| "The ladder will beat a normal U-Net" | ✓ on good bricks (129 vs 150) and with enough width; ✗ at equal parameters on v1 bricks |
| "At equal params: jumping > unet > stepping" | partly: unet > stepping ✓; jumping > unet ✗ (a tie) |
| "Bigger models do worse initially, so the ladder's early lead means it will win" | ✗ reasoning (bigger models usually learn faster per step), but the ladder did keep its lead on FID |
| "jump_down_step_up is bigger, so it can be scrapped" | the logic was off (no smaller model beat it), but it tied `w_net`, so dropping it was reasonable |
| "Global information matters more than fine detail" | partly: global structure drives FID, but full-resolution capacity can't be starved; the exaggerated coarse-to-fine schedule hurt |
| "The ladder (and reverse ladder) will do much better at big scale" | untested; needs a scaling curve |

## 6. Decided next steps (in priority order)

1. **Train SWL and the original U-Net equally long** (16k–30k steps), 2–3 seeds each. Does the −28% hold?
2. **Ablate SWL one piece at a time:** shared vs separate weights, Haar vs strided conv, adaLN-Zero vs additive time, gates on/off.
3. **SWL on blur + noise** and with **v-prediction**.
4. **Compare at equal sampling compute** (SWL is ~2–3× slower per sample).
5. **Scaling curve** (base 16/32/64) for SWL vs the U-Net, to test the "better at big scale" hope.
6. Later: **REPA-style alignment** for every model equally; full CIFAR-10.

**Dropped (unless new evidence appears):** the noise-routed U-Net in its current form, the spectral wavelet schedules,
a third shared SWL pass, and the v1 bricks.
