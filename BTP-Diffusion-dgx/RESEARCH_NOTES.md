# Diffusion architecture research: running notes

These notes record what has been built, what has been run, what the numbers say, and what to do next.
They were written on 2026-10-01 at the end of a long session, so that the next session (with a person or with
Claude) can continue without re-deriving anything. **Read section 2 first if "path unet" is confusing.**

---

## 0. One-paragraph summary

We train small DDPM diffusion models on **CIFAR-10 airplanes only** (5,000 images at 32×32) in TensorFlow/Keras on a
**Colab T4**. The goal is to find a U-Net variant that beats a normal U-Net. The user's idea is a **"ladder"**:
instead of one trip down and back up the resolution levels, the network makes several trips that get deeper each
time. So far, `ladder_step` (the stepping ladder) beats the plain U-Net built from the same parts
(FID 178.9 against 209.4), and it beats a bigger 2-U-Net stack (`w_net`, 191.1). But it **loses at equal parameter
count** (`ladder_step_b24`, FID 227.7). **All of these "path" models lose badly to the original hand-written
`UNet`** (FID 124.5) in Gaussian diffusion. In blur + noise diffusion the gap almost disappears (141.2 against
149.3), and `ladder_step` ties the original (141.1). The current best explanation is that the path models'
building blocks (mainly 2×2 downsampling and thin full-resolution layers) are bad at predicting *noise*,
which needs pixel-precise output. A **noise-routed U-Net** (different blocks for different noise levels) is
written and in the notebook, but has **not been run yet**.

---

## 1. Files and where things live

### Local (Mac): `/Users/armaanjotsingh/Desktop/diffusion/`
| File | What it is |
|---|---|
| `diffusionRedo.ipynb` | **The main notebook.** Upload this to Colab. 83 cells. |
| `diffusion-3-main/diffusionRedo.ipynb` | An identical copy (keep both in sync if you edit). |
| `diffusion-3-main/diffusion-3.ipynb` | The notebook `diffusionRedo` was copied from (has the Drive caching). Don't edit. |
| `diffusion-3-main/ddpm_paper.py` | Paper-style DDPM on full CIFAR-10 (SLURM jobs). Not used on Colab. |
| `diffusion-3.ipynb`, `diffusion.ipynb`, `new/new.ipynb` | Older notebooks. `new/new.ipynb` is a fresh rewrite the user started. |
| `RESEARCH_NOTES.md` | This file. |

### Google Drive (Colab): `MyDrive/diffusion_cache/`
**Never change these paths.** The user has trained checkpoints in all of them.

| Path | Contents |
|---|---|
| `cifar10_airplanes.npz` | The cached airplane-only dataset (`CACHE_PATH`) |
| `checkpoints/` | Original `UNet`, Gaussian (base 32) |
| `checkpoints_blur/` | Original `UNet`, blur + noise (`model_blur`, base 32, 100 epochs, old non-identical stream) |
| `checkpoints_base64/`, `checkpoints_blur_base64/` | Old base-64 runs, moved aside (they caused a shape-mismatch error) |
| `checkpoints_path_unet/<name>/` | Gaussian path models: `unet`, `ladder_step`, `ladder_jump`, `ladder_reverse`, `jump_down_step_up`, `w_net`, `ladder_step_b24`, `unet_b40` |
| `checkpoints_path_unet/eval_results_n5000_ddim100.json` | Gaussian FID/IS results |
| `fid_reference_airplanes.npz` | Inception statistics of the 5,000 real airplanes (FID reference) |
| `checkpoints_path_unet_blur/<name>/` + `eval_results_n5000_steps100.json` | Blur + noise runs of every architecture (150 epochs, identical stream) |
| `checkpoints_routed_unet/<name>/` + `eval_results_n5000_ddim100.json` | Noise-routed U-Net runs (**not run yet**) |

### Environment
- Colab: Python 3.13, a recent TensorFlow + Keras 3, T4 GPU. Local Mac venv: `.venv` (TF 2.21, Keras 3.15, Python 3.13).
- Diffusion settings: `timesteps = 1000`, cosine schedule (`ALPHA_BAR`, t is 0-indexed, 0..999), `embedding_dim = 128`.
- Training: batch 128, Adam 2e-4, EMA 0.999 with warm-up, random horizontal flips. 100 epochs = 39 steps × 100 = **3,900 steps**.

---

## 2. What is "path unet"? (the confusing part, explained slowly)

There are **two different U-Nets** in the notebook that both do "down then up". They are built from **different
parts**, so they behave differently even though their shape looks the same.

### 2.1 The original `UNet` (top of the notebook, cells 17–21)
This is the U-Net written by hand at the start of the project. It's a fixed class:

```
input 32×32
  │ input_conv (32 ch)
  ├─ DownBlock: ResBlock 32→64 ch at 32×32, then 3×3 stride-2 conv ─► 16×16   (skip1: 64 ch @32×32)
  ├─ DownBlock: ResBlock 64→128 ch at 16×16, then 3×3 stride-2 conv ─► 8×8    (skip2: 128 ch @16×16)
  ├─ Bottleneck at 8×8: ResBlock, ATTENTION (64 positions), ResBlock
  ├─ UpBlock: transposed conv ─► 16×16, concat skip2, ResBlock → 64 ch
  ├─ UpBlock: transposed conv ─► 32×32, concat skip1, ResBlock → 32 ch
output conv
```
- 3 resolution levels: 32 → 16 → 8. **It never goes to 4×4.**
- 64 channels at full resolution (in the first ResBlock).
- 3×3 stride-2 downsampling (overlapping), transposed-conv upsampling.
- Attention at **8×8**.
- Takes the timestep as a precomputed sinusoidal vector `PE[t]`.
- 1,663,235 params at `base=32` (it was 6,064,131 at the old `base=64`).
- Lives in Python as `model` (Gaussian) and `model_blur` (blur). In the newer experiments it is retrained under the name **`original_unet`**.

### 2.2 `PathUNet`: a U-Net *builder* (the "new part", cells 59–62)
`PathUNet` is **not one network**. It's a recipe that builds a network from a **list of levels to visit**:

```python
PathUNet([0, 1, 2, 3, 2, 1, 0])     # level 0 = 32×32, 1 = 16×16, 2 = 8×8, 3 = 4×4
```
Rules:
- Every number in the list = "go to that resolution and run one ResBlock there".
- Moving between levels uses a `Resample` layer: down = conv with **kernel = stride = 2^k** (for a single step that's a
  **2×2 stride-2 conv**, non-overlapping), up = nearest-neighbour upsample + 3×3 conv.
- **Skip connections are automatic:** when the path lands on a level it has visited before, the last feature map from
  that level is concatenated in first.
- Channels per level at `base=32`: 32, 64, 64, 128 (`mults=(1, 2, 2, 4)`).
- Attention **only at 4×4** (level 3).
- Takes the timestep as an integer `t` and embeds it inside the model.

So **all the ladder models are `PathUNet` with different lists**, made from the **same parts**:

| Name | The list given to `PathUNet` |
|---|---|
| **`unet`** ← this is "path unet" | `[0, 1, 2, 3, 2, 1, 0]` (a plain down-and-up) |
| `ladder_step` | `[0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0]` |
| `ladder_jump` | `[0, 1, 0, 2, 0, 3, 0]` |
| `ladder_reverse` | `[0, 3, 0, 2, 0, 1, 0]` |
| `jump_down_step_up` | `[0, 1, 0, 2, 1, 0, 3, 2, 1, 0]` |
| `w_net` | `[0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]` |

### 2.3 "Path unet" in one sentence
**"Path unet" = `PathUNet([0, 1, 2, 3, 2, 1, 0])`: the plain U-Net shape, built from the same Lego bricks as the
ladders.** It's the fair baseline for the ladders, because only the route changes between them. The original `UNet`
has the same general idea (down, then up) but is built from **different bricks**.

Analogy: two houses with the same floor plan, one built with the original bricks (strong) and one with the new bricks
(weaker for some jobs). Comparing ladders against path `unet` compares floor plans fairly. Comparing path `unet`
against the original `UNet` compares the bricks.

### 2.4 Side by side
| | **Path `unet`** | **Original `UNet`** |
|---|---|---|
| Defined in | cell 61 (`PathUNet`), list `[0,1,2,3,2,1,0]` | cell 21 (`class UNet`) |
| Bricks | `PResBlock`, `Resample`, `PAttentionBlock` (cell 60) | `ResBlock`, `DownBlock`, `UpBlock`, `AttentionBlock` (cells 17–20) |
| Levels | 32 → 16 → 8 → **4** | 32 → 16 → **8** |
| Width at 32×32 | 32 ch | up to **64** ch |
| Downsampling | **2×2** stride 2 (non-overlapping) | **3×3** stride 2 (overlapping) |
| Upsampling | nearest + 3×3 conv | transposed conv |
| Attention | 4×4 only (16 positions) | **8×8** (64 positions) |
| Timestep input | integer `t` | `PE[t]` vector |
| Params | 1,544,675 | 1,663,235 |
| Gaussian FID | 209.4 | **124.5** |
| Blur FID | 149.3 | **141.2** |

---

## 3. Notebook map: `diffusionRedo.ipynb` (0-based cell indices)

| Cells | Content |
|---|---|
| 0–12 | Imports, GPU memory growth, data load (Drive cache), normalise to [-1, 1], schedules |
| 14–16 | Sinusoidal timestep encoding `PE` |
| 17–21 | **Original `UNet`** blocks and class (`base` default 32) |
| 23–28 | Shape and gradient tests |
| 29 | Optimizer (**`optimizer.build(...)`**) + EMA helpers (`swap_in_ema`, `restore_weights`) |
| 30 | Gaussian checkpoint manager (`checkpoints/`) |
| 31 | `PE`, `BETA`, `ALPHA_BAR`, `train_step` (**`@tf.function(autograph=False)`**) |
| 33–35 | `show_images`, `ddim_sample`, `ddpm_sample` |
| 38 | Gaussian training loop (`epochs = 100`, `dataset`) |
| 40–43 | Generate samples, save weights |
| 44–54 | **Blur + noise (cold diffusion) pipeline**: `degrade_blur_noise`, `model_blur`, x0-prediction, `cold_sample` |
| 55–58 | Paper-style DDPM section (import of `ddpm_paper` is guarded, so it's skipped on Colab) |
| 59 | **"This is new part"** (markdown) |
| 60 | Path building blocks: `timestep_embedding`, `PResBlock`, `PAttentionBlock`, `Resample` |
| 61 | `PathUNet` class |
| 62 | `PATH_BASE = 32`, `ARCHITECTURES` dict (6 shapes), parameter check |
| 64 | Train every path U-Net (Gaussian), checkpoints under `checkpoints_path_unet/` |
| 66 | FID/IS tools: `InceptionScorer` (**batch=50**, to avoid OOM), `ddim_generate`, `evaluate`, real reference |
| 67 | Evaluate the original `UNet` + all path models, print table |
| 68 | Bar charts + sample grids |
| 69–72 | Size-matched: `EXTRA_ARCHITECTURES` (`ladder_step_b24`, `unet_b40`), train, evaluate |
| 73–74 | Fixed validation loss split by noise level (held-out test airplanes) |
| 75–78 | **Blur + noise for every architecture** (150 epochs, identical stream, includes `original_unet`) |
| 79–82 | **Noise-routed U-Net** (dense / fixed / learned + `original_unet`), identical stream, FID + compute |

---

## 4. Concepts and glossary

- **Level:** a resolution. 0 = 32×32, 1 = 16×16, 2 = 8×8, 3 = 4×4.
- **Stepping:** moving one level at a time (each move halves or doubles the resolution).
- **Jumping:** moving several levels in one layer (e.g. 32×32 → 4×4 with one stride-8 conv).
- **Ladder:** several down-and-up passes, each going deeper (`ladder_step`), or jumping (`ladder_jump`).
- **Reverse ladder:** deepest pass first, then shallower ones (`ladder_reverse` = jumping version; `ladder_step_reverse` = stepping version, **not tested yet**: `[0,1,2,3,2,1,0,1,2,1,0,1,0]`).
- **base / mults:** channel width. Channels at level k = `base × mults[k]`. GroupNorm(8) needs channels divisible by 8, so base must be a multiple of 8.
- **EMA:** exponential moving average of the weights; all sampling and FID use the EMA weights.
- **Identical stream:** every model sees exactly the same images, flips, timesteps and noise at each training step (`blur_batch(step)`, `stream_batch(step)`). **The first Gaussian path runs (cell 64) and the size-matched runs did NOT have this**: they shared one data iterator, so each model saw different batches and noise. The blur and routed sections do have it.
- **FID:** Fréchet distance between Inception-V3 features of generated and real images. Lower is better. Not linear: points at FID 200 are "cheaper" than points at FID 30. Only compare numbers with the same sample count (here 5,000).
- **IS:** Inception Score; higher is better. On a single class it maxes out low (real airplanes: **4.91 ± 0.25**). Treat FID as the main number.
- **ε-prediction (Gaussian pipeline):** the model predicts the noise. Needs pixel-precise full-resolution output.
- **x0-prediction (blur pipeline):** the model predicts the clean image. Smooth target; tends toward blurry averages.
- **v-prediction:** a mix of the two; a common middle ground (not tried yet).
- **Evaluation protocol:** 5,000 samples, the same starting noise (seed 1234), deterministic DDIM (Gaussian) or cold sampler (blur), 100 sampling steps.

---

## 5. Architectures: size and cost (base 32 unless noted)

| Model | Params | MFLOPs / image (forward) | ms/step (T4, train) |
|---|---|---|---|
| original `UNet` | 1,663,235 | n/a | n/a |
| `unet` (path) | 1,544,675 | 314 | 122 |
| `ladder_step` | 2,353,059 | 742 | 256 |
| `ladder_jump` | 1,523,491 | 456 | 170 |
| `ladder_reverse` | 1,523,491 | 456 | 170 |
| `jump_down_step_up` | 2,109,283 | 603 | 212 |
| `w_net` | 2,994,883 | 621 | 218 |
| `ladder_step_b24` (base 24) | 1,543,035 | 420 | 200 |
| `unet_b40` (base 40) | 2,154,907 | ~490 (estimated) | 160 |
| `routed_dense` / `routed_fixed` | 3,937,635 | n/a | not run |
| `routed_learned` | 3,945,843 | n/a | not run |

Where `ladder_step`'s extra parameters live (vs `unet`): 16×16 blocks 716K vs 259K (visited 5 times, and each
revisit concatenates the previous features, doubling the input channels), 8×8 blocks 411K vs 259K.
`ladder_jump`'s single 32→4 jump layer alone is ~262K params (8×8 kernel, 32→128 ch).

---

## 6. Results

### 6.1 Gaussian, path models, training loss (mean of each 500-step block)
Non-identical batches between models (shared iterator). 3,900 steps.

| step | unet | ladder_step | ladder_jump | ladder_reverse | jump_down_step_up | w_net | ladder_step_b24 | unet_b40 |
|---|---|---|---|---|---|---|---|---|
| 500 | .2466 | .2580 | .2722 | .2814 | .2444 | .2821 | .2870 | .2489 |
| 1000 | .0996 | .0932 | .0960 | .1024 | .0931 | .0964 | .1016 | .0967 |
| 1500 | .0842 | .0800 | .0822 | .0865 | .0805 | .0824 | .0872 | .0825 |
| 2000 | .0762 | .0738 | .0749 | .0775 | .0734 | .0751 | .0799 | .0759 |
| 2500 | .0720 | .0677 | .0712 | .0731 | .0699 | .0705 | .0747 | .0708 |
| 3000 | .0676 | .0642 | .0681 | .0689 | .0661 | .0668 | .0714 | .0667 |
| 3500 | .0654 | .0619 | .0662 | .0661 | .0634 | .0639 | .0674 | .0651 |
| **3900** | **.0642** | **.0605** | **.0652** | **.0642** | **.0621** | **.0623** | **.0657** | **.0633** |
| vs unet | — | −5.8% | +1.6% | 0.0% | −3.3% | −3.0% | +2.3% | −1.4% |

### 6.2 Gaussian FID / IS (5,000 samples, DDIM 100, seed 1234)
| Model | Params | FID | IS |
|---|---|---|---|
| real airplanes | – | 0 | 4.91 ± 0.25 |
| **original `UNet`** (cell 38 training, own stream) | 1,663,235 | **124.53** | 3.14 ± 0.04 |
| `ladder_step` | 2,353,059 | 178.86 | 2.47 ± 0.05 |
| `w_net` | 2,994,883 | 191.13 | 2.36 ± 0.06 |
| `jump_down_step_up` | 2,109,283 | 194.10 | 2.31 ± 0.03 |
| `unet_b40` | 2,154,907 | 203.55 | 2.22 |
| `ladder_jump` | 1,523,491 | 209.04 | 2.16 ± 0.04 |
| `unet` (path) | 1,544,675 | 209.35 | 2.20 ± 0.05 |
| `ladder_step_b24` | 1,543,035 | 227.72 | 2.09 |
| `ladder_reverse` | 1,523,491 | 234.93 | 2.07 ± 0.03 |

Sample images: all still blurry sky-coloured blobs (undertrained). **Every path model shows a blocky, mosaic
texture; the original `UNet` doesn't.** The original's images are smoother, with more contrast and more coherent
shapes.

### 6.3 Blur + noise FID / IS (150 epochs = 5,850 steps, identical stream, cold sampler 100 steps)
Partial: 3 of 7 models finished at the time of writing.

| Model | Params | FID | IS | eval time |
|---|---|---|---|---|
| `ladder_step` | 2,353,059 | **141.06** | 2.87 ± 0.05 | 6.3 min |
| `original_unet` | 1,663,235 | **141.18** | 2.83 ± 0.05 | 4.3 min |
| `unet` (path) | 1,544,675 | 149.27 | 2.75 ± 0.06 | 3.5 min |
| `ladder_jump`, `ladder_reverse`, `jump_down_step_up`, `w_net` | | *pending* | | |

Sample images: **no blocky texture at all** in any model (same path bricks!). Softer/blurrier overall (typical of
x0-prediction). Plane-like dark silhouettes against bright sky in some samples. The three models look almost
identical by eye.

---

## 7. Conclusions so far (with how sure we are)

| # | Claim | Status |
|---|---|---|
| 1 | `ladder_step` beats path `unet` (Gaussian: −5.8% loss, −30 FID; blur: −8 FID) | Fairly solid (consistent across two pipelines, but one run each) |
| 2 | The ladder's gain isn't just size: `ladder_step` beats `unet_b40` (−25 FID) and `w_net` (−12 FID, 27% bigger) | Fairly solid, **but** `ladder_step` uses more compute (742 vs ~490 / 621 MFLOPs) |
| 3 | At equal params, stepping **loses** (`ladder_step_b24` +2.3% loss, +18 FID) | Solid. Likely because width at 32×32 drops to 24 ch |
| 4 | Jumping doesn't help at equal params (`ladder_jump` ≈ `unet`); reverse jumping is worst on FID | Solid for this setup |
| 5 | Loss and FID mostly agree. Exception: `ladder_reverse` (same loss as `unet`, much worse FID) | Observed |
| 6 | `jump_down_step_up` ≈ `w_net` (0.0621 vs 0.0623 loss; 194 vs 191 FID) | Tie (an earlier claim that it "beats" `w_net` was overstated) |
| 7 | The path bricks are bad for ε-prediction: an 85-FID gap to the original in Gaussian, only 8 in blur; blockiness only in Gaussian samples | **Hypothesis, strongly suggested.** Prime suspects: 2×2 non-overlapping downsampling, nearest upsampling, 32 ch at 32×32, attention only at 4×4 |
| 8 | All models are undertrained (loss ~0.06; the notebook note says <0.05 needed for shapes, 0.03–0.04 is good) | Solid |
| 9 | At equal GPU time, plain `unet` might match `ladder_step` (it's 2.1× faster per step) | Untested extrapolation |

User predictions:
- "At equal params: jumping > unet > stepping." The **unet > stepping** part is confirmed; **jumping > unet** is not (a tie).
- "Global info matters more than fine detail." Partly supported by the literature (Perception Prioritized Training, Simple diffusion), **but** detail capacity can't be starved (b24 result; the original wins with 64 ch at full res). The balance also depends on noise level.

---

## 8. Noise-routed U-Net (written, in cells 79–82, NOT run yet)

The user's idea: **different blocks for different noise levels.** Each level has two blocks going down
(`e{level}a`, `e{level}b`) and two going up (`d{level}a`, `d{level}b`). The bottleneck is at 4×4 right after level 3
(ResBlock + attention + ResBlock). The way up mirrors the way down with its own blocks (same letters).

| t range | Blocks used (down; mirrored up) | Share of dense block compute |
|---|---|---|
| 0–100 (low noise, detail) | L0a, L0b, L1a, L1b, bottleneck | 78% |
| 100–250 | L1a, L2a, L3a, bottleneck | 31% |
| 250–400 | L2a, L2b, L3a, bottleneck | 18% |
| 400–1000 (high noise, global) | L2a, L3a, L3b, bottleneck | 16% |

Average over 100 evenly spaced sampling steps: **29%** of the dense block compute.

Design details:
- Levels without blocks are still passed through (resize + skip), so **every route keeps a full-resolution path**. This matters for ε-prediction at high noise, where the target is nearly the input.
- A 1×1 `fuse` conv at every level merges skip + upsampled features, so blocks keep constant channel counts.
- Uses **3×3 stride-2 downsampling** (fixes the suspected 2×2 issue) and **attention at 8×8** in the level-2 blocks, plus the 4×4 bottleneck.
- Training the fixed route: each batch is split by route (`tf.boolean_mask`); 4 sub-passes per step.
- **Three versions + baseline** (identical stream, 100 epochs, `checkpoints_routed_unet/`):
  - `routed_dense`: all blocks at every t (upper bound on quality)
  - `routed_fixed`: the table above
  - `routed_learned`: every block has a gate `g(t) = sigmoid(...)`, initialised at ~0.88. The loss adds `GATE_PENALTY × compute used` (default 0.01: raise to 0.05 if the gates stay ~1, lower to 0.002 if they all close). At sampling, blocks with gate < 0.1 are skipped. Shows a heat map of gates vs t.
  - `original_unet`: retrained on the same stream as the baseline.
- **Caveat:** routed models have **3.94M params, 2.4× the original `UNet`**. Compare FID together with sampling time/compute. A follow-up could use `base=24` to match the original's size.
- Estimated time on the T4 (not measured): ~1.5–2 h training + 30–40 min evaluation.

---

## 9. Bugs hit and fixes (useful if errors come back)

| Problem | Cause | Fix (already in the file) |
|---|---|---|
| `Shapes (3,3,64,64) and (3,3,128,128) are incompatible` on checkpoint restore | Old base-64 checkpoints on Drive | Moved to `checkpoints_base64/`, `checkpoints_blur_base64/` |
| `train_step` returned `None` / `'Adam' object has no attribute '_momentums'` | Keras 3 builds the optimizer lazily; failed inside `tf.function` on Colab | `optimizer.build(model.trainable_variables)` right after creating every optimizer; `@tf.function(autograph=False)` on the train steps |
| `property 'path' of 'PathUNet' object has no setter` | Keras reserves `.path` | The attribute is called `self.levels` |
| `ResourceExhaustedError` in Inception (BatchNormalization, 250×71×71×192) | Inception at 299×299 with batch 250 + other models in GPU memory | `InceptionScorer.__call__(batch=50)`; `gc.collect()`; or restart the runtime |
| Paper section `import ddpm_paper` crashes on Colab | The file isn't uploaded | Import guarded with try/except |
| FID cell silent for many minutes | The baseline evaluation has no print line | Expected; ~3–6 min per model |
| Models saw different batches (first Gaussian path runs) | A shared `train_data` iterator across models + unseeded t/noise | The newer sections use step-seeded streams (`blur_batch`, `stream_batch`) |

Interrupting training is safe: checkpoints every 500 steps; resume continues on the identical stream. Before starting
another section, run `del p, step_fn; gc.collect()`. Avoid interrupting exactly while a `step ... loss` line prints
(that's when it saves).

---

## 10. Next steps (prioritised)

1. **Finish the blur run** (`ladder_jump`, `ladder_reverse`, `jump_down_step_up`, `w_net`) and check whether the Gaussian ranking holds.
2. **Run the noise-routed U-Net** (cells 79–82). Compare `routed_fixed` vs `routed_dense` (quality lost vs compute saved), `routed_learned` vs `routed_fixed`, and everything vs `original_unet`.
3. **Test the ε-prediction hypothesis cheaply:** train path `unet` in Gaussian with x0- (or v-) prediction. If its gap to the original shrinks from ~85 to ~10 FID, the bricks are the problem for noise prediction.
4. **Fix the base bricks:** 3×3 stride-2 down (or Haar-wavelet lossless down/up), 64 ch at 32×32, stop at 8×8 with attention there (`PathUNet([0,1,2,1,0], base=32, mults=(2,4,4), attn_levels=(2,))`). Check that it reproduces ~125 FID.
5. **Ladder on the fixed base:** `[0,1,0,1,2,1,0]` (3 levels), plus `ladder_step_reverse`.
6. **Fair comparisons still missing:** equal compute (path `unet` trained ~2× the steps, or `unet` at base 48 ≈ 742 MFLOPs), and identical-stream re-runs for the Gaussian path models.
7. **Train longer** (the notebook's own note suggests 400 epochs; all current models are undertrained at FID 125–235).
8. **Scaling curve:** `unet`, `ladder_step`, `ladder_step_reverse` at base 16/32/64. If the ladder's gap grows with size, the "better at big scale" hope has evidence.
9. **Later:** a REPA-style representation alignment loss (align the denoiser's mid features with a pretrained DINOv2 / I-JEPA encoder; **never** Inception, which would game FID). Apply it equally to all models.

### Brainstormed architecture ideas (not built yet)
U-Net strengths to keep: multi-scale coarse-to-fine, long skips, conv bias (data-efficient), cheap at high resolution.
Weaknesses and matching ideas:
- Same compute at every noise level → **noise-dependent routing / per-pass gates** (being tested in section 8)
- Global context only once at the bottom → **ladder passes**, ideally with **shared weights per level** (extra compute without extra params)
- Skips can dominate (FreeU) → **learned skip gates α(t)**
- Aliasing from strided down/up → **Haar wavelet resampling** (lossless)
- Weak global reasoning → **a few global tokens** that all levels read/write via cross-attention (RIN-like)
- Weak timestep conditioning → **adaLN-Zero**; unstable magnitudes → **EDM2-style magnitude-preserving layers**
- Compute stuck at full resolution → put more blocks at 16×16 / 8×8 (Simple diffusion)
- Possible name: an adaptive wavelet ladder. Claim to aim for: better FID per parameter/FLOP than a U-Net, plus faster sampling via skipped passes.

---

## 11. Working preferences (user)

- Code delivery: the user often wants **code blocks in chat** while constructing, and **writes into `diffusionRedo.ipynb`** when they ask. Follow the latest instruction.
- **Don't run the user's notebook end to end locally** unless asked. Quick isolated checks of new code (tiny settings, a stand-in scorer) are fine and were useful.
- **Never change Drive checkpoint paths.**
- Never add Co-Authored-By / Claude trailers to commits.
- Runs happen on a Colab T4; keep GPU memory and session drops in mind (checkpoint everything, make cells resumable).
- Model note: Opus 5.5 is reported to beat Fable 5.1 on several agentic coding benchmarks at lower price (VentureBeat / MarkTechPost, Sep 2026); Anthropic says the real-world gap is narrower. For hard design work, raising effort to high is reasonable.

---

## 12. DGX run, 2026-11-06 (the big result: SWL)

### Setup
- DGX A100 (`ssh dgx` = `asrivastava@10.206.5.228`; key in the macOS Keychain/agent; SLURM `gpu` partition; **`--mem=128G` is mandatory and also the maximum**). The machine is shared: usually only 1 GPU is free in SLURM, and someone may run on "your" GPU outside SLURM (job 9915 died from OOM because of that).
- Code: `~/arch_bench/` on the DGX, copied to **`Desktop/diffusion/dgx_results/`** (scripts, logs, eval.json, loss.csv, 32 sample images per model as `.npy`, `samples_top.png`, `report_all.txt`).
  - `arch_bench.py`: data stream, every architecture, training (`train_multi` = all models round-robin in one process → identical step counts), FID/IS evaluation, `report`.
  - `novel.py`: Haar-wavelet spectral schedules, the **SWL** architecture, a DDIM sampler generalised to per-coefficient alpha.
- Protocol: **identical stream** for every model (same images, flips, t and noise at each step; t stratified 13/19/19/77 per noise range); batch 128; Adam 2e-4; EMA 0.999; **4,000 steps** (≈100 epochs); FID/IS on **5,000 samples**, seed 1234, DDIM 100 steps (cold sampler for blur); Keras InceptionV3 (same as Colab: the Gaussian original UNet gave 128.5 here vs 124.5 on Colab, and unet_v1 209.4 vs 209.4).

### New things tested
- **v2 "fixed bricks"** path U-Net: 3×3 stride-2 down, 64/128/128 ch, 3 levels (32→16→8), attention at 8×8.
- **SWL = Shared-Weight Wavelet Ladder** (novel): lossless Haar down/up (space-to-depth with orthonormal Haar + 1×1 conv); two ladder passes (depth 1 then depth 2) that **reuse the same per-level blocks**; a pass embedding added to the time embedding; a noise-dependent gate per pass `g_p(t) = 1 + tanh(Dense(t_emb))` (starts at 1); adaLN-Zero ResBlocks (scale/shift from the conditioning, zero-init output conv); attention at 8×8. 2.72M params. Variants: `swl_wide` (80/160/160 ch, 3.74M), `swl_p3` (3 passes 1,2,2; +1K params), `swl_long` (SWL trained to 8,000 steps).
- **Spectral wavelet schedules** (novel): a 2-level Haar transform; each band (coarse LL2 / level-2 detail / level-1 detail) gets the cosine log-SNR plus a shift computed from the band's power on real airplanes (coarse band ≈13× the mean power). `spec_eq` = shift −log(P_b/P̄) (all bands cross SNR=1 together); `spec_c2f` = +0.5·log(P_b/P̄) (fine bands destroyed earlier). Haar is orthonormal, so the pixel noise stays white; the sampler is DDIM per coefficient.

### Results: Gaussian (ε-prediction), FID on 5k samples (lower is better)
| model | params | loss | **FID** | IS |
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
| routed_learned pruned (38%) | 3.95M | 0.0602 | 292.9 | 2.02 |
Real airplanes: IS 4.91.

### Results: blur + noise (x0-prediction)
ladder_rev_v2 **136.6** · ladder_v2 141.3 · unet_v2 151.8 · ladder_v1 154.7 · original_unet 160.4 · unet_v1 162.6 · routed_dense 165.2 · routed_learned 179–180 · routed_fixed 224.1. (SWL was not run on blur yet.)

### Results: spectral schedules (FID)
| | standard (gauss) | spec_eq | spec_c2f |
|---|---|---|---|
| swl | **92.8** | 111.7 | 113.2 |
| original_unet | 128.5 | 149.9 | 182.0 |
| ladder_v2 | 128.9 | 142.4 | 196.3 |

### Conclusions
1. **SWL is the best architecture by far:** FID 92.8 vs 128.5 for the original UNet at the same 4k steps, **−28%**, with 2.7M params (smaller than ladder_v2). Trained to 8k steps it reaches **FID 72.0, IS 4.42** (real 4.91). The samples are visibly smoother and more coherent. *Caveats:* one seed; SWL uses ~2.4× the original's sampling time; 8k vs 4k isn't equal training.
2. **The bricks hypothesis is confirmed:** the v1→v2 brick fix took the Gaussian plain U-Net from 209 to 150, and the ladder from 176 to 129.
3. The ladder helps on the fixed bricks (unet_v2 150 → ladder_v2 129); **stepping forward beats reverse in Gaussian (129 vs 146), but reverse is best in blur (137)**.
4. Widening SWL helps a little (87 vs 93; 37% more params); **a third shared pass hurts** (100.5). **More training helps the most** (72), and SWL is still improving.
5. **The spectral wavelet schedules hurt every architecture at 4k steps.** SWL is the most robust to them (+19 FID vs +53 for the original UNet under c2f). Not worth pursuing in this form.
6. **The noise-routed U-Net failed:** dense 193, fixed routing 228, and pruning learned gates collapsed it (293). Routing blocks by noise level doesn't work at this scale/training length. Pruning by gate < 0.1 is too aggressive (the network still relied on those blocks).

### Next steps (updated)
- Train **SWL / swl_wide longer** (16k–30k steps) with 2–3 seeds; check that it keeps the lead over original_unet trained equally long.
- Run **SWL on blur** and with **v-prediction**.
- Ablate SWL's pieces one at a time (shared weights vs separate, Haar vs strided conv, adaLN-Zero vs additive time, gates on/off) to find what gives the −28%.
- Compare at equal sampling compute (SWL is ~2–3× slower per sample than the original UNet).
