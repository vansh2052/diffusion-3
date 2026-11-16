# To do next (blur-diffusion formulas)

Written 16 Nov 2026. Read `context.md` (setup), `eval.md` (numbers) and `decision.md` (conclusions) first.
Each item says where to run it and roughly how long it takes.

---

## A. Finish the laptop prototypes (Mac, low-power queue, ~10 min each)

Queue file: `queue_micro.txt`; runner: `./run_micro.sh` (efficiency cores only); table: `python blur_micro.py report`.

- [ ] Read the results of the runs still in the queue on 16 Nov: `lin1.25_x0_hf0.01` (additive spectral boost),
      `lin1.25_x0` with `eta1q2` (η = 1 + quadratic steps), `gauss_x0`, `she1.25_x0`, `lin1.25_x0_sc` (self-conditioning).
      Update `eval.md` / `decision.md` with them.
- [ ] **Second seed** for the top 3 formulas (`... 3000 1`) to check the gaps are real.
- [ ] **Re-check the sampler gains** (`eta1`, `eta0q2`, `eta1q2`) on `old_x0`, `wien0.5_x0`, `gauss_eps`:
      `python blur_micro.py reeval <run> eta1,eta0q2,eta1q2`.
- [ ] Sweep the boost strength if `hf0.01` helps: `hf0.003`, `hf0.03` (add to `PROCS` in `blur_micro.py`).
- [ ] Blur-strength sweep with x0 + best sampler: `lin0.6_x0`, `lin2.5_x0`, `sin2.5_x0` (add to `PROCS`).

## B. Scale up on the DGX (overnight; `blur_proto.py` is the 32×32 version of the same framework)

Connect with `ssh dgx` (turn ProtonVPN off first, or exclude `10.206.5.228` from it). SLURM rules: partition `gpu`,
`--mem=128G`, 1 GPU. Copy `blur_proto.py` + `arch_bench.py` + `novel.py` to `~/arch_bench/` and set `BENCH_ROOT`.

- [ ] Port the new pieces from `blur_micro.py` into `blur_proto.py` first: Wiener blur (`wiener` family), additive boost
      (`hf`), self-conditioning (`sc`), the `eta<η>q<p>` samplers and `reeval`. (`blur_proto.py` already has the DCT framework,
      `u`, `suw`, `she`, and Inception FID/IS on 5,000 samples.)
- [ ] 32×32, **original `UNet` base 32** and **SWL** (the best DGX architecture), 4,000 and 8,000 steps:
      `gauss_eps` (baseline), `old_x0` (cold vs ddim), `lin2.5_x0`, `wien0.5_x0`, best of `hf`/`sc`, each with
      `eta0`, `eta1`, `eta0q2`, `eta1q2`. Compare with the DGX numbers: original UNet Gaussian 128.5, old blur 160.4, SWL Gaussian 92.8.
- [ ] Revisit **u** and **u + SUW** at full scale. They lost here because the tiny model couldn't fit colour, which may not
      hold for a 2–3M-parameter model trained longer.
- [ ] If blur beats Gaussian at 32×32, write the formula up as the project's blur-diffusion method.

## C. Ideas not tried yet
- [ ] **Per-frequency η**: stochastic only where the signal is hidden (`a_k²P_k ≪ s_k²`), deterministic elsewhere.
- [ ] **Step spacing from the spectrum**: place sampling steps uniformly in the median-frequency log-SNR instead of in t.
- [ ] **Hybrid loss** `MSE(x0) + β·MSE(u)` (the same as the additive boost, but time-dependent).
- [ ] **Two-stage sampling**: x0 model at high noise, eps/u model at low noise.
- [ ] **REPA-style alignment** on the best blur model (DINOv2 features, never Inception).

## D. Housekeeping
- [ ] Merge this branch (`blur-prototypes-(16-Nov)`) into `main` once the results are final.
- [ ] Keep the laptop cool: always run prototypes with `taskpolicy -b` and ≤ 4 threads (`run_micro.sh` does this).
