# To do next (blur-diffusion formulas)

Written 16 Nov 2026 at the end of the prototype session. Read `context.md` (setup), `eval.md` (numbers) and `decision.md`
(conclusions) first. **Current best:** stochastic heat equation + x0 + additive spectral boost (β = 0.01) + η = 1 with
quadratic steps = **KID×1000 97.5** (old pipeline 219.6, Gaussian eps 208.7), on 16×16 prototypes.

---

## 1. Most important open question
**Does blur help at all at full scale?** On the 16×16 prototypes, Gaussian x0 with the same sampler (108.9) ≈ mild blur (110.0)
≈ strong blur (110.5). All the big gains came from the sampler, the x0 target and the spectral boost. The DGX runs (§3)
must include Gaussian controls with exactly the same target, loss and sampler.

## 2. Finish the laptop prototypes (Mac, ~10 min each, cool: `run_micro.sh` uses efficiency cores only)
Add a line `<process> <steps> <seed> [samplers]` to `queue_micro.txt`, start `./run_micro.sh`, read `python blur_micro.py report`.

- [ ] **Second seed** for the top 5 (`... 3000 1 eta0,eta1q2`): `she1.25_x0_hf0.01`, `she1.25_x0`, `lin1.25_x0_hf0.01`,
      `gauss_x0_hf0.01`, `gauss_x0`. The top results are within ~10 KID of each other; a second seed tells which gaps are real.
- [ ] **Self-conditioning** with the best setup (`lin1.25_x0_sc`, `she1.25_x0_hf0.01` + `sc=True`): removed from today's queue for time.
- [ ] **Heat-equation strength:** `she0.6_x0_hf0.01`, `she2.5_x0_hf0.01` (add to `PROCS`).
- [ ] **Wiener-blur + boost / + heat equation:** `wien0.5_x0_hf0.01`, and a "Wiener heat equation" (Wiener attenuation with coloured noise).
- [ ] **Boost fine-tuning:** β = 0.003, 0.005, 0.02.
- [ ] **Sampler:** 80 steps instead of 40 (`EVAL_STEPS`), η = 0.8, and **per-frequency η** (stochastic only where `a_k²P_k ≪ s_k²`).
- [ ] Longer training (6,000 steps) for the top 3, to see whether the ranking holds as models improve.

## 3. Scale up on the DGX (overnight; `blur_proto.py` is the 32×32 version of the framework)
Connect: `ssh dgx` (turn ProtonVPN off first, or exclude `10.206.5.228` in its split tunnelling). SLURM rules: partition `gpu`,
`#SBATCH --mem=128G` (required, and the maximum), 1 GPU. Code goes in `~/arch_bench/` next to `arch_bench.py` and `novel.py`.

- [ ] **Port to `blur_proto.py`** the pieces that only exist in `blur_micro.py`: `wiener` family, additive boost (`hf`),
      self-conditioning (`sc`), the `eta<η>q<p>` samplers, `reeval`, and the `old` pixel process with both samplers.
      (`blur_proto.py` already has the DCT framework, eps/x0/v/u, `suw`, `she`, Inception FID/IS on 5,000 samples.)
- [ ] 32×32, **original `UNet` base 32** and **SWL**, 4,000 then 8,000 steps, each with `eta0` and `eta1q2`:
  - `gauss_eps` (DGX baseline FID 128.5; SWL 92.8), `gauss_x0`, `gauss_x0_hf0.01`
  - `old_x0` with `cold` and `ddim` (DGX baseline FID 160.4)
  - `lin2.5_x0`, `lin2.5_x0_hf0.01`, `she2.5_x0`, `she2.5_x0_hf0.01`, `wien0.5_x0`
  - (blur widths double at 32×32: 1.25 px at 16×16 = 2.5 px at 32×32)
- [ ] Re-test **u** and **u + SUW** at full scale: they lost on the tiny model because it couldn't fit colour.
- [ ] If blur beats Gaussian at 32×32 with matched target/loss/sampler, write it up as the project's blur-diffusion method.

## 4. Ideas not tried yet
- [ ] **Step spacing from the spectrum:** place steps uniformly in the median-frequency log-SNR instead of in t.
- [ ] **Time-dependent boost:** `MSE(x0) + β·MSE(u)` (adds `β(a²/s² + 1/P_k)`, emphasising detail at low noise).
- [ ] **Two-stage sampling:** x0 model at high noise, eps model at low noise.
- [ ] **REPA-style alignment** on the best blur model (DINOv2 features, never Inception).

## 5. Housekeeping
- [ ] Merge branch `blur-prototypes-(16-Nov)` into `main` (PR on GitHub) once the second-seed results are in.
- [ ] Keep the laptop cool: prototypes always with `taskpolicy -b` and ≤ 4 threads.
- [ ] Moving to another computer: clone the repo, then ask a new Claude Code session to read `RESEARCH_NOTES.md` and the files in
      `BTP-Diffusion-dgx/blur_lab/` (or copy the chat file from `~/.claude/projects/...` and run `claude --resume`).
