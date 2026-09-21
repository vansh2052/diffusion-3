"""Paper-style DDPM (Ho et al. 2020) on CIFAR-10 with a selectable noise schedule,
plus FID and Inception Score evaluation.

    python ddpm_paper.py stats                                   # reference statistics
    python ddpm_paper.py train --schedule linear --steps 50000   # train (resumes) + final eval
    python ddpm_paper.py eval  --schedule linear --n-samples 10000

Paper setup: 35.7M-parameter U-Net, batch 128, Adam 2e-4, EMA 0.9999, dropout 0.1,
random horizontal flips, 1000 diffusion steps, noise-prediction MSE loss.
Practical additions (not in the paper text): bf16 mixed precision + XLA for speed,
5000-step learning-rate warm-up, global gradient-norm clipping at 1.0, EMA warm-up.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

TIMESTEPS = 1000
REF_PATH = "fid_reference_cifar10_train.npz"
REF_INFO = "fid_reference.json"


# ----------------------------------------------------------------------------
# Model (Ho et al. CIFAR-10 config)
# ----------------------------------------------------------------------------
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = tf.exp(-math.log(10000.0) * tf.range(half, dtype=tf.float32) / (half - 1))
    args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


class ResBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.norm1 = layers.GroupNormalization(groups=32)
        self.conv1 = layers.Conv2D(out_ch, 3, padding="same")
        self.temb_proj = layers.Dense(out_ch)
        self.norm2 = layers.GroupNormalization(groups=32)
        self.drop = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(out_ch, 3, padding="same")
        self.shortcut = layers.Conv2D(out_ch, 1) if in_ch != out_ch else None

    def call(self, x, temb, training=False):
        h = self.conv1(tf.nn.swish(self.norm1(x)))
        h = h + self.temb_proj(tf.nn.swish(temb))[:, None, None, :]
        h = self.drop(tf.nn.swish(self.norm2(h)), training=training)
        h = self.conv2(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


class AttnBlock(layers.Layer):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.norm = layers.GroupNormalization(groups=32)
        self.q = layers.Dense(ch)
        self.k = layers.Dense(ch)
        self.v = layers.Dense(ch)
        self.proj = layers.Dense(ch)

    def call(self, x):
        b = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]
        y = tf.reshape(self.norm(x), [b, h * w, self.ch])
        q, k, v = self.q(y), self.k(y), self.v(y)
        att = tf.matmul(q, k, transpose_b=True) * (self.ch ** -0.5)
        att = tf.nn.softmax(att, axis=-1)
        y = self.proj(tf.matmul(att, v))
        return x + tf.reshape(y, [b, h, w, self.ch])


class DDPMUNet(tf.keras.Model):
    def __init__(self, ch=128, ch_mult=(1, 2, 2, 2), num_res=2, attn_res=(16,),
                 dropout=0.1, img=32):
        super().__init__()
        self.ch = ch
        self.temb1 = layers.Dense(ch * 4)
        self.temb2 = layers.Dense(ch * 4)
        self.conv_in = layers.Conv2D(ch, 3, padding="same")

        self.res_layers, self.attn_layers = [], []
        self.down_layers, self.up_layers = [], []
        self.down_plan, self.up_plan = [], []

        def add_res(i_ch, o_ch, res):
            self.res_layers.append(ResBlock(i_ch, o_ch, dropout))
            attn_idx = -1
            if res in attn_res:
                self.attn_layers.append(AttnBlock(o_ch))
                attn_idx = len(self.attn_layers) - 1
            return len(self.res_layers) - 1, attn_idx

        cur, res = ch, img
        skip_chs = [ch]
        for i, m in enumerate(ch_mult):
            for _ in range(num_res):
                r, a = add_res(cur, ch * m, res)
                cur = ch * m
                self.down_plan.append(("blk", r, a))
                skip_chs.append(cur)
            if i != len(ch_mult) - 1:
                self.down_layers.append(layers.Conv2D(cur, 3, strides=2, padding="same"))
                self.down_plan.append(("down", len(self.down_layers) - 1, -1))
                skip_chs.append(cur)
                res //= 2

        self.mid_res1 = ResBlock(cur, cur, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid_res2 = ResBlock(cur, cur, dropout)

        for i in reversed(range(len(ch_mult))):
            for _ in range(num_res + 1):
                skip = skip_chs.pop()
                r, a = add_res(cur + skip, ch * ch_mult[i], res)
                cur = ch * ch_mult[i]
                self.up_plan.append(("blk", r, a))
            if i != 0:
                self.up_layers.append(layers.Conv2D(cur, 3, padding="same"))
                self.up_plan.append(("up", len(self.up_layers) - 1, -1))
                res *= 2

        self.norm_out = layers.GroupNormalization(groups=32)
        self.conv_out = layers.Conv2D(3, 3, padding="same")
        self.upsample = layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        x, t = inputs
        temb = self.temb2(tf.nn.swish(self.temb1(timestep_embedding(t, self.ch))))

        h = self.conv_in(x)
        hs = [h]
        for kind, idx, attn in self.down_plan:
            if kind == "blk":
                h = self.res_layers[idx](h, temb, training=training)
                if attn >= 0:
                    h = self.attn_layers[attn](h)
            else:
                h = self.down_layers[idx](h)
            hs.append(h)

        h = self.mid_res1(h, temb, training=training)
        h = self.mid_attn(h)
        h = self.mid_res2(h, temb, training=training)

        for kind, idx, attn in self.up_plan:
            if kind == "blk":
                h = tf.concat([h, hs.pop()], axis=-1)
                h = self.res_layers[idx](h, temb, training=training)
                if attn >= 0:
                    h = self.attn_layers[attn](h)
            else:
                h = self.up_layers[idx](self.upsample(h))

        return self.conv_out(tf.nn.swish(self.norm_out(h)))


# ----------------------------------------------------------------------------
# Noise schedules
# ----------------------------------------------------------------------------
def make_schedule(name, timesteps=TIMESTEPS):
    """Return float64 numpy arrays for the chosen schedule.

    linear: beta from 1e-4 to 0.02 (Ho et al. 2020).
    cosine: Nichol & Dhariwal 2021, betas clipped at 0.999.
    """
    if name == "linear":
        betas = np.linspace(1e-4, 0.02, timesteps, dtype=np.float64)
    elif name == "cosine":
        s = 0.008
        steps = np.arange(timesteps + 1, dtype=np.float64)
        f = np.cos(((steps / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
        ab = f / f[0]
        betas = np.clip(1.0 - ab[1:] / ab[:-1], 0.0, 0.999)
    else:
        raise ValueError(f"unknown schedule: {name}")
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_prev = np.concatenate([[1.0], alpha_bar[:-1]])
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar,
            "alpha_bar_prev": alpha_bar_prev}


class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, warmup):
        super().__init__()
        self.lr = lr
        self.warmup = warmup

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.lr * tf.minimum(1.0, (step + 1.0) / float(self.warmup))

    def get_config(self):
        return {"lr": self.lr, "warmup": self.warmup}


# ----------------------------------------------------------------------------
# Evaluation: FID and Inception Score (Keras InceptionV3, ImageNet weights)
# ----------------------------------------------------------------------------
class InceptionScorer:
    """Pool features (2048-d) for FID and class probabilities for IS.

    Uses Keras' InceptionV3, not the exact TF-GAN network of the original FID/IS
    code, so absolute numbers differ slightly from published ones. Compare
    against the real-data reference numbers written by `stats`.
    """

    def __init__(self):
        old = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        base = tf.keras.applications.InceptionV3(include_top=True, weights="imagenet")
        self.model = tf.keras.Model(
            base.input, [base.get_layer("avg_pool").output, base.output])
        tf.keras.mixed_precision.set_global_policy(old)

    def __call__(self, imgs_uint8, batch=250):
        feats, probs = [], []
        for i in range(0, len(imgs_uint8), batch):
            x = tf.cast(imgs_uint8[i:i + batch], tf.float32)
            x = tf.image.resize(x, (299, 299), method="bilinear") / 127.5 - 1.0
            f, p = self.model(x, training=False)
            feats.append(f.numpy())
            probs.append(p.numpy())
        return np.concatenate(feats).astype(np.float64), np.concatenate(probs).astype(np.float64)


def fid_from_features(feats, mu_ref, sigma_ref):
    import scipy.linalg
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    diff = mu - mu_ref
    covmean = scipy.linalg.sqrtm(sigma.dot(sigma_ref))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma + offset).dot(sigma_ref + offset))
    covmean = np.real(covmean)
    return float(diff.dot(diff) + np.trace(sigma) + np.trace(sigma_ref) - 2.0 * np.trace(covmean))


def inception_score(probs, splits=10):
    scores = []
    for part in np.array_split(probs, min(splits, len(probs))):
        py = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(axis=1).mean()
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def ensure_reference(scorer, ref_path=REF_PATH, info_path=REF_INFO, limit=None):
    """Statistics of the real CIFAR-10 training set (cached), plus sanity numbers."""
    if os.path.exists(ref_path):
        d = np.load(ref_path)
        return d["mu"], d["sigma"]
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    if limit:
        x_train, x_test = x_train[:limit], x_test[:limit]
    t0 = time.time()
    f_tr, p_tr = scorer(x_train)
    mu, sigma = f_tr.mean(axis=0), np.cov(f_tr, rowvar=False)
    f_te, p_te = scorer(x_test)
    info = {
        "n_train": int(len(x_train)), "n_test": int(len(x_test)),
        "fid_test_vs_train": fid_from_features(f_te, mu, sigma),
        "is_train": inception_score(p_tr), "is_test": inception_score(p_te),
        "seconds": round(time.time() - t0, 1),
    }
    tmp = ref_path + ".tmp.npz"
    np.savez(tmp, mu=mu, sigma=sigma)
    os.replace(tmp, ref_path)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print("reference statistics:", info, flush=True)
    return mu, sigma


def save_grid(imgs_uint8, path, n=64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    side = int(math.sqrt(min(n, len(imgs_uint8))))
    g = imgs_uint8[:side * side].reshape(side, side, 32, 32, 3).transpose(0, 2, 1, 3, 4)
    g = g.reshape(side * 32, side * 32, 3)
    g = np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)
    mpimg.imsave(path, g)


# ----------------------------------------------------------------------------
# Training / sampling run
# ----------------------------------------------------------------------------
class Run:
    def __init__(self, schedule, out_dir=None, precision="bf16", xla=True, batch=128,
                 lr=2e-4, warmup=5000, grad_clip=1.0, ema_decay=0.9999, dropout=0.1,
                 ckpt_every=5000, variance="large"):
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        tf.keras.mixed_precision.set_global_policy(
            "mixed_bfloat16" if precision == "bf16" else "float32")

        self.schedule_name = schedule
        self.out_dir = out_dir or os.path.join("runs", schedule)
        os.makedirs(self.out_dir, exist_ok=True)
        self.xla, self.batch, self.ckpt_every = bool(xla), batch, ckpt_every
        self.ema_decay, self.variance = ema_decay, variance

        self.sched = {k: tf.constant(v, tf.float32)
                      for k, v in make_schedule(schedule).items()}

        self.model = DDPMUNet(dropout=dropout)
        self.model([tf.zeros((1, 32, 32, 3)), tf.zeros((1,), tf.int32)])
        self.optimizer = tf.keras.optimizers.Adam(
            Warmup(lr, warmup), global_clipnorm=grad_clip)
        self.optimizer.build(self.model.trainable_variables)
        self.ema = [tf.Variable(v, trainable=False) for v in self.model.trainable_variables]
        self.step_var = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                        ema=self.ema, step=self.step_var)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, os.path.join(self.out_dir, "ckpt"), max_to_keep=3)
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.optimizer.iterations.assign(self.step_var)
            print(f"resumed from {self.manager.latest_checkpoint} at step {self.step}",
                  flush=True)

        self._train_step = self._build_train_step()
        self._ddpm_step, self._ddim_step = self._build_samplers()

    @property
    def step(self):
        return int(self.optimizer.iterations.numpy())

    def _build_train_step(self):
        S, model, opt, ema, decay = self.sched, self.model, self.optimizer, self.ema, self.ema_decay

        @tf.function(jit_compile=self.xla)
        def train_step(x0):
            b = tf.shape(x0)[0]
            t = tf.random.uniform((b,), 0, TIMESTEPS, dtype=tf.int32)
            noise = tf.random.normal(tf.shape(x0))
            ab = tf.gather(S["alpha_bar"], t)[:, None, None, None]
            xt = tf.sqrt(ab) * x0 + tf.sqrt(1.0 - ab) * noise
            with tf.GradientTape() as tape:
                pred = tf.cast(model([xt, t], training=True), tf.float32)
                loss = tf.reduce_mean(tf.square(noise - pred))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            n = tf.cast(opt.iterations, tf.float32)
            d = tf.minimum(decay, (1.0 + n) / (10.0 + n))
            for e, v in zip(ema, model.trainable_variables):
                e.assign(d * e + (1.0 - d) * v)
            return loss

        return train_step

    def _build_samplers(self):
        S, model = self.sched, self.model
        large = self.variance == "large"

        @tf.function(jit_compile=self.xla)
        def ddpm_step(x, t):
            eps = tf.cast(model([x, tf.fill([x.shape[0]], t)], training=False), tf.float32)
            ab, abp = S["alpha_bar"][t], S["alpha_bar_prev"][t]
            beta, alpha = S["betas"][t], S["alphas"][t]
            x0 = tf.clip_by_value(
                (x - tf.sqrt(1.0 - ab) * eps) / tf.sqrt(ab), -1.0, 1.0)
            mean = (tf.sqrt(abp) * beta / (1.0 - ab) * x0
                    + tf.sqrt(alpha) * (1.0 - abp) / (1.0 - ab) * x)
            var = beta if large else beta * (1.0 - abp) / (1.0 - ab)
            noise = tf.random.normal(tf.shape(x))
            return mean + tf.where(t > 0, tf.sqrt(var), 0.0) * noise

        @tf.function(jit_compile=self.xla)
        def ddim_step(x, t, t_prev):
            eps = tf.cast(model([x, tf.fill([x.shape[0]], t)], training=False), tf.float32)
            ab = S["alpha_bar"][t]
            abp = tf.where(t_prev >= 0, S["alpha_bar"][tf.maximum(t_prev, 0)], 1.0)
            x0 = tf.clip_by_value(
                (x - tf.sqrt(1.0 - ab) * eps) / tf.sqrt(ab), -1.0, 1.0)
            eps = (x - tf.sqrt(ab) * x0) / tf.sqrt(1.0 - ab)
            return tf.sqrt(abp) * x0 + tf.sqrt(1.0 - abp) * eps

        return ddpm_step, ddim_step

    # -- EMA weights ----------------------------------------------------------
    def use_ema(self):
        backup = [tf.identity(v) for v in self.model.trainable_variables]
        for v, e in zip(self.model.trainable_variables, self.ema):
            v.assign(e)
        return backup

    def restore_weights(self, backup):
        for v, b in zip(self.model.trainable_variables, backup):
            v.assign(b)

    # -- sampling -------------------------------------------------------------
    def sample(self, n, sampler="ddpm", steps=TIMESTEPS, batch=500):
        out = []
        for _ in range(math.ceil(n / batch)):
            x = tf.random.normal((batch, 32, 32, 3))
            if sampler == "ddpm":
                for t in reversed(range(TIMESTEPS)):
                    x = self._ddpm_step(x, tf.constant(t, tf.int32))
            else:
                idx = np.linspace(0, TIMESTEPS - 1, steps).round().astype(int)[::-1]
                for i, t in enumerate(idx):
                    t_prev = idx[i + 1] if i + 1 < len(idx) else -1
                    x = self._ddim_step(x, tf.constant(int(t), tf.int32),
                                        tf.constant(int(t_prev), tf.int32))
            out.append(x.numpy())
        x = np.concatenate(out)[:n]
        return np.clip(np.round((x + 1.0) * 127.5), 0, 255).astype(np.uint8)

    def evaluate(self, scorer, ref, n_samples, sampler, sample_steps, sample_batch, tag):
        backup = self.use_ema()
        t0 = time.time()
        imgs = self.sample(n_samples, sampler, sample_steps, sample_batch)
        self.restore_weights(backup)
        t_sample = time.time() - t0
        feats, probs = scorer(imgs)
        fid = fid_from_features(feats, *ref)
        is_mean, is_std = inception_score(probs)
        entry = {"tag": tag, "step": self.step, "schedule": self.schedule_name,
                 "sampler": sampler, "sample_steps": sample_steps if sampler == "ddim" else TIMESTEPS,
                 "variance": self.variance, "n_samples": n_samples,
                 "fid": fid, "is_mean": is_mean, "is_std": is_std,
                 "sample_seconds": round(t_sample, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        path = os.path.join(self.out_dir, "metrics.json")
        data = json.load(open(path)) if os.path.exists(path) else []
        data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        save_grid(imgs, os.path.join(self.out_dir, f"samples_{tag}_step{self.step}.png"))
        print(f"[{tag}] step {self.step} {sampler}: FID {fid:.2f}  IS {is_mean:.2f}+-{is_std:.2f} "
              f"({n_samples} samples, {t_sample:.0f}s sampling)", flush=True)
        return entry

    # -- training -------------------------------------------------------------
    def save(self):
        self.step_var.assign(self.step)
        self.manager.save(checkpoint_number=self.step)

    def train(self, total_steps, data_limit=None, log_every=200, on_monitor=None,
              monitor_every=0):
        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
        x = x_train[:data_limit] if data_limit else x_train
        x = x.astype("float32") / 127.5 - 1.0
        ds = (tf.data.Dataset.from_tensor_slices(x).shuffle(len(x)).repeat()
              .map(tf.image.random_flip_left_right, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(self.batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
        it = iter(ds)

        loss_path = os.path.join(self.out_dir, "loss.csv")
        if not os.path.exists(loss_path):
            with open(loss_path, "w") as f:
                f.write("step,loss\n")
        step, losses, t0 = self.step, [], time.time()
        last_log = step
        print(f"training {self.schedule_name}: step {step} -> {total_steps}", flush=True)
        while step < total_steps:
            losses.append(self._train_step(next(it)))
            step += 1
            if step % log_every == 0 or step == total_steps:
                mean = float(tf.reduce_mean(tf.stack(losses)))
                rate = (step - last_log) / (time.time() - t0)
                losses, t0, last_log = [], time.time(), step
                with open(loss_path, "a") as f:
                    f.write(f"{step},{mean:.6f}\n")
                print(f"step {step}/{total_steps}  loss {mean:.4f}  {rate:.1f} steps/s", flush=True)
            if step % self.ckpt_every == 0 or step == total_steps:
                self.save()
            if monitor_every and on_monitor and step % monitor_every == 0 and step < total_steps:
                on_monitor()
                t0, last_log = time.time(), step


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--schedule", choices=["linear", "cosine"], required=True)
        sp.add_argument("--out-dir")
        sp.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
        sp.add_argument("--xla", type=int, default=1)
        sp.add_argument("--variance", choices=["large", "small"], default="large")
        sp.add_argument("--sample-batch", type=int, default=500)
        sp.add_argument("--smoke", action="store_true", help="tiny CPU check")

    sub.add_parser("stats")
    tr = sub.add_parser("train")
    common(tr)
    tr.add_argument("--steps", type=int, required=True)
    tr.add_argument("--eval-every", type=int, default=0)
    tr.add_argument("--monitor-samples", type=int, default=5000)
    tr.add_argument("--final-samples", type=int, default=0)
    tr.add_argument("--final-sampler", choices=["ddpm", "ddim"], default="ddpm")
    tr.add_argument("--final-sample-steps", type=int, default=100)
    ev = sub.add_parser("eval")
    common(ev)
    ev.add_argument("--n-samples", type=int, default=10000)
    ev.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ev.add_argument("--sample-steps", type=int, default=100)
    a = p.parse_args()

    smoke = getattr(a, "smoke", False) or False
    ref_kw = dict(ref_path="fid_reference_smoke.npz", info_path="fid_reference_smoke.json",
                  limit=200) if smoke else {}

    if a.cmd == "stats":
        ensure_reference(InceptionScorer(), **ref_kw)
        return

    out_dir = a.out_dir or (os.path.join("runs_smoke", a.schedule) if smoke else None)
    run = Run(a.schedule, out_dir=out_dir, precision=a.precision, xla=bool(a.xla),
              batch=4 if smoke else 128, ckpt_every=2 if smoke else 5000,
              variance=a.variance)
    scorer = InceptionScorer()
    ref = ensure_reference(scorer, **ref_kw)
    sb = 8 if smoke else a.sample_batch

    if a.cmd == "train":
        with open(os.path.join(run.out_dir, "train_args.json"), "w") as f:
            json.dump(vars(a), f, indent=2)
        monitor = None
        if a.eval_every and not smoke:
            monitor = lambda: run.evaluate(scorer, ref, a.monitor_samples, "ddim", 100, sb, "monitor")
        run.train(a.steps, data_limit=64 if smoke else None, log_every=1 if smoke else 200,
                  on_monitor=monitor, monitor_every=a.eval_every)
        n_final = 8 if smoke else a.final_samples
        if n_final:
            run.evaluate(scorer, ref, n_final, "ddim" if smoke else a.final_sampler,
                         5 if smoke else a.final_sample_steps, sb, "final")
    else:
        run.evaluate(scorer, ref, 8 if smoke else a.n_samples,
                     "ddim" if smoke else a.sampler,
                     5 if smoke else a.sample_steps, sb, "eval")


if __name__ == "__main__":
    main()
