"""Novel pipeline: wavelet-spectral noise schedules + Shared-Weight Wavelet Ladder (SWL).

Imports everything from arch_bench (data stream, original UNet, ladder_v2, eval/FID tools).
kinds: gauss (standard cosine), spec_eq (band-equalised SNR), spec_c2f (exaggerated coarse-to-fine).
"""
import os, sys, json, time, gc
sys.argv_backup = sys.argv; sys.argv = sys.argv[:1]
import arch_bench as ab
sys.argv = sys.argv_backup
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

ROOT = ab.ROOT


# ---------------------------------------------------------------- orthonormal 2-level Haar (Mallat layout)
def _haar1(x):
    a, b = x[:, 0::2, 0::2], x[:, 0::2, 1::2]
    c, d = x[:, 1::2, 0::2], x[:, 1::2, 1::2]
    ll, lh = (a + b + c + d) / 2, (a - b + c - d) / 2
    hl, hh = (a + b - c - d) / 2, (a - b - c + d) / 2
    return tf.concat([tf.concat([ll, lh], 2), tf.concat([hl, hh], 2)], 1)


def _ihaar1(w):
    h, ww = w.shape[1] // 2, w.shape[2] // 2
    ll, lh, hl, hh = w[:, :h, :ww], w[:, :h, ww:], w[:, h:, :ww], w[:, h:, ww:]
    a, b = (ll + lh + hl + hh) / 2, (ll - lh + hl - hh) / 2
    c, d = (ll + lh - hl - hh) / 2, (ll - lh - hl + hh) / 2
    top = tf.reshape(tf.stack([a, b], 3), (-1, h, 2 * ww, w.shape[3]))
    bot = tf.reshape(tf.stack([c, d], 3), (-1, h, 2 * ww, w.shape[3]))
    return tf.reshape(tf.stack([top, bot], 2), (-1, 2 * h, 2 * ww, w.shape[3]))


def haar2(x):
    w = _haar1(x)
    tl = _haar1(w[:, :16, :16])
    return tf.concat([tf.concat([tl, w[:, :16, 16:]], 2), w[:, 16:, :]], 1)


def ihaar2(w):
    tl = _ihaar1(w[:, :16, :16])
    return _ihaar1(tf.concat([tf.concat([tl, w[:, :16, 16:]], 2), w[:, 16:, :]], 1))


BAND = np.full((32, 32), 2, np.int32)     # 2 = level-1 detail (finest)
BAND[:16, :16] = 1                        # level-2 detail
BAND[:8, :8] = 0                          # coarse LL2


def band_shifts(kind):
    """log-SNR shift per band, from the data's per-band power."""
    x = tf.constant(ab.load_data()[:2000])
    w = haar2(x).numpy()
    power = np.array([np.mean(w[:, BAND == b] ** 2) for b in range(3)])
    ref = np.mean(w ** 2)
    if kind == "spec_eq":
        return -np.log(power / ref)                 # every band crosses SNR=1 together
    return 0.5 * np.log(power / ref)                # spec_c2f: fine bands lose even earlier


def alpha_map(kind):
    """(T, 32, 32, 1) per-coefficient alpha_bar in wavelet layout."""
    ab_ = np.clip(ab.ALPHA_BAR_NP.astype(np.float64), 1e-8, 1 - 1e-8)
    if kind == "gauss":
        return None
    logsnr = np.log(ab_ / (1 - ab_))[:, None]
    sh = band_shifts(kind)
    per_band = 1 / (1 + np.exp(-(logsnr + sh[None, :])))          # (T, 3)
    per_band[-1] = np.minimum(per_band[-1], 1e-4)                  # terminal state ~ pure noise
    m = per_band[:, BAND]                                          # (T, 32, 32)
    print(kind, "band shifts", np.round(sh, 2), "alpha_bar @t=500 per band", np.round(per_band[500], 3), flush=True)
    return tf.constant(m[..., None].astype(np.float32))


def noisy(kind, A, x0, t, noise):
    if A is None:
        a = tf.gather(ab.ALPHA_BAR, t)[:, None, None, None]
        return tf.sqrt(a) * x0 + tf.sqrt(1 - a) * noise
    a = tf.gather(A, t)
    return ihaar2(tf.sqrt(a) * haar2(x0) + tf.sqrt(1 - a) * haar2(noise))


# ---------------------------------------------------------------- SWL architecture
class AdaRes(layers.Layer):
    """ResBlock with adaLN-Zero conditioning (scale/shift from cond, zero-init output conv)."""
    def __init__(self, in_ch, ch):
        super().__init__()
        self.n1 = layers.GroupNormalization(8, center=False, scale=False)
        self.m1 = layers.Dense(2 * in_ch, kernel_initializer="zeros")
        self.c1 = layers.Conv2D(ch, 3, padding="same")
        self.n2 = layers.GroupNormalization(8, center=False, scale=False)
        self.m2 = layers.Dense(2 * ch, kernel_initializer="zeros")
        self.c2 = layers.Conv2D(ch, 3, padding="same", kernel_initializer="zeros")
        self.skip = layers.Conv2D(ch, 1) if in_ch != ch else None

    def call(self, x, c):
        s1, b1 = tf.split(self.m1(c)[:, None, None, :], 2, -1)
        h = self.c1(tf.nn.silu(self.n1(x) * (1 + s1) + b1))
        s2, b2 = tf.split(self.m2(c)[:, None, None, :], 2, -1)
        h = self.c2(tf.nn.silu(self.n2(h) * (1 + s2) + b2))
        return h + (self.skip(x) if self.skip is not None else x)


class HaarDown(layers.Layer):
    def __init__(self, ch):
        super().__init__(); self.proj = layers.Conv2D(ch, 1)
    def call(self, x):
        return self.proj(tf.nn.space_to_depth(x, 2) if False else _haar_s2d(x))


class HaarUp(layers.Layer):
    def __init__(self, ch):
        super().__init__(); self.ch = ch; self.proj = layers.Conv2D(4 * ch, 1)
    def call(self, x):
        return _haar_d2s(self.proj(x), self.ch)


def _haar_s2d(x):
    a, b = x[:, 0::2, 0::2], x[:, 0::2, 1::2]
    c, d = x[:, 1::2, 0::2], x[:, 1::2, 1::2]
    return tf.concat([(a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2], -1)


def _haar_d2s(y, ch):
    ll, lh, hl, hh = tf.split(y, 4, -1)
    a, b = (ll + lh + hl + hh) / 2, (ll - lh + hl - hh) / 2
    c, d = (ll + lh - hl - hh) / 2, (ll - lh - hl + hh) / 2
    h, w = y.shape[1], y.shape[2]
    top = tf.reshape(tf.stack([a, b], 3), (-1, h, 2 * w, ch))
    bot = tf.reshape(tf.stack([c, d], 3), (-1, h, 2 * w, ch))
    return tf.reshape(tf.stack([top, bot], 2), (-1, 2 * h, 2 * w, ch))


class SWL(tf.keras.Model):
    """Shared-Weight Wavelet Ladder: passes of depth 1 then 2 reuse the same per-level blocks;
    each pass's update is scaled by a noise-dependent gate g_p(t) = 1 + tanh(.) (starts at 1)."""
    def __init__(self, ch=(64, 128, 128), time_dim=128, passes=(1, 2)):
        super().__init__()
        self.ch, self.passes, self.time_dim = ch, passes, time_dim
        cd = time_dim * 4
        self.time_mlp = tf.keras.Sequential([layers.Dense(cd, activation="swish"), layers.Dense(cd)])
        self.pass_emb = [self.add_weight(shape=(cd,), initializer="zeros", name=f"pass{p}") for p in range(len(passes))]
        self.gates = [layers.Dense(1, kernel_initializer="zeros") for _ in passes]
        self.stem = layers.Conv2D(ch[0], 3, padding="same")
        self.enc = [AdaRes(ch[l], ch[l]) for l in range(3)]
        self.dec = [AdaRes(ch[l], ch[l]) for l in range(2)]
        self.down = [HaarDown(ch[l + 1]) for l in range(2)]
        self.up = [HaarUp(ch[l]) for l in range(2)]
        self.merge = [layers.Conv2D(ch[l], 1) for l in range(2)]
        self.attn = ab.PAttentionBlock(ch[2])
        self.out_norm = layers.GroupNormalization(8)
        self.out_conv = layers.Conv2D(3, 3, padding="same")

    def call(self, inputs, training=False):
        x, t = inputs
        te = self.time_mlp(ab.timestep_embedding(t, self.time_dim))
        h0 = self.stem(x)
        for p, depth in enumerate(self.passes):
            c = te + self.pass_emb[p]
            h, skips = h0, []
            for l in range(depth):
                h = self.enc[l](h, c); skips.append(h); h = self.down[l](h)
            h = self.enc[depth](h, c)
            if depth == 2:
                h = self.attn(h)
            for l in reversed(range(depth)):
                h = self.up[l](h)
                h = self.merge[l](tf.concat([h, skips[l]], -1))
                h = self.dec[l](h, c)
            g = 1.0 + tf.tanh(self.gates[p](te))[:, :, None, None]
            h0 = h0 + g * (h - h0)
        return self.out_conv(tf.nn.silu(self.out_norm(h0)))


_ab_build = ab.build


SWL_VARIANTS = {"swl": {}, "swl_long": {}, "swl_wide": dict(ch=(80, 160, 160)), "swl_p3": dict(passes=(1, 2, 2))}


def build(name):
    if name in SWL_VARIANTS:
        tf.keras.utils.set_random_seed(0)
        net = SWL(**SWL_VARIANTS[name])
        fwd = lambda x, t: net([x, t])
        fwd(tf.zeros((2, 32, 32, 3)), tf.constant([0, 999], tf.int32))
        return net, fwd
    return _ab_build(name)


ab.build = build          # so ab.make_state / evaluate pick up SWL


# ---------------------------------------------------------------- train (round-robin, identical stream)
def train_multi(names, kind, steps, ckpt_every=500):
    A = alpha_map(kind)
    x_all = ab.load_data()
    runs = []
    for name in names:
        d = ab.run_dir(kind, name); os.makedirs(d, exist_ok=True)
        net, fwd, opt, ema, ema_n, step, ckpt = ab.make_state(name)
        mgr = tf.train.CheckpointManager(ckpt, os.path.join(d, "ckpt"), max_to_keep=1)
        if mgr.latest_checkpoint:
            ckpt.restore(mgr.latest_checkpoint).expect_partial()

        def make(net=net, fwd=fwd, opt=opt, ema=ema, ema_n=ema_n):
            @tf.function(jit_compile=True)
            def f(x0, t, noise):
                xt = noisy(kind, A, x0, t, noise)
                with tf.GradientTape() as tape:
                    mse = tf.reduce_mean(tf.square(fwd(xt, t) - noise))
                g = tape.gradient(mse, net.trainable_variables)
                opt.apply_gradients(zip(g, net.trainable_variables))
                ema_n.assign_add(1.0)
                dd = tf.minimum(0.999, (1.0 + ema_n) / (10.0 + ema_n))
                for e, v in zip(ema, net.trainable_variables):
                    e.assign(dd * e + (1 - dd) * v)
                return mse
            return f
        runs.append(dict(name=name, d=d, net=net, step=step, mgr=mgr, fn=make(), losses=[], secs=0.0,
                         log=open(os.path.join(d, "loss.csv"), "a")))
        print(f"[{kind}/{name}] params {net.count_params():,}", flush=True)
    s = min(int(r["step"].numpy()) for r in runs)
    while s < steps:
        x, t, n = ab.stream_batch(x_all, s)
        xt, tt, nt = tf.constant(x), tf.constant(t), tf.constant(n)
        for r in runs:
            if int(r["step"].numpy()) > s:
                continue
            t0 = time.time(); r["losses"].append(float(r["fn"](xt, tt, nt))); r["secs"] += time.time() - t0
            r["step"].assign(s + 1)
        s += 1
        if s % ckpt_every == 0 or s == steps:
            for r in runs:
                r["mgr"].save(); m = float(np.mean(r["losses"][-ckpt_every:]))
                r["log"].write(f"{s},{m:.6f}\n"); r["log"].flush()
                print(f"[{kind}/{r['name']}] step {s} loss {m:.4f} {r['secs'] / max(1, len(r['losses'])) * 1000:.0f} ms/step", flush=True)
    for r in runs:
        json.dump(dict(params=int(r["net"].count_params()), steps=int(r["step"].numpy()),
                       ms_per_step=r["secs"] / max(1, len(r["losses"])) * 1000), open(os.path.join(r["d"], "train_done.json"), "w"))


# ---------------------------------------------------------------- eval (DDIM generalised to per-coefficient alpha)
def generate_spectral(kind, predict):
    A = alpha_map(kind)
    rng = np.random.default_rng(ab.EVAL_SEED)
    idx = ab.SAMPLE_TS[::-1]
    out = []
    for start in range(0, ab.N_EVAL, ab.EVAL_BATCH):
        n = min(ab.EVAL_BATCH, ab.N_EVAL - start)
        x_t = tf.constant(rng.standard_normal((n, 32, 32, 3)), tf.float32)
        for i, t in enumerate(idx):
            tp = idx[i + 1] if i + 1 < len(idx) else -1
            a_t = A[int(t)][None]; a_p = A[int(tp)][None] if tp >= 0 else tf.ones_like(a_t)
            eps_w = haar2(predict(x_t, int(t)))
            xw = haar2(x_t)
            x0 = tf.clip_by_value(ihaar2((xw - tf.sqrt(1 - a_t) * eps_w) / tf.sqrt(a_t)), -1, 1)
            x_t = ihaar2(tf.sqrt(a_p) * haar2(x0) + tf.sqrt(1 - a_p) * eps_w)
        out.append(np.clip((x_t.numpy() + 1) / 2, 0, 1))
    return np.concatenate(out)


_orig_generate = ab.generate
ab.generate = lambda kind, predict: _orig_generate(kind, predict) if kind in ("gauss", "blur") else generate_spectral(kind, predict)


if __name__ == "__main__":
    cmd, kind, models = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    if cmd == "train":
        train_multi(models, kind, int(sys.argv[4]))
    else:
        for m in models:
            ab.evaluate(m, kind); gc.collect()
