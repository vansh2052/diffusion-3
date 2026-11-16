"""Blur-diffusion micro-prototypes: 16x16 airplanes, tiny U-Net, light on a laptop CPU.

Same framework as blur_proto.py (32x32, for the DGX), scaled down:
every process is a linear Gaussian diffusion diagonal in the 2-D DCT basis,

    x_t = IDCT( a[t,k] * DCT(x0) + s[t,k] * DCT(eps) ),   eps ~ N(0, I)

with an exact per-frequency DDIM sampler, and the network predicting eps / x0 / v / u
(u = per-frequency EDM preconditioning with the data power spectrum P_k).
Score: FID on Inception-V3 pool features at 75x75 ("FID75"), 1,000 samples, same seed for every model.

  python blur_micro.py run <process> [steps] [seed]
  python blur_micro.py report
"""
import gc, glob, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "micro_runs")
os.environ.setdefault("BENCH_ROOT", RUNS)
os.environ.setdefault("BENCH_DATA", os.path.join(ROOT, "cifar10_airplanes.npz"))
_argv, sys.argv = sys.argv, sys.argv[:1]
import arch_bench as ab            # noqa: E402  (original UNet + sinusoidal timestep encoding + cosine schedule)
sys.argv = _argv
import numpy as np                 # noqa: E402
import tensorflow as tf            # noqa: E402
from tensorflow.keras import layers  # noqa: E402

N, T = 16, ab.timesteps
BATCH, LR, EMA = 64, 4e-4, 0.998
BASE = int(os.environ.get("MICRO_BASE", 8))
N_EVAL, EVAL_STEPS, EVAL_BATCH, EVAL_SEED = 1000, 40, 250, 1234
SAMPLE_TS = np.linspace(0, T - 1, EVAL_STEPS).round().astype(int)
BETA = np.clip(ab.beta_schedule.astype(np.float64), 0, 0.999)
ABAR = np.cumprod(1.0 - BETA)


# ---------------------------------------------------------------- data (2x2 average-pooled airplanes, [-1, 1])
_DATA = {}


def load_data(split="x_train"):
    if split not in _DATA:
        x = np.load(os.environ["BENCH_DATA"])[split].astype(np.float32) / 127.5 - 1.0
        _DATA[split] = x.reshape(-1, N, 2, N, 2, 3).mean(axis=(2, 4))
    return _DATA[split]


# ---------------------------------------------------------------- DCT
def dct_matrix(n=N):
    k, i = np.arange(n)[:, None], np.arange(n)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0] /= np.sqrt(2.0)
    return c


CMAT = tf.constant(dct_matrix().astype(np.float32))
_w = np.pi * np.arange(N) / N
LAM = _w[:, None] ** 2 + _w[None, :] ** 2


def dct2(x):
    return tf.einsum("kh,bhwc,lw->bklc", CMAT, x, CMAT)


def idct2(X):
    return tf.einsum("kh,bklc,lw->bhwc", CMAT, X, CMAT)


_r = np.sqrt(np.arange(N)[:, None] ** 2 + np.arange(N)[None, :] ** 2)
BAND_EDGES = [0, 1, 2, 4, 8, 16, 99]
BAND_ID = np.digitize(_r, BAND_EDGES[1:-1])
NB = len(BAND_EDGES) - 1
BANDMAT = tf.constant(np.eye(NB)[BAND_ID].astype(np.float32))
BAND_FRAC = tf.constant((np.bincount(BAND_ID.ravel(), minlength=NB) / N ** 2).astype(np.float32))

_PK = []


def power_spectrum():
    if not _PK:
        X = dct2(tf.constant(load_data()[:2000])).numpy()
        _PK.append((X ** 2).mean(axis=(0, 3)).astype(np.float64))
    return _PK[0]


# ---------------------------------------------------------------- processes (blur widths in 16-px units)
def heat_time(cfg):
    kind, smax = cfg.get("heat", "lin"), cfg.get("smax", 0.0)
    if kind == "lin":
        sb = smax * (1.0 - np.sqrt(ABAR))
    elif kind == "sin":
        sb = smax * np.sin(np.pi * (np.arange(T) + 1) / T / 2)
    else:
        raise ValueError(kind)
    return sb ** 2 / 2


def build_process(cfg):
    fam = cfg["family"]
    hla = 0.5 * np.log(ABAR)[:, None, None]
    ones = np.ones((T, N, N))
    s_iso = np.sqrt(1 - ABAR)[:, None, None] * ones
    if fam == "gauss":
        return hla * ones, s_iso
    if fam == "wiener":
        snr = (ABAR / (1 - ABAR))[:, None, None] * power_spectrum()[None]
        return hla + cfg["gamma"] * (np.log(snr) - np.log1p(snr)), s_iso
    tau = heat_time(cfg)[:, None, None]
    if fam == "blur":
        logd = -LAM[None] * tau
        if cfg.get("dmin", 0) > 0:
            logd = np.log((1 - cfg["dmin"]) * np.exp(logd) + cfg["dmin"])
        return hla + logd, s_iso
    if fam == "she":
        dtau = np.diff(np.concatenate([np.zeros((1, 1, 1)), tau], 0), axis=0)
        loga, s2 = np.zeros((T, N, N)), np.zeros((T, N, N))
        la, v = np.zeros((N, N)), np.zeros((N, N))
        for t in range(T):
            logm = 0.5 * np.log(1 - BETA[t]) - LAM * dtau[t]
            la, v = la + logm, np.exp(2 * logm) * v + BETA[t]
            loga[t], s2[t] = la, v
        return loga, np.sqrt(s2)
    raise ValueError(fam)


# "old" = the previous pipeline: zero-padded pixel blur sigma_b = 1.25 (1 - sqrt(abar)) (= 2.5 at 32 px), x0-prediction
PROCS = {
    "gauss_eps":     dict(family="gauss", target="eps"),
    "gauss_u":       dict(family="gauss", target="u"),
    "old_x0":        dict(family="old", target="x0"),
    "lin1.25_x0":    dict(family="blur", heat="lin", smax=1.25, target="x0"),
    "lin1.25_v":     dict(family="blur", heat="lin", smax=1.25, target="v"),
    "lin1.25_u":     dict(family="blur", heat="lin", smax=1.25, target="u"),
    "lin2.5_u":      dict(family="blur", heat="lin", smax=2.5, target="u"),
    "lin0.6_u":      dict(family="blur", heat="lin", smax=0.6, target="u"),
    "sin2.5_u":      dict(family="blur", heat="sin", smax=2.5, dmin=1e-3, target="u"),
    "she1.25_u":     dict(family="she", heat="lin", smax=1.25, target="u"),
    "wien1_u":       dict(family="wiener", gamma=1.0, target="u"),
    "wien0.5_u":     dict(family="wiener", gamma=0.5, target="u"),
    "lin1.25_u_suw": dict(family="blur", heat="lin", smax=1.25, target="u", suw=True),
    # x0 target + learned per-band uncertainty weights (= per-band log-MSE objective)
    "lin1.25_x0_suw": dict(family="blur", heat="lin", smax=1.25, target="x0", suw=True),
    "lin1.25_x0_sc":  dict(family="blur", heat="lin", smax=1.25, target="x0", sc=True),
    "lin1.25_x0_suw_sc": dict(family="blur", heat="lin", smax=1.25, target="x0", suw=True, sc=True),
    "lin2.5_x0_suw":  dict(family="blur", heat="lin", smax=2.5, target="x0", suw=True),
    "wien0.5_x0":     dict(family="wiener", gamma=0.5, target="x0"),
    "wien0.5_x0_suw": dict(family="wiener", gamma=0.5, target="x0", suw=True),
    "she1.25_x0":     dict(family="she", heat="lin", smax=1.25, target="x0"),
    "she1.25_x0_suw": dict(family="she", heat="lin", smax=1.25, target="x0", suw=True),
    "gauss_x0":       dict(family="gauss", target="x0"),
    "gauss_x0_suw":   dict(family="gauss", target="x0", suw=True),
    # x0 target + additive spectral boost of fine frequencies
    "lin1.25_x0_hf0.01": dict(family="blur", heat="lin", smax=1.25, target="x0", hf=0.01),
    "lin1.25_x0_hf0.03": dict(family="blur", heat="lin", smax=1.25, target="x0", hf=0.03),
    "gauss_x0_hf0.01":   dict(family="gauss", target="x0", hf=0.01),
    "lin2.5_x0_hf0.01":  dict(family="blur", heat="lin", smax=2.5, target="x0", hf=0.01),
    "she1.25_x0_hf0.01": dict(family="she", heat="lin", smax=1.25, target="x0", hf=0.01),
    # partial spectral preconditioning: P_k^gamma (gamma = 1 -> u, gamma = 0 -> v)
    "lin1.25_u0.5":  dict(family="blur", heat="lin", smax=1.25, target="u0.5"),
    "lin1.25_u0.25": dict(family="blur", heat="lin", smax=1.25, target="u0.25"),
    "gauss_u0.5":    dict(family="gauss", target="u0.5"),
    "wien0.5_u0.5":  dict(family="wiener", gamma=0.5, target="u0.5"),
    "she1.25_u0.5":  dict(family="she", heat="lin", smax=1.25, target="u0.5"),
    "wien0.5_u_suw": dict(family="wiener", gamma=0.5, target="u", suw=True),
    "gauss_eps_suw": dict(family="gauss", target="eps", suw=True),
}
PROCS.update({k.replace("_u", "_u_sc"): dict(v, sc=True) for k, v in list(PROCS.items()) if v.get("target") == "u"})


# zero-padded pixel blur for the "old" process (per-sample Gaussian kernel, 9x9)
_K = 9


def _kernels():
    ax = np.arange(-(_K // 2), _K // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    ks = []
    for sb in 1.25 * (1.0 - np.sqrt(ABAR)):
        k = np.exp(-(xx ** 2 + yy ** 2) / (2 * max(sb, 1e-3) ** 2))
        ks.append(k / k.sum())
    return tf.constant(np.stack(ks).astype(np.float32))


OLD_K = None


def pixel_blur(x, t):
    b, c = tf.shape(x)[0], x.shape[3]
    k = tf.repeat(tf.gather(OLD_K, t)[..., None], c, axis=-1)            # B, K, K, C
    k = tf.reshape(tf.transpose(k, [1, 2, 0, 3]), [_K, _K, b * c, 1])
    xr = tf.reshape(tf.transpose(x, [1, 2, 0, 3]), [1, N, N, b * c])
    y = tf.nn.depthwise_conv2d(xr, k, [1, 1, 1, 1], "SAME")
    return tf.transpose(tf.reshape(y, [N, N, b, c]), [2, 0, 1, 3])


class Process:
    def __init__(self, name):
        self.name, self.cfg = name, PROCS[name]
        self.target, self.family = self.cfg["target"], self.cfg["family"]
        if self.family == "old":
            global OLD_K
            if OLD_K is None:
                OLD_K = _kernels()                             # build eagerly, outside any tf.function
            return
        self.loga, self.s = build_process(self.cfg)
        self.a = np.exp(self.loga)
        self.A = tf.constant(self.a.astype(np.float32))
        self.S = tf.constant(self.s.astype(np.float32))
        gamma = float(self.target[1:]) if self.target.startswith("u") and len(self.target) > 1 else 1.0
        self.sqP = tf.constant((power_spectrum() ** (gamma / 2))[None, :, :, None].astype(np.float32))   # sqrt(P_k^gamma)

    def forward(self, x0, t, noise):
        if self.family == "old":
            ab_t = tf.gather(tf.constant(ABAR.astype(np.float32)), t)[:, None, None, None]
            return tf.sqrt(ab_t) * pixel_blur(x0, t) + tf.sqrt(1 - ab_t) * noise, x0
        X0, E = dct2(x0), dct2(noise)
        a, s = tf.gather(self.A, t)[..., None], tf.gather(self.S, t)[..., None]
        Xt = a * X0 + s * E
        if self.target == "eps":
            tgt = noise
        elif self.target == "x0":
            tgt = x0
        elif self.target == "v":
            tgt = idct2((a * E - s * X0) / tf.sqrt(a * a + s * s))
        else:
            A = a * self.sqP
            tgt = idct2((A * E - s * X0 / self.sqP) / tf.sqrt(A * A + s * s))
        return idct2(Xt), tgt

    def decode(self, out, X, a, s):
        O = dct2(out)
        if self.target == "eps":
            return (X - s * O) * a / (a * a + 1e-6), O
        if self.target == "x0":
            return O, (X - a * O) / s
        if self.target == "v":
            n = tf.sqrt(a * a + s * s)
            return (a / n) * (X / n) - (s / n) * O, (s / n) * (X / n) + (a / n) * O
        A = a * self.sqP
        m = tf.sqrt(A * A + s * s)
        return ((A / m) * (X / m) - (s / m) * O) * self.sqP, (s / m) * (X / m) + (A / m) * O


# ---------------------------------------------------------------- model, stream, training
def build_model(seed=0, in_ch=3):
    tf.keras.utils.set_random_seed(seed)
    net = ab.UNet(input_shape=(N, N, in_ch), time_dim=ab.embedding_dim, base=BASE)
    fwd = lambda x, t: net([x, tf.gather(ab.PE, t)])
    fwd(tf.zeros((1, N, N, in_ch)), tf.zeros((1,), tf.int32))
    return net, fwd


def stream_batch(x_all, step, seed=0):
    spe = len(x_all) // BATCH
    epoch, i = divmod(step, spe)
    perm = np.random.default_rng([seed, 0, epoch]).permutation(len(x_all))
    x = x_all[perm[i * BATCH:(i + 1) * BATCH]].copy()
    rng = np.random.default_rng([seed, 1, step])
    flip = rng.random(BATCH) < 0.5
    x[flip] = x[flip][:, :, ::-1, :]
    t = np.floor((np.arange(BATCH) + rng.random(BATCH)) / BATCH * T).astype("int32")
    noise = rng.standard_normal(x.shape, dtype=np.float32)
    sc_coin = (rng.random(BATCH) < 0.5).astype(np.float32)                 # self-conditioning: which samples get it
    return x, t, noise, sc_coin


def run_dir(name, seed):
    return os.path.join(RUNS, name + (f"_s{seed}" if seed else ""))


def train(name, steps, seed=0, ckpt_every=500):
    d = run_dir(name, seed)
    os.makedirs(d, exist_ok=True)
    P = Process(name)
    sc, suw, hf = bool(P.cfg.get("sc")), bool(P.cfg.get("suw")), float(P.cfg.get("hf", 0.0))
    if hf:                                                     # additive spectral boost: w_k = 1 + hf / P_k (DCT domain)
        HFW = tf.constant((1.0 + hf / power_spectrum())[None, :, :, None].astype(np.float32))
    net, fwd = build_model(seed, in_ch=6 if sc else 3)
    wnet = tf.keras.Sequential([layers.Dense(64, activation="silu"), layers.Dense(NB, kernel_initializer="zeros")])
    wnet(tf.zeros((1, ab.embedding_dim)))
    train_vars = net.trainable_variables + (wnet.trainable_variables if suw else [])
    opt = tf.keras.optimizers.Adam(LR)
    opt.build(train_vars)
    ema = [tf.Variable(v, trainable=False) for v in net.trainable_variables]
    ema_n = tf.Variable(0.0, trainable=False)
    step = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(model=net, optimizer=opt, ema=ema, ema_n=ema_n, step=step, wnet=wnet)
    mgr = tf.train.CheckpointManager(ckpt, os.path.join(d, "ckpt"), max_to_keep=1)
    if mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint).expect_partial()

    @tf.function(jit_compile=True)
    def f(x0, t, noise, coin):
        xt, tgt = P.forward(x0, t, noise)
        if sc:                                                 # self-conditioning (Chen et al. 2022) on the x0 estimate
            X = dct2(xt)
            a, s = tf.gather(P.A, t)[..., None], tf.gather(P.S, t)[..., None]
            first = fwd(tf.concat([xt, tf.zeros_like(xt)], -1), t)
            x0_sc = tf.stop_gradient(tf.clip_by_value(idct2(P.decode(first, X, a, s)[0]), -1, 1))
            inp = tf.concat([xt, x0_sc * coin[:, None, None, None]], -1)
        else:
            inp = xt
        with tf.GradientTape() as tape:
            err = fwd(inp, t) - tgt
            mse = tf.reduce_mean(tf.square(err))
            if hf:
                total = tf.reduce_mean(HFW * tf.square(dct2(err)))
            elif suw:
                band = tf.einsum("bklc,kln->bn", tf.square(dct2(err)), BANDMAT) / (BAND_FRAC * N * N * 3)
                u = wnet(tf.gather(ab.PE, t))
                total = tf.reduce_mean(tf.reduce_sum(BAND_FRAC * (band * tf.exp(-u) + u), axis=1))
            else:
                total = mse
        g = tape.gradient(total, train_vars)
        opt.apply_gradients(zip(g, train_vars))
        ema_n.assign_add(1.0)
        dd = tf.minimum(EMA, (1.0 + ema_n) / (10.0 + ema_n))
        for e, v in zip(ema, net.trainable_variables):
            e.assign(dd * e + (1 - dd) * v)
        return mse

    x_all = load_data()
    log = open(os.path.join(d, "loss.csv"), "a")
    s0, losses, t0 = int(step.numpy()), [], time.time()
    for s_ in range(s0, steps):
        x, t, n, coin = stream_batch(x_all, s_, seed)
        losses.append(float(f(tf.constant(x), tf.constant(t), tf.constant(n), tf.constant(coin))))
        step.assign(s_ + 1)
        if (s_ + 1) % ckpt_every == 0 or s_ + 1 == steps:
            mgr.save()
            m = float(np.mean(losses[-ckpt_every:]))
            log.write(f"{s_ + 1},{m:.6f}\n"); log.flush()
            print(f"[{name} s{seed}] step {s_ + 1} loss {m:.4f} ({(time.time() - t0) / len(losses) * 1000:.0f} ms/step)",
                  flush=True)
    if steps > s0:
        json.dump(dict(steps=steps, seconds=time.time() - t0, params=int(net.count_params())),
                  open(os.path.join(d, "train_done.json"), "w"))
    if suw:
        tg = np.arange(0, T, 50)
        w = np.exp(-wnet(tf.gather(ab.PE, tf.constant(tg))).numpy())
        json.dump(dict(t=tg.tolist(), band_edges=BAND_EDGES, weight=w.round(4).tolist()),
                  open(os.path.join(d, "suw_weights.json"), "w"))
    for v, e in zip(net.trainable_variables, ema):
        v.assign(e)
    return net, fwd, sc


# ---------------------------------------------------------------- sampling
def sample_ts(power=1.0):
    u = np.linspace(0, 1, EVAL_STEPS) ** power
    return np.unique((u * (T - 1)).round().astype(int))


def parse_sampler(name):
    eta, _, q = name[3:].partition("q")
    return float(eta), (float(q) if q else 1.0)


def dct_tables(P, eta, power=1.0):
    idx = sample_ts(power)[::-1]
    f = lambda z: tf.constant(z[None, ..., None].astype(np.float32))
    rows = []
    for i, t in enumerate(idx):
        tp = idx[i + 1] if i + 1 < len(idx) else -1
        a, s = P.a[t], P.s[t]
        if tp < 0:
            rows.append((t, f(a), f(s), None)); continue
        ap, sp = P.a[tp], P.s[tp]
        r = np.exp(P.loga[t] - P.loga[tp])
        q2 = np.maximum(s ** 2 - (r * sp) ** 2, 0.0)
        sig2 = eta ** 2 * sp ** 2 * q2 / s ** 2
        rows.append((t, f(a), f(s), (f(ap), f(np.sqrt(np.maximum(sp ** 2 - sig2, 0))), f(np.sqrt(sig2)))))
    return rows


def generate(P, predict, sampler="eta0", sc=False, n_total=N_EVAL, seed=EVAL_SEED):
    rng = np.random.default_rng(seed)
    idx = SAMPLE_TS[::-1]
    out = []
    for start in range(0, n_total, EVAL_BATCH):
        n = min(EVAL_BATCH, n_total - start)
        z = tf.constant(rng.standard_normal((n, N, N, 3)), tf.float32)
        if P.family == "old":
            x = z
            for i, t in enumerate(idx):
                tp = idx[i + 1] if i + 1 < len(idx) else -1
                x0 = tf.clip_by_value(predict(x, int(t)), -1.0, 1.0)
                if tp < 0:
                    x = x0
                    continue
                at, ap = float(ABAR[t]), float(ABAR[tp])
                bt, bp = pixel_blur(x0, tf.fill([n], int(t))), pixel_blur(x0, tf.fill([n], int(tp)))
                if sampler == "cold":                          # Bansal et al.: reuse the initial noise
                    x = x - (np.sqrt(at) * bt + np.sqrt(1 - at) * z) + (np.sqrt(ap) * bp + np.sqrt(1 - ap) * z)
                else:                                          # re-estimate the noise from the current state
                    x = np.sqrt(ap) * bp + np.sqrt(1 - ap) * (x - np.sqrt(at) * bt) / np.sqrt(1 - at)
            out.append(np.clip((x.numpy() + 1) / 2, 0, 1))
            continue
        eta, power = parse_sampler(sampler)
        X = tf.constant(P.s[T - 1][None, ..., None].astype(np.float32)) * dct2(z)
        x0_prev = tf.zeros((n, N, N, 3))
        for t, a, s, nxt in dct_tables(P, eta, power):
            xin = idct2(X)
            if sc:
                xin = tf.concat([xin, x0_prev], -1)
            X0, E = P.decode(predict(xin, int(t)), X, a, s)
            x0 = tf.clip_by_value(idct2(X0), -1.0, 1.0)
            x0_prev, X0 = x0, dct2(x0)
            E = (X - a * X0) / s
            if nxt is None:
                X = X0
            else:
                ap, sdet, sig = nxt
                X = ap * X0 + sdet * E
                if eta > 0:
                    X = X + sig * dct2(tf.constant(rng.standard_normal((n, N, N, 3)), tf.float32))
        out.append(np.clip((idct2(X).numpy() + 1) / 2, 0, 1))
    return np.concatenate(out)


# ---------------------------------------------------------------- FID75 (Inception-V3 pool features at 75x75)
_FEAT = []


def feature_model():
    if not _FEAT:
        full = tf.keras.applications.InceptionV3(include_top=True, weights="imagenet")
        small = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=(75, 75, 3), pooling="avg")
        src = [l for l in full.layers if l.weights and l.name != "predictions"]
        dst = [l for l in small.layers if l.weights]
        assert len(src) == len(dst)
        for a, b in zip(src, dst):
            b.set_weights(a.get_weights())
        del full
        gc.collect()
        f = tf.function(lambda x: small(tf.image.resize(x, (75, 75)) * 2.0 - 1.0, training=False))
        _FEAT.append(f)
    return _FEAT[0]


def features(images01):
    f = feature_model()
    return np.concatenate([f(tf.constant(images01[i:i + 250], tf.float32)).numpy()
                           for i in range(0, len(images01), 250)]).astype(np.float64)


def _reference():
    ref = os.path.join(RUNS, "fid75_reference.npz")
    if not os.path.exists(ref):
        os.makedirs(RUNS, exist_ok=True)
        fr = features(np.clip((load_data() + 1) / 2, 0, 1))
        np.savez(ref, mu=fr.mean(0), sigma=np.cov(fr, rowvar=False), sub=fr[::2].astype(np.float32))
    return np.load(ref)


def kid(fx, fy):
    """Unbiased KID (Binkowski et al. 2018), cubic polynomial kernel; returns MMD^2 x 1000."""
    d = fx.shape[1]
    k = lambda a, b: (a @ b.T / d + 1.0) ** 3
    kxx, kyy, kxy = k(fx, fx), k(fy, fy), k(fx, fy)
    m, n = len(fx), len(fy)
    return 1000 * ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                   - 2 * kxy.mean())


def metrics(images01):
    r = _reference()
    f = features(images01)
    return ab.fid_from_features(f, r["mu"], r["sigma"]), kid(f, r["sub"].astype(np.float64))


def evaluate(name, fwd, sc, seed=0, samplers=None):
    P = Process(name)
    samplers = samplers or (["cold", "ddim"] if P.family == "old" else ["eta0"])
    d = run_dir(name, seed)
    path = os.path.join(d, "eval.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    pf = tf.function(lambda x, t: fwd(x, t), jit_compile=True)
    predict = lambda x, tv: pf(x, tf.fill([tf.shape(x)[0]], tv))
    for smp in samplers:
        if smp in res:
            continue
        t0 = time.time()
        imgs = generate(P, predict, smp, sc)
        fd, kd = metrics(imgs)
        res[smp] = dict(fid75=fd, kid75=kd, n=len(imgs), seconds=time.time() - t0)
        np.save(os.path.join(d, f"samples_{smp}.npy"), (imgs[:64] * 255).astype(np.uint8))
        json.dump(res, open(path, "w"), indent=2)
        print(f"[{name} s{seed}] {smp}: KID75x1000 {kd:.2f}  FID75 {fd:.2f}  ({time.time() - t0:.0f}s)", flush=True)


def report():
    rows = []
    for p in glob.glob(os.path.join(RUNS, "*", "eval.json")):
        run = os.path.basename(os.path.dirname(p))
        lp = os.path.join(os.path.dirname(p), "loss.csv")
        last = np.loadtxt(lp, delimiter=",", ndmin=2)[-1] if os.path.exists(lp) and os.path.getsize(lp) else [0, np.nan]
        for smp, r in json.load(open(p)).items():
            rows.append((run, smp, int(last[0]), last[1], r["kid75"], r["fid75"]))
    print(f"{'run':22s} {'sampler':7s} {'steps':>6s} {'loss':>8s} {'KIDx1000':>9s} {'FID75':>8s}")
    for row in sorted(rows, key=lambda x: x[4]):
        print(f"{row[0]:22s} {row[1]:7s} {row[2]:>6d} {row[3]:8.4f} {row[4]:9.2f} {row[5]:8.2f}")


def reeval(name, samplers, seed=0):
    P = Process(name)
    net, fwd = build_model(seed, in_ch=6 if P.cfg.get("sc") else 3)
    ema = [tf.Variable(v, trainable=False) for v in net.trainable_variables]
    ckpt = tf.train.Checkpoint(model=net, ema=ema)
    ckpt.restore(tf.train.latest_checkpoint(os.path.join(run_dir(name, seed), "ckpt"))).expect_partial()
    for v, e in zip(net.trainable_variables, ema):
        v.assign(e)
    evaluate(name, fwd, bool(P.cfg.get("sc")), seed, samplers)


if __name__ == "__main__":
    if sys.argv[1] == "reeval":
        reeval(sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif sys.argv[1] == "run":
        name = sys.argv[2]
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        samplers = sys.argv[5].split(",") if len(sys.argv) > 5 else None
        net, fwd, sc = train(name, steps, seed)
        evaluate(name, fwd, sc, seed, samplers)
    else:
        report()
