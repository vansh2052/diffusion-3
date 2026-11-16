"""Blur-diffusion prototypes (small models, runs on a laptop CPU).

Every process is a linear Gaussian diffusion that is diagonal in the 2-D DCT basis:

    x_t = IDCT( a[t,k] * DCT(x0) + s[t,k] * DCT(eps) ),       eps ~ N(0, I) in pixel space

  * isotropic Gaussian diffusion: a = sqrt(abar_t), s = sqrt(1 - abar_t) at every frequency k
  * blur of std sigma (reflective boundary) multiplies frequency k by exp(-lam_k sigma^2 / 2),
    lam_k = (pi kx/N)^2 + (pi ky/N)^2, so blur + noise is a[t,k] = sqrt(abar_t) * d[t,k]
  * stochastic heat equation (blur and noise injected together): same a, coloured s[t,k]
The reverse step is then exact (DDIM per frequency), and the network can predict
  eps, x0, v (per-frequency v) or u (per-frequency EDM preconditioning with the data's power spectrum P_k).

  python blur_proto.py run <process> [steps] [seed]     train + evaluate one prototype
  python blur_proto.py report
"""
import gc, glob, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
os.environ.setdefault("BENCH_ROOT", RUNS)
os.environ.setdefault("BENCH_DATA", os.path.join(ROOT, "cifar10_airplanes.npz"))
os.environ.setdefault("BENCH_N_EVAL", "2000")
os.environ.setdefault("BENCH_EVAL_STEPS", "50")
_argv, sys.argv = sys.argv, sys.argv[:1]
import arch_bench as ab            # noqa: E402  (original UNet, timestep encoding, pixel blur, Inception FID/IS)
sys.argv = _argv
import numpy as np                 # noqa: E402
import tensorflow as tf            # noqa: E402

N, T = 32, ab.timesteps
BATCH, LR, EMA = 64, 4e-4, 0.998
BASE = 16                                                     # tiny original UNet: 526k params
BETA = np.clip(ab.beta_schedule.astype(np.float64), 0, 0.999)
ABAR = np.cumprod(1.0 - BETA)


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


# radial frequency bands for the spectral uncertainty weighting: edges in |k| = sqrt(kx^2 + ky^2)
_r = np.sqrt(np.arange(N)[:, None] ** 2 + np.arange(N)[None, :] ** 2)
BAND_EDGES = [0, 1, 2, 4, 8, 16, 32, 99]
BAND_ID = np.digitize(_r, BAND_EDGES[1:-1])
NB = len(BAND_EDGES) - 1
BANDMAT = tf.constant(np.eye(NB)[BAND_ID].astype(np.float32))                      # (32, 32, NB) one-hot
BAND_FRAC = tf.constant((np.bincount(BAND_ID.ravel(), minlength=NB) / N ** 2).astype(np.float32))


def idct2(X):
    return tf.einsum("kh,bklc,lw->bhwc", CMAT, X, CMAT)


def load_data():
    return ab.load_data()


_PK = []


def power_spectrum():
    """Mean DCT power of the training images per frequency (averaged over images and channels)."""
    if not _PK:
        X = dct2(tf.constant(load_data()[:2000])).numpy()
        _PK.append((X ** 2).mean(axis=(0, 3)).astype(np.float64))
    return _PK[0]


# ---------------------------------------------------------------- processes
def heat_time(cfg):
    kind, smax = cfg.get("heat", "lin"), cfg.get("smax", 0.0)
    if kind == "lin":                                         # old pipeline: sigma_b = smax (1 - sqrt(abar))
        sb = smax * (1.0 - np.sqrt(ABAR))
    elif kind == "sin":                                       # Hoogeboom & Salimans shape
        sb = smax * np.sin(np.pi * (np.arange(T) + 1) / T / 2)
    elif kind == "snr":                                       # blur width ~ noise/signal amplitude ratio
        sb = np.minimum(cfg["c"] * np.sqrt((1 - ABAR) / ABAR), smax)
    else:
        raise ValueError(kind)
    return sb ** 2 / 2


def build_process(cfg):
    """float64 loga[T,N,N], s[T,N,N]."""
    fam = cfg["family"]
    hla = 0.5 * np.log(ABAR)[:, None, None]
    ones = np.ones((T, N, N))
    s_iso = np.sqrt(1 - ABAR)[:, None, None] * ones
    if fam == "gauss":
        return hla * ones, s_iso
    if fam == "wiener":                                       # attenuate each frequency by its Wiener gain^gamma
        snr = (ABAR / (1 - ABAR))[:, None, None] * power_spectrum()[None]
        return hla + cfg["gamma"] * (np.log(snr) - np.log1p(snr)), s_iso
    tau = heat_time(cfg)[:, None, None]
    if fam == "blur":
        logd = -LAM[None] * tau
        if cfg.get("dmin", 0) > 0:
            logd = np.log((1 - cfg["dmin"]) * np.exp(logd) + cfg["dmin"])
        return hla + logd, s_iso
    if fam == "she":                                          # dx = (Laplacian - beta/2) x dt + sqrt(beta) dW
        dtau = np.diff(np.concatenate([np.zeros((1, 1, 1)), tau], 0), axis=0)
        loga, s2 = np.zeros((T, N, N)), np.zeros((T, N, N))
        la, v = np.zeros((N, N)), np.zeros((N, N))
        for t in range(T):
            logm = 0.5 * np.log(1 - BETA[t]) - LAM * dtau[t]
            la, v = la + logm, np.exp(2 * logm) * v + BETA[t]
            loga[t], s2[t] = la, v
        return loga, np.sqrt(s2)
    raise ValueError(fam)


PROCS = {
    "gauss_eps":  dict(family="gauss", target="eps"),
    "gauss_x0":   dict(family="gauss", target="x0"),
    "gauss_v":    dict(family="gauss", target="v"),
    "gauss_u":    dict(family="gauss", target="u"),
    "old_x0":     dict(family="old", target="x0"),               # previous pipeline (pixel conv blur, zero padding)
    "lin2.5_x0":  dict(family="blur", heat="lin", smax=2.5, target="x0"),
    "lin2.5_eps": dict(family="blur", heat="lin", smax=2.5, target="eps"),
    "lin2.5_v":   dict(family="blur", heat="lin", smax=2.5, target="v"),
    "lin2.5_u":   dict(family="blur", heat="lin", smax=2.5, target="u"),
    "lin1.2_u":   dict(family="blur", heat="lin", smax=1.2, target="u"),
    "lin5_u":     dict(family="blur", heat="lin", smax=5.0, target="u"),
    "sin5_u":     dict(family="blur", heat="sin", smax=5.0, dmin=1e-3, target="u"),
    "sin20_eps":  dict(family="blur", heat="sin", smax=20.0, dmin=1e-3, target="eps"),
    "snr1_u":     dict(family="blur", heat="snr", c=1.0, smax=8.0, target="u"),
    "she2.5_u":   dict(family="she", heat="lin", smax=2.5, target="u"),
    "she2.5_v":   dict(family="she", heat="lin", smax=2.5, target="v"),
    # data-adaptive "Wiener blur": d_k(t) = (SNR_k / (1 + SNR_k))^gamma, SNR_k from the data power spectrum
    "wien1_u":    dict(family="wiener", gamma=1.0, target="u"),
    "wien0.5_u":  dict(family="wiener", gamma=0.5, target="u"),
    # spectral uncertainty weighting (learned per-band, per-noise-level loss weights)
    "lin2.5_u_suw": dict(family="blur", heat="lin", smax=2.5, target="u", suw=True),
    "she2.5_u_suw": dict(family="she", heat="lin", smax=2.5, target="u", suw=True),
    "gauss_eps_suw": dict(family="gauss", target="eps", suw=True),
}


class Process:
    def __init__(self, name):
        self.name, self.cfg = name, PROCS[name]
        self.target, self.family = self.cfg["target"], self.cfg["family"]
        if self.family == "old":
            return
        self.loga, self.s = build_process(self.cfg)
        self.a = np.exp(self.loga)
        self.A = tf.constant(self.a.astype(np.float32))
        self.S = tf.constant(self.s.astype(np.float32))
        self.sqP = tf.constant(np.sqrt(power_spectrum())[None, :, :, None].astype(np.float32))

    def forward(self, x0, t, noise):
        if self.family == "old":
            return ab.degrade_blur_noise(x0, t, noise), x0
        X0, E = dct2(x0), dct2(noise)
        a, s = tf.gather(self.A, t)[..., None], tf.gather(self.S, t)[..., None]
        Xt = a * X0 + s * E
        if self.target == "eps":
            tgt = noise
        elif self.target == "x0":
            tgt = x0
        elif self.target == "v":
            tgt = idct2((a * E - s * X0) / tf.sqrt(a * a + s * s))
        else:                                                 # u: v with each frequency scaled by its data std
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
        x0n = (A / m) * (X / m) - (s / m) * O
        return x0n * self.sqP, (s / m) * (X / m) + (A / m) * O


# ---------------------------------------------------------------- model / data stream
def build_model(seed=0):
    tf.keras.utils.set_random_seed(seed)
    net = ab.UNet(input_shape=(32, 32, 3), time_dim=ab.embedding_dim, base=BASE)
    fwd = lambda x, t: net([x, tf.gather(ab.PE, t)])
    fwd(tf.zeros((1, 32, 32, 3)), tf.zeros((1,), tf.int32))
    return net, fwd


def stream_batch(x_all, step, seed=0):
    """Identical for every prototype: images, flips, t (stratified uniform) and noise depend only on (seed, step)."""
    spe = len(x_all) // BATCH
    epoch, i = divmod(step, spe)
    perm = np.random.default_rng([seed, 0, epoch]).permutation(len(x_all))
    x = x_all[perm[i * BATCH:(i + 1) * BATCH]].copy()
    rng = np.random.default_rng([seed, 1, step])
    flip = rng.random(BATCH) < 0.5
    x[flip] = x[flip][:, :, ::-1, :]
    t = np.floor((np.arange(BATCH) + rng.random(BATCH)) / BATCH * T).astype("int32")
    noise = rng.standard_normal(x.shape, dtype=np.float32)
    return x, t, noise


def run_dir(name, seed):
    return os.path.join(RUNS, f"{name}" + (f"_s{seed}" if seed else ""))


def train(name, steps, seed=0, ckpt_every=500):
    d = run_dir(name, seed)
    os.makedirs(d, exist_ok=True)
    P = Process(name)
    net, fwd = build_model(seed)
    suw = bool(P.cfg.get("suw"))
    wnet = tf.keras.Sequential([tf.keras.layers.Dense(64, activation="silu"),
                                tf.keras.layers.Dense(NB, kernel_initializer="zeros")])
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
    def f(x0, t, noise):
        xt, tgt = P.forward(x0, t, noise)
        with tf.GradientTape() as tape:
            err = fwd(xt, t) - tgt
            mse = tf.reduce_mean(tf.square(err))
            if suw:                                           # sum_b frac_b * (mse_b * exp(-u_b(t)) + u_b(t))
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
    for s in range(s0, steps):
        x, t, n = stream_batch(x_all, s, seed)
        losses.append(float(f(tf.constant(x), tf.constant(t), tf.constant(n))))
        step.assign(s + 1)
        if (s + 1) % ckpt_every == 0 or s + 1 == steps:
            mgr.save()
            m = float(np.mean(losses[-ckpt_every:]))
            log.write(f"{s + 1},{m:.6f}\n"); log.flush()
            print(f"[{name} s{seed}] step {s + 1} loss {m:.4f} ({(time.time() - t0) / len(losses) * 1000:.0f} ms/step)",
                  flush=True)
    if steps > s0:
        json.dump(dict(steps=steps, seconds=time.time() - t0, params=int(net.count_params())),
                  open(os.path.join(d, "train_done.json"), "w"))
    if suw:                                                   # learned relative weight exp(-u) per band and t
        tg = np.arange(0, T, 50)
        w = np.exp(-wnet(tf.gather(ab.PE, tf.constant(tg))).numpy())
        json.dump(dict(t=tg.tolist(), band_edges=BAND_EDGES, weight=w.round(4).tolist()),
                  open(os.path.join(d, "suw_weights.json"), "w"))
    for v, e in zip(net.trainable_variables, ema):
        v.assign(e)
    return net, fwd


# ---------------------------------------------------------------- sampling
SAMPLE_TS = ab.SAMPLE_TS


def dct_tables(P, eta):
    idx = SAMPLE_TS[::-1]
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


def generate(P, predict, sampler="eta0", n_total=None, seed=None):
    n_total = n_total or ab.N_EVAL
    rng = np.random.default_rng(ab.EVAL_SEED if seed is None else seed)
    idx = SAMPLE_TS[::-1]
    out = []
    if P.family != "old":
        eta = float(sampler[3:])
        tables = dct_tables(P, eta)
        s_prior = tf.constant(P.s[T - 1][None, ..., None].astype(np.float32))
    for start in range(0, n_total, ab.EVAL_BATCH):
        n = min(ab.EVAL_BATCH, n_total - start)
        z = tf.constant(rng.standard_normal((n, N, N, 3)), tf.float32)
        if P.family == "old":                                 # pixel blur, x0-prediction
            x = z
            for i, t in enumerate(idx):
                tp = idx[i + 1] if i + 1 < len(idx) else -1
                x0 = tf.clip_by_value(predict(x, int(t)), -1.0, 1.0)
                if tp < 0:
                    x = x0
                elif sampler == "cold":                       # Bansal et al.: reuse the initial noise z
                    x = x - ab.degrade_blur_noise(x0, tf.fill([n], int(t)), z) + ab.degrade_blur_noise(x0, tf.fill([n], int(tp)), z)
                else:                                         # "ddim": re-estimate the noise from the current state
                    at, ap = float(ABAR[t]), float(ABAR[tp])
                    eps = (x - np.sqrt(at) * ab.apply_blur(x0, tf.fill([n], int(t)))) / np.sqrt(1 - at)
                    x = np.sqrt(ap) * ab.apply_blur(x0, tf.fill([n], int(tp))) + np.sqrt(1 - ap) * eps
            out.append(np.clip((x.numpy() + 1) / 2, 0, 1))
            continue
        X = s_prior * dct2(z)
        for t, a, s, nxt in tables:
            X0, E = P.decode(predict(idct2(X), int(t)), X, a, s)
            X0 = dct2(tf.clip_by_value(idct2(X0), -1.0, 1.0))
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


_SC = []


def score(images):
    if not _SC:
        sc = ab.InceptionScorer()
        _SC.append((sc,) + tuple(ab.reference(sc)))
    sc, mu, sig, _ = _SC[0]
    f, pr = sc(images)
    m, sd = ab.inception_score(pr)
    return ab.fid_from_features(f, mu, sig), m, sd


def evaluate(name, fwd, seed=0, samplers=None):
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
        imgs = generate(P, predict, smp)
        fid, m, sd = score(imgs)
        res[smp] = dict(fid=fid, is_mean=m, is_std=sd, n=len(imgs), seconds=time.time() - t0)
        np.save(os.path.join(d, f"samples_{smp}.npy"), (imgs[:64] * 255).astype(np.uint8))
        json.dump(res, open(path, "w"), indent=2)
        print(f"[{name} s{seed}] {smp}: FID {fid:.2f}  IS {m:.2f}  ({time.time() - t0:.0f}s)", flush=True)


def report():
    rows = []
    for p in glob.glob(os.path.join(RUNS, "*", "eval.json")):
        run = os.path.basename(os.path.dirname(p))
        lp = os.path.join(os.path.dirname(p), "loss.csv")
        last = np.loadtxt(lp, delimiter=",", ndmin=2)[-1] if os.path.exists(lp) and os.path.getsize(lp) else [0, np.nan]
        for smp, r in json.load(open(p)).items():
            rows.append((run, smp, int(last[0]), last[1], r))
    print(f"{'run':18s} {'sampler':7s} {'steps':>6s} {'loss':>8s} {'FID':>8s} {'IS':>6s}")
    for run, smp, st, l, r in sorted(rows, key=lambda x: x[4]["fid"]):
        print(f"{run:18s} {smp:7s} {st:>6d} {l:8.4f} {r['fid']:8.2f} {r['is_mean']:6.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        name = sys.argv[2]
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        samplers = sys.argv[5].split(",") if len(sys.argv) > 5 else None
        net, fwd = train(name, steps, seed)
        evaluate(name, fwd, seed, samplers)
    else:
        report()
