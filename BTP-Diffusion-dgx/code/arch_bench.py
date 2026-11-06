"""Architecture benchmark for small diffusion models on CIFAR-10 airplanes.

  python arch_bench.py train --model NAME --kind gauss|blur --steps N
  python arch_bench.py eval  --model NAME --kind gauss|blur
  python arch_bench.py report

Every model of a kind sees the IDENTICAL training stream (images, flips, t, noise
depend only on the step number). t is stratified so each batch has a fixed number of
samples per noise range (needed to train the routed U-Net with static shapes).
gauss = epsilon-prediction + DDIM; blur = blur+noise degradation, x0-prediction + cold sampler.
"""
import argparse, gc, glob, json, os, sys, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/arch_bench"))
DATA = os.environ.get("BENCH_DATA", os.path.expanduser("~/diffusion-3/cifar10_airplanes.npz"))
timesteps = 1000
embedding_dim = 128
BATCH = 128
ROUTE_COUNTS = [(0, 100, 13), (100, 250, 19), (250, 400, 19), (400, 1000, 77)]   # sums to 128
assert sum(c for *_, c in ROUTE_COUNTS) == BATCH


# ---- original notebook code (verbatim)
def cosine_schedule(timesteps, s=0.008):
    """Nichol & Dhariwal cosine schedule.

    The linear schedule destroys most of the signal in the first ~200 steps
    at 32x32, so a lot of the model's capacity is spent on timesteps that
    carry no information. The cosine schedule spreads the damage evenly.
    """
    steps = np.arange(timesteps + 1)
    f = np.cos(((steps / timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return np.clip(betas, 0.0001, 0.999)

def timestep_positional_encoding(timesteps, embedding_dim):
    pe = np.zeros((timesteps, embedding_dim))
    position = np.arange(0, timesteps, 1)
    power = np.arange(0, embedding_dim, 2) / embedding_dim
    pe[:, 0::2] = np.sin(position[:, np.newaxis] / (10000 ** power))
    pe[:, 1::2] = np.cos(position[:, np.newaxis] / (10000 ** power))
    return pe

class ResBlock(layers.Layer):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = layers.Conv2D(
            out_channels, 3, padding="same"
        )
        # GroupNorm, not BatchNorm: every image in the batch carries a
        # different noise level, so batch statistics are meaningless here.
        self.norm1 = layers.GroupNormalization(groups=8)

        self.conv2 = layers.Conv2D(
            out_channels, 3, padding="same"
        )
        self.norm2 = layers.GroupNormalization(groups=8)

        self.time_proj = layers.Dense(out_channels)

        self.activation = layers.Activation("swish")

        if in_channels != out_channels:
            self.skip = layers.Conv2D(
                out_channels, 1, padding="same"
            )
        else:
            self.skip = None

    def call(self, x, t_emb):

        if self.skip is not None:
            residual = self.skip(x)
        else:
            residual = x

        h = self.conv1(x)
        h = self.norm1(h)

        time = self.time_proj(t_emb)
        time = time[:, tf.newaxis, tf.newaxis, :]

        h = h + time
        h = self.activation(h)

        h = self.conv2(h)
        h = self.norm2(h)

        h = h + residual
        h = self.activation(h)

        return h

class AttentionBlock(layers.Layer):
    """Self-attention over the 8x8 feature map.

    Convolutions only see a local window; one attention layer at the
    bottleneck lets every pixel look at every other pixel, which is what
    keeps a generated object globally coherent instead of locally plausible.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.norm = layers.GroupNormalization(groups=8)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels // num_heads
        )

    def call(self, x):

        b = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]

        y = self.norm(x)
        y = tf.reshape(y, (b, h * w, c))
        y = self.attention(y, y)
        y = tf.reshape(y, (b, h, w, c))

        return x + y

class DownBlock(layers.Layer):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.resblock = ResBlock(
            in_channels,
            out_channels,
            time_dim
        )

        self.downsample = layers.Conv2D(
            out_channels,
            3,
            strides=2,
            padding="same"
        )

    def call(self, x, t_emb):

        x = self.resblock(x, t_emb)

        skip = x

        x = self.downsample(x)

        return x, skip

class UpBlock(layers.Layer):

    def __init__(self, in_channels, skip_channels, out_channels, time_dim):
        super().__init__()

        self.upsample = layers.Conv2DTranspose(
            out_channels,
            3,
            strides=2,
            padding="same"
        )

        self.resblock = ResBlock(
            out_channels + skip_channels,
            out_channels,
            time_dim
        )

    def call(self, x, skip, t_emb):

        x = self.upsample(x)

        x = tf.concat([x, skip], axis=-1)

        x = self.resblock(x, t_emb)

        return x

class UNet(tf.keras.Model):

    def __init__(self, input_shape, time_dim=128, base=32):
        super().__init__()

        self.time_dim = time_dim

        c1, c2, c3 = base, base * 2, base * 4

        # Let the network reshape the sinusoidal encoding itself
        self.time_mlp = tf.keras.Sequential([
            layers.Dense(time_dim * 2, activation="swish"),
            layers.Dense(time_dim * 2),
        ])

        # Input
        self.input_conv = layers.Conv2D(
            c1,
            3,
            padding="same"
        )

        # Downsampling
        self.down1 = DownBlock(
            c1, c2, time_dim
        )

        self.down2 = DownBlock(
            c2, c3, time_dim
        )

        # Bottleneck
        self.bottleneck1 = ResBlock(
            c3, c3, time_dim
        )

        self.attention = AttentionBlock(c3)

        self.bottleneck2 = ResBlock(
            c3, c3, time_dim
        )

        # Upsampling
        self.up2 = UpBlock(
            c3, c3, c2, time_dim
        )

        self.up1 = UpBlock(
            c2, c2, c1, time_dim
        )

        # Output
        self.output_conv = layers.Conv2D(
            3,
            3,
            padding="same"
        )

    def call(self, inputs):

        x, t_emb = inputs

        t_emb = self.time_mlp(t_emb)

        # Input
        x = self.input_conv(x)

        # Down
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)

        # Bottleneck
        x = self.bottleneck1(x, t_emb)
        x = self.attention(x)
        x = self.bottleneck2(x, t_emb)

        # Up
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        # Output
        x = self.output_conv(x)

        return x


# ---------------------------------------------------------------- schedule / data
beta_schedule = cosine_schedule(timesteps)
ALPHA_BAR_NP = np.cumprod(1.0 - beta_schedule).astype(np.float32)
ALPHA_BAR = tf.constant(ALPHA_BAR_NP)
PE = tf.constant(timestep_positional_encoding(timesteps, embedding_dim), dtype=tf.float32)


def load_data():
    d = np.load(DATA)
    x = d["x_train"].astype(np.float32) / 127.5 - 1.0
    return x


def stream_batch(x_all, step, seed=0):
    spe = len(x_all) // BATCH
    epoch, i = divmod(step, spe)
    perm = np.random.default_rng([seed, 0, epoch]).permutation(len(x_all))
    x = x_all[perm[i * BATCH:(i + 1) * BATCH]].copy()
    rng = np.random.default_rng([seed, 1, step])
    flip = rng.random(BATCH) < 0.5
    x[flip] = x[flip][:, :, ::-1, :]
    t = np.concatenate([rng.integers(lo, hi, c) for lo, hi, c in ROUTE_COUNTS]).astype("int32")
    noise = rng.standard_normal(x.shape, dtype=np.float32)
    return x, t, noise



# ---- blur + noise degradation (verbatim)
SIGMA_MAX = 2.5
KERNEL_SIZE = 17


def gaussian_kernel_2d(sigma, size):
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * max(sigma, 1e-3) ** 2))
    kernel /= kernel.sum()
    return kernel


# Blur sigma ramps up on the same schedule as the noise (alpha_bar), so both
# forms of corruption reach full strength together at the last timestep.
blur_sigma = SIGMA_MAX * (1.0 - np.sqrt(ALPHA_BAR_NP))

BLUR_KERNELS = tf.constant(
    np.stack([gaussian_kernel_2d(s, KERNEL_SIZE) for s in blur_sigma]),
    dtype=tf.float32
)  # shape: (timesteps, KERNEL_SIZE, KERNEL_SIZE)



def apply_blur(x, t_indices):
    """Blur every image in the batch by its own per-example kernel (indexed
    by t), in a single depthwise convolution.

    The trick: fold the batch dimension into the channel dimension so each
    example gets its own filter, since tf.nn.depthwise_conv2d applies one
    filter per input channel.
    """
    b = tf.shape(x)[0]
    h, w, c = x.shape[1], x.shape[2], x.shape[3]

    kernels = tf.gather(BLUR_KERNELS, t_indices)                    # [B, K, K]
    kernels = tf.repeat(kernels[:, :, :, tf.newaxis], c, axis=-1)   # [B, K, K, C]
    kernels = tf.transpose(kernels, [1, 2, 0, 3])                   # [K, K, B, C]
    kernels = tf.reshape(kernels, [KERNEL_SIZE, KERNEL_SIZE, b * c, 1])

    x_r = tf.transpose(x, [1, 2, 0, 3])                             # [H, W, B, C]
    x_r = tf.reshape(x_r, [1, h, w, b * c])

    blurred = tf.nn.depthwise_conv2d(
        x_r, kernels, strides=[1, 1, 1, 1], padding="SAME"
    )                                                                # [1, H, W, B*C]

    blurred = tf.reshape(blurred, [h, w, b, c])
    blurred = tf.transpose(blurred, [2, 0, 1, 3])                   # [B, H, W, C]
    return blurred


def degrade_blur_noise(x0, t_indices, noise):
    """Combined corruption: blur x0 by the timestep's sigma, then blend with
    noise using the same alpha_bar weighting as the Gaussian pipeline."""
    alpha_bar_t = tf.gather(ALPHA_BAR, t_indices)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    blurred = apply_blur(x0, t_indices)
    return tf.sqrt(alpha_bar_t) * blurred + tf.sqrt(1.0 - alpha_bar_t) * noise

# ---------------------------------------------------------------- path U-Net (v1 = Colab bricks, v2 = fixed bricks)
def timestep_embedding(t, dim):
    t = tf.cast(tf.reshape(t, (-1, 1)), tf.float32)
    half = dim // 2
    freqs = tf.exp(-np.log(10000.0) * tf.range(half, dtype=tf.float32) / half)
    args = t * freqs[tf.newaxis, :]
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


class PResBlock(layers.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = layers.GroupNormalization(groups=8)
        self.conv1 = layers.Conv2D(out_ch, 3, padding="same")
        self.time_proj = layers.Dense(out_ch)
        self.norm2 = layers.GroupNormalization(groups=8)
        self.conv2 = layers.Conv2D(out_ch, 3, padding="same")
        self.skip = layers.Conv2D(out_ch, 1) if in_ch != out_ch else None

    def call(self, x, t_emb):
        h = self.conv1(tf.nn.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, tf.newaxis, tf.newaxis, :]
        h = self.conv2(tf.nn.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class PAttentionBlock(layers.Layer):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = layers.GroupNormalization(groups=8)
        self.attention = layers.MultiHeadAttention(num_heads, channels // num_heads)

    def call(self, x):
        b = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        y = tf.reshape(self.norm(x), (b, h * w, c))
        y = self.attention(y, y)
        return x + tf.reshape(y, (b, h, w, c))


class Resample(layers.Layer):
    """v1: down k levels = conv kernel=stride=2**k (non-overlapping).
       v2: down = 3x3 stride-2 conv for one level, avg-pool + 3x3 conv for bigger jumps."""
    def __init__(self, from_level, to_level, out_ch, v2):
        super().__init__()
        k = to_level - from_level
        if k > 0 and not v2:
            self.op = layers.Conv2D(out_ch, 2 ** k, strides=2 ** k)
        elif k == 1:
            self.op = layers.Conv2D(out_ch, 3, strides=2, padding="same")
        elif k > 1:
            self.op = tf.keras.Sequential([layers.AveragePooling2D(2 ** k), layers.Conv2D(out_ch, 3, padding="same")])
        else:
            self.op = tf.keras.Sequential([layers.UpSampling2D(2 ** (-k), interpolation="nearest"),
                                           layers.Conv2D(out_ch, 3, padding="same")])

    def call(self, x):
        return self.op(x)


class PathUNet(tf.keras.Model):
    def __init__(self, path, base=32, mults=(1, 2, 2, 4), time_dim=128, attn_levels=(3,), v2=False):
        super().__init__()
        self.levels = list(path)
        self.time_dim = time_dim
        ch = [base * m for m in mults]
        self.time_mlp = tf.keras.Sequential([layers.Dense(time_dim * 4, activation="swish"), layers.Dense(time_dim * 4)])
        self.input_conv = layers.Conv2D(ch[0], 3, padding="same")
        self.moves, self.blocks, self.attns = [], [], []
        seen = {0: ch[0]}
        cur = 0
        self.blocks.append(PResBlock(ch[0], ch[0]))
        for nxt in self.levels[1:]:
            self.moves.append(Resample(cur, nxt, ch[nxt], v2))
            self.blocks.append(PResBlock(ch[nxt] + seen.get(nxt, 0), ch[nxt]))
            self.attns.append(PAttentionBlock(ch[nxt]) if nxt in attn_levels else None)
            seen[nxt] = ch[nxt]
            cur = nxt
        self.out_norm = layers.GroupNormalization(groups=8)
        self.output_conv = layers.Conv2D(3, 3, padding="same")

    def call(self, inputs, training=False):
        x, t = inputs
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x = self.blocks[0](self.input_conv(x), t_emb)
        memory = {0: x}
        for nxt, move, block, attn in zip(self.levels[1:], self.moves, self.blocks[1:], self.attns):
            x = move(x)
            if nxt in memory:
                x = tf.concat([x, memory[nxt]], axis=-1)
            x = block(x, t_emb)
            if attn is not None:
                x = attn(x)
            memory[nxt] = x
        return self.output_conv(tf.nn.silu(self.out_norm(x)))


# ---------------------------------------------------------------- noise-routed U-Net
ROUTES = [(0, 100, {0: "ab", 1: "ab", 2: "", 3: ""}),
          (100, 250, {0: "", 1: "a", 2: "a", 3: "a"}),
          (250, 400, {0: "", 1: "", 2: "ab", 3: "a"}),
          (400, 1000, {0: "", 1: "", 2: "a", 3: "ab"})]
DENSE_ROUTE = {0: "ab", 1: "ab", 2: "ab", 3: "ab"}
ROUTED_BLOCKS = [f"{s}{l}{k}" for s in "ed" for l in range(4) for k in "ab"]
LEVEL_COST = {0: 1.0, 1: 1.0, 2: 0.33, 3: 0.25}
BLOCK_COST = np.array([2 * LEVEL_COST[int(b[1])] for b in ROUTED_BLOCKS], np.float32)
GATE_PENALTY = 0.01


def route_cost(route):
    return sum(2 * LEVEL_COST[l] * len(route[l]) for l in range(4)) / BLOCK_COST.sum()


class RoutedUNet(tf.keras.Model):
    def __init__(self, mode, base=32, mults=(1, 2, 2, 4), time_dim=128):
        super().__init__()
        self.mode = mode
        self.time_dim = time_dim
        ch = [base * m for m in mults]
        self.time_mlp = tf.keras.Sequential([layers.Dense(time_dim * 4, activation="swish"), layers.Dense(time_dim * 4)])
        self.stem = layers.Conv2D(ch[0], 3, padding="same")
        self.downs = [layers.Conv2D(ch[l], 3, strides=2, padding="same") for l in (1, 2, 3)]
        self.ups = [tf.keras.Sequential([layers.UpSampling2D(2, interpolation="nearest"),
                                         layers.Conv2D(ch[l - 1], 3, padding="same")]) for l in (1, 2, 3)]
        self.fuse = [layers.Conv2D(ch[l], 1) for l in range(4)]
        self.blocks = {n: PResBlock(ch[int(n[1])], ch[int(n[1])]) for n in ROUTED_BLOCKS}
        self.attns = {n: PAttentionBlock(ch[2]) for n in ROUTED_BLOCKS if n[1] == "2"}
        self.mid1 = PResBlock(ch[3], ch[3])
        self.mid_attn = PAttentionBlock(ch[3])
        self.mid2 = PResBlock(ch[3], ch[3])
        if mode == "learned":
            self.gate_head = layers.Dense(len(ROUTED_BLOCKS), kernel_initializer="zeros",
                                          bias_initializer=tf.keras.initializers.Constant(2.0))
        self.out_norm = layers.GroupNormalization(groups=8)
        self.out_conv = layers.Conv2D(3, 3, padding="same")

    def gate_values(self, t):
        return tf.sigmoid(self.gate_head(self.time_mlp(timestep_embedding(t, self.time_dim))))

    def forward(self, x, t, route=None, active=None, return_gates=False):
        route = route or DENSE_ROUTE
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        gates = tf.sigmoid(self.gate_head(t_emb)) if self.mode == "learned" else None

        def run(name, h):
            if name[2] not in route[int(name[1])] or (active is not None and name not in active):
                return h
            y = self.blocks[name](h, t_emb)
            if name in self.attns:
                y = self.attns[name](y)
            if gates is not None:
                g = gates[:, ROUTED_BLOCKS.index(name)][:, None, None, None]
                y = h + g * (y - h)
            return y

        h = self.stem(x)
        skips = []
        for level in range(4):
            if level > 0:
                h = self.downs[level - 1](h)
            for k in "ab":
                h = run(f"e{level}{k}", h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)
        for level in (3, 2, 1, 0):
            h = self.fuse[level](tf.concat([h, skips[level]], axis=-1))
            for k in "ab":
                h = run(f"d{level}{k}", h)
            if level > 0:
                h = self.ups[level - 1](h)
        out = self.out_conv(tf.nn.silu(self.out_norm(h)))
        return (out, gates) if return_gates else out

    def call(self, inputs, training=False):
        x, t = inputs
        return self.forward(x, t)


# ---------------------------------------------------------------- model registry
V2 = dict(base=32, mults=(2, 4, 4), attn_levels=(2,), v2=True)      # 64/128/128 ch, stop at 8x8, attention at 8x8
MODELS = {
    "original_unet":  ("orig", None),
    "unet_v1":        ("path", dict(path=[0, 1, 2, 3, 2, 1, 0], base=32)),
    "ladder_v1":      ("path", dict(path=[0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0], base=32)),
    "unet_v2":        ("path", dict(path=[0, 1, 2, 1, 0], **V2)),
    "ladder_v2":      ("path", dict(path=[0, 1, 0, 1, 2, 1, 0], **V2)),
    "ladder_rev_v2":  ("path", dict(path=[0, 1, 2, 1, 0, 1, 0], **V2)),
    "wnet_v2":        ("path", dict(path=[0, 1, 2, 1, 0, 1, 2, 1, 0], **V2)),
    "routed_dense":   ("routed", "dense"),
    "routed_fixed":   ("routed", "fixed"),
    "routed_learned": ("routed", "learned"),
}


def build(name):
    tf.keras.utils.set_random_seed(0)
    kind, cfg = MODELS[name]
    if kind == "orig":
        net = UNet(input_shape=(32, 32, 3), time_dim=embedding_dim, base=32)
        fwd = lambda x, t: net([x, tf.gather(PE, t)])
    elif kind == "path":
        net = PathUNet(**cfg)
        fwd = lambda x, t: net([x, t])
    else:
        net = RoutedUNet(cfg)
        fwd = lambda x, t: net.forward(x, t)
        net([tf.zeros((2, 32, 32, 3)), tf.constant([0, 999], tf.int32)])
    fwd(tf.zeros((2, 32, 32, 3)), tf.constant([0, 999], tf.int32))
    return net, fwd


def model_loss(name, net, fwd, x_in, t, target):
    """Returns (total, mse). routed_fixed sends each fixed-size route slice through its route."""
    if name == "routed_fixed":
        sq, i = 0.0, 0
        for (lo, hi, c), (_, _, route) in zip(ROUTE_COUNTS, ROUTES):
            pred = net.forward(x_in[i:i + c], t[i:i + c], route=route)
            sq += tf.reduce_sum(tf.square(pred - target[i:i + c]))
            i += c
        mse = sq / tf.cast(tf.size(target), tf.float32)
        return mse, mse
    if name == "routed_learned":
        pred, gates = net.forward(x_in, t, return_gates=True)
        mse = tf.reduce_mean(tf.square(pred - target))
        used = tf.reduce_mean(tf.reduce_sum(gates * BLOCK_COST, axis=1)) / BLOCK_COST.sum()
        return mse + GATE_PENALTY * used, mse
    mse = tf.reduce_mean(tf.square(fwd(x_in, t) - target))
    return mse, mse


def run_dir(kind, name):
    return os.path.join(ROOT, kind, name)


def make_state(name):
    net, fwd = build(name)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
    opt.build(net.trainable_variables)
    ema = [tf.Variable(v, trainable=False) for v in net.trainable_variables]
    ema_n = tf.Variable(0.0, trainable=False)
    step = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(model=net, optimizer=opt, ema_weights=ema, ema_step=ema_n, step=step)
    return net, fwd, opt, ema, ema_n, step, ckpt


# ---------------------------------------------------------------- train
def train(name, kind, steps, ckpt_every=500):
    d = run_dir(kind, name)
    os.makedirs(d, exist_ok=True)
    net, fwd, opt, ema, ema_n, step, ckpt = make_state(name)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(d, "ckpt"), max_to_keep=1)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
    start = int(step.numpy())
    x_all = load_data()
    print(f"[{kind}/{name}] params {net.count_params():,}  steps {start} -> {steps}", flush=True)

    @tf.function(jit_compile=(name != "routed_fixed"), reduce_retracing=True)
    def step_fn(x_0, t, noise):
        a = tf.gather(ALPHA_BAR, t)[:, None, None, None]
        if kind == "gauss":
            x_t, target = tf.sqrt(a) * x_0 + tf.sqrt(1.0 - a) * noise, noise
        else:
            x_t, target = degrade_blur_noise(x_0, t, noise), x_0
        with tf.GradientTape() as tape:
            total, mse = model_loss(name, net, fwd, x_t, t, target)
        grads = tape.gradient(total, net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        ema_n.assign_add(1.0)
        dd = tf.minimum(0.999, (1.0 + ema_n) / (10.0 + ema_n))
        for e, v in zip(ema, net.trainable_variables):
            e.assign(dd * e + (1.0 - dd) * v)
        return mse

    log = open(os.path.join(d, "loss.csv"), "a")
    losses, t0 = [], time.time()
    target_file = os.path.join(ROOT, "target_steps")
    s = start
    while True:
        if s % 50 == 0 and os.path.exists(target_file):
            try:
                steps = int(open(target_file).read().strip())
            except ValueError:
                pass
        if s >= steps:
            break
        x, t, n = stream_batch(x_all, s)
        losses.append(float(step_fn(x, t, n)))
        step.assign(s + 1)
        if (s + 1) % ckpt_every == 0:
            manager.save()
            m = float(np.mean(losses[-ckpt_every:]))
            log.write(f"{s + 1},{m:.6f}\n"); log.flush()
            print(f"[{kind}/{name}] step {s + 1} loss {m:.4f}  {(time.time() - t0) / len(losses) * 1000:.0f} ms/step", flush=True)
        s += 1
    manager.save()
    json.dump(dict(params=int(net.count_params()), steps=int(step.numpy()),
                   ms_per_step=(time.time() - t0) / max(1, len(losses)) * 1000), open(os.path.join(d, "train_done.json"), "w"))


def train_multi(names, kind, steps, ckpt_every=500):
    """Train several models in ONE process, one step each in turn: identical step counts by construction."""
    x_all = load_data()
    runs = []
    for name in names:
        d = run_dir(kind, name)
        os.makedirs(d, exist_ok=True)
        net, fwd, opt, ema, ema_n, step, ckpt = make_state(name)
        manager = tf.train.CheckpointManager(ckpt, os.path.join(d, "ckpt"), max_to_keep=1)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint).expect_partial()

        def make(name=name, net=net, fwd=fwd, opt=opt, ema=ema, ema_n=ema_n):
            @tf.function(jit_compile=(name != "routed_fixed"), reduce_retracing=True)
            def step_fn(x_0, t, noise):
                a = tf.gather(ALPHA_BAR, t)[:, None, None, None]
                if kind == "gauss":
                    x_t, target = tf.sqrt(a) * x_0 + tf.sqrt(1.0 - a) * noise, noise
                else:
                    x_t, target = degrade_blur_noise(x_0, t, noise), x_0
                with tf.GradientTape() as tape:
                    total, mse = model_loss(name, net, fwd, x_t, t, target)
                grads = tape.gradient(total, net.trainable_variables)
                opt.apply_gradients(zip(grads, net.trainable_variables))
                ema_n.assign_add(1.0)
                dd = tf.minimum(0.999, (1.0 + ema_n) / (10.0 + ema_n))
                for e, v in zip(ema, net.trainable_variables):
                    e.assign(dd * e + (1.0 - dd) * v)
                return mse
            return step_fn
        runs.append(dict(name=name, d=d, net=net, step=step, manager=manager, fn=make(), losses=[], secs=0.0,
                         log=open(os.path.join(d, "loss.csv"), "a")))
        print(f"[{kind}/{name}] params {net.count_params():,}  at step {int(step.numpy())}", flush=True)
    s = min(int(r["step"].numpy()) for r in runs)
    target_file = os.path.join(ROOT, "target_steps")
    while True:
        if s % 50 == 0 and os.path.exists(target_file):
            try:
                steps = int(open(target_file).read().strip())
            except ValueError:
                pass
        if s >= steps:
            break
        x, t, n = stream_batch(x_all, s)
        xt, tt, nt = tf.constant(x), tf.constant(t), tf.constant(n)
        for r in runs:
            if int(r["step"].numpy()) > s:
                continue
            t0 = time.time()
            r["losses"].append(float(r["fn"](xt, tt, nt)))
            r["secs"] += time.time() - t0
            r["step"].assign(s + 1)
        s += 1
        if s % ckpt_every == 0:
            for r in runs:
                r["manager"].save()
                m = float(np.mean(r["losses"][-ckpt_every:]))
                r["log"].write(f"{s},{m:.6f}\n"); r["log"].flush()
                print(f"[{kind}/{r['name']}] step {s} loss {m:.4f}  {r['secs'] / max(1, len(r['losses'])) * 1000:.0f} ms/step", flush=True)
    for r in runs:
        r["manager"].save()
        json.dump(dict(params=int(r["net"].count_params()), steps=int(r["step"].numpy()),
                       ms_per_step=r["secs"] / max(1, len(r["losses"])) * 1000), open(os.path.join(r["d"], "train_done.json"), "w"))


# ---------------------------------------------------------------- eval
N_EVAL = int(os.environ.get("BENCH_N_EVAL", 5000)); EVAL_STEPS = int(os.environ.get("BENCH_EVAL_STEPS", 100))
EVAL_BATCH, EVAL_SEED = min(500, N_EVAL), 1234
SAMPLE_TS = np.linspace(0, timesteps - 1, EVAL_STEPS).round().astype(int)


class InceptionScorer:
    def __init__(self):
        base = tf.keras.applications.InceptionV3(include_top=True, weights="imagenet")
        self.model = tf.keras.Model(base.input, [base.get_layer("avg_pool").output, base.output])
        self.f = tf.function(lambda x: self.model(tf.image.resize(x, (299, 299)) * 2.0 - 1.0, training=False))

    def __call__(self, images_01, batch=100):
        feats, probs = [], []
        for i in range(0, len(images_01), batch):
            f, p = self.f(tf.convert_to_tensor(images_01[i:i + batch], tf.float32))
            feats.append(f.numpy()); probs.append(p.numpy())
        return np.concatenate(feats).astype(np.float64), np.concatenate(probs).astype(np.float64)


def fid_from_features(feats, mu_ref, sigma_ref):
    import scipy.linalg
    mu, sigma = feats.mean(axis=0), np.cov(feats, rowvar=False)
    diff = mu - mu_ref
    covmean = scipy.linalg.sqrtm(sigma.dot(sigma_ref))
    if not np.isfinite(covmean).all():
        off = np.eye(sigma.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma + off).dot(sigma_ref + off))
    return float(diff.dot(diff) + np.trace(sigma) + np.trace(sigma_ref) - 2.0 * np.trace(np.real(covmean)))


def inception_score(probs, splits=10):
    s = []
    for part in np.array_split(probs, splits):
        py = part.mean(axis=0, keepdims=True)
        s.append(np.exp((part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(axis=1).mean()))
    return float(np.mean(s)), float(np.std(s))


def reference(scorer):
    p = os.path.join(ROOT, "fid_reference_airplanes.npz")
    if not os.path.exists(p):
        f, pr = scorer(np.clip((load_data() + 1) / 2, 0, 1))
        m, is_ = inception_score(pr)
        np.savez(p, mu=f.mean(0), sigma=np.cov(f, rowvar=False), is_mean=m, is_std=is_)
    r = np.load(p)
    return r["mu"], r["sigma"], (float(r["is_mean"]), float(r["is_std"]))


def predictors(name, net, fwd):
    """Yields (label, predict_fn(x, t_int_scalar) , block-compute fraction or None)."""
    if name == "routed_fixed":
        per = [tf.function(lambda x, t, r=r: net.forward(x, t, route=r)) for *_, r in ROUTES]
        def p(x, tv):
            i = next(i for i, (lo, hi, _) in enumerate(ROUTES) if lo <= tv < hi or i == len(ROUTES) - 1)
            return per[i](x, tf.fill([tf.shape(x)[0]], tv))
        bc = float(np.mean([route_cost(next(r for lo, hi, r in ROUTES if lo <= tv < hi or hi == 1000)) for tv in SAMPLE_TS]))
        yield name, p, bc
    elif name == "routed_learned":
        f_all = tf.function(lambda x, t: net.forward(x, t))
        yield name + "_all", (lambda x, tv: f_all(x, tf.fill([tf.shape(x)[0]], tv))), 1.0
        gates = net.gate_values(tf.constant(SAMPLE_TS, tf.int32)).numpy()
        act = {int(tv): frozenset(b for b, g in zip(ROUTED_BLOCKS, row) if g >= 0.1) for tv, row in zip(SAMPLE_TS, gates)}
        cache = {}
        def p(x, tv):
            a = act[tv]
            if a not in cache:
                cache[a] = tf.function(lambda x, t, a=a: net.forward(x, t, active=a))
            return cache[a](x, tf.fill([tf.shape(x)[0]], tv))
        bc = float(np.mean([BLOCK_COST[[b in act[int(tv)] for b in ROUTED_BLOCKS]].sum() / BLOCK_COST.sum() for tv in SAMPLE_TS]))
        np.save(os.path.join(ROOT, "learned_gates.npy"), net.gate_values(tf.constant(np.arange(0, 1000, 10), tf.int32)).numpy())
        yield name + "_pruned", p, bc
    else:
        f = tf.function(lambda x, t: fwd(x, t))
        yield name, (lambda x, tv: f(x, tf.fill([tf.shape(x)[0]], tv))), (1.0 if name == "routed_dense" else None)


def generate(kind, predict):
    rng = np.random.default_rng(EVAL_SEED)
    idx = SAMPLE_TS[::-1]
    out = []
    for start in range(0, N_EVAL, EVAL_BATCH):
        n = min(EVAL_BATCH, N_EVAL - start)
        z = tf.constant(rng.standard_normal((n, 32, 32, 3)), tf.float32)
        x_t = z
        for i, t in enumerate(idx):
            tp = idx[i + 1] if i + 1 < len(idx) else -1
            if kind == "gauss":
                a_t = ALPHA_BAR[t]; a_p = ALPHA_BAR[tp] if tp >= 0 else tf.constant(1.0)
                eps = predict(x_t, int(t))
                x0 = tf.clip_by_value((x_t - tf.sqrt(1.0 - a_t) * eps) / tf.sqrt(a_t), -1.0, 1.0)
                x_t = tf.sqrt(a_p) * x0 + tf.sqrt(1.0 - a_p) * eps
            else:
                x0 = tf.clip_by_value(predict(x_t, int(t)), -1.0, 1.0)
                if tp >= 0:
                    x_t = x_t - degrade_blur_noise(x0, tf.fill([n], int(t)), z) + degrade_blur_noise(x0, tf.fill([n], int(tp)), z)
                else:
                    x_t = x0
        out.append(np.clip((x_t.numpy() + 1) / 2, 0, 1))
    return np.concatenate(out)


_SCORER = []


def evaluate(name, kind):
    d = run_dir(kind, name)
    res_path = os.path.join(d, "eval.json")
    if os.path.exists(res_path):
        print(open(res_path).read()); return
    net, fwd, opt, ema, ema_n, step, ckpt = make_state(name)
    ckpt.restore(tf.train.latest_checkpoint(os.path.join(d, "ckpt"))).expect_partial()
    for v, e in zip(net.trainable_variables, ema):
        v.assign(e)
    if not _SCORER:
        _SCORER.append(InceptionScorer())
    scorer = _SCORER[0]
    mu, sig, real_is = reference(scorer)
    results = {}
    for label, p, bc in predictors(name, net, fwd):
        t0 = time.time()
        imgs = generate(kind, p)
        samp = time.time() - t0
        f, pr = scorer(imgs)
        m, s = inception_score(pr)
        results[label] = dict(fid=fid_from_features(f, mu, sig), is_mean=m, is_std=s, params=int(net.count_params()),
                              block_compute=bc, sample_seconds=samp, steps=int(step.numpy()))
        np.save(os.path.join(d, f"samples_{label}.npy"), (imgs[:32] * 255).astype(np.uint8))
        print(f"[{kind}/{label}] FID {results[label]['fid']:.2f} IS {m:.2f}  sampling {samp:.0f}s", flush=True)
    json.dump(results, open(res_path, "w"), indent=2)


def report():
    rows = []
    for p in glob.glob(os.path.join(ROOT, "*", "*", "eval.json")):
        kind = p.split(os.sep)[-3]
        tr = json.load(open(os.path.join(os.path.dirname(p), "train_done.json")))
        loss = np.loadtxt(os.path.join(os.path.dirname(p), "loss.csv"), delimiter=",", ndmin=2)
        for label, r in json.load(open(p)).items():
            rows.append((kind, label, r, tr, loss[-1, 1]))
    for kind in ("gauss", "blur", "spec_eq", "spec_c2f"):
        print(f"\n=== {kind} ===")
        print(f"{'model':22s} {'params':>10s} {'ms/step':>8s} {'loss':>8s} {'FID':>8s} {'IS':>6s} {'blk%':>5s} {'samp s':>7s}")
        for k, label, r, tr, l in sorted([x for x in rows if x[0] == kind], key=lambda x: x[2]["fid"]):
            bc = "-" if r["block_compute"] is None else f"{r['block_compute'] * 100:.0f}"
            print(f"{label:22s} {r['params']:>10,} {tr['ms_per_step']:8.0f} {l:8.4f} {r['fid']:8.2f} {r['is_mean']:6.2f} {bc:>5s} {r['sample_seconds']:7.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "report", "params", "train_multi", "eval_multi"])
    ap.add_argument("--model"); ap.add_argument("--models", default=""); ap.add_argument("--kind", default="gauss"); ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.model, a.kind, a.steps)
    elif a.cmd == "eval":
        evaluate(a.model, a.kind)
    elif a.cmd == "train_multi":
        train_multi(a.models.split(","), a.kind, a.steps)
    elif a.cmd == "eval_multi" and "," in a.kind:
        import subprocess
        procs = [subprocess.Popen([sys.executable, "-u", __file__, "eval_multi", "--models", a.models, "--kind", k],
                                  stdout=open(os.path.join(ROOT, "logs", f"eval_{k}.log"), "w"), stderr=subprocess.STDOUT)
                 for k in a.kind.split(",")]
        for p in procs:
            p.wait()
    elif a.cmd == "eval_multi":
        for k in a.kind.split(","):
            for m in a.models.split(","):
                evaluate(m, k); gc.collect()
    elif a.cmd == "params":
        for n in MODELS:
            net, _ = build(n); print(f"{n:16s} {net.count_params():>10,}"); del net; gc.collect()
    else:
        report()
