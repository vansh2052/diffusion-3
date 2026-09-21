"""Training-speed benchmark for the paper-sized DDPM model in ddpm_paper.py."""
import argparse
import time

import tensorflow as tf

from ddpm_paper import DDPMUNet, TIMESTEPS

PAPER_STEPS = 800_000


def make_train_step(model, optimizer, ema_vars, xla):
    betas = tf.linspace(1e-4, 0.02, TIMESTEPS)
    alpha_bar = tf.math.cumprod(1.0 - betas)

    @tf.function(jit_compile=xla)
    def train_step(x0):
        b = tf.shape(x0)[0]
        t = tf.random.uniform((b,), 0, TIMESTEPS, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x0))
        ab = tf.gather(alpha_bar, t)[:, None, None, None]
        xt = tf.sqrt(ab) * x0 + tf.sqrt(1.0 - ab) * noise
        with tf.GradientTape() as tape:
            pred = tf.cast(model([xt, t], training=True), tf.float32)
            loss = tf.reduce_mean(tf.square(noise - pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for e, v in zip(ema_vars, model.trainable_variables):
            e.assign(0.9999 * e + (1.0 - 0.9999) * v)
        return loss

    return train_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--xla", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--smoke", action="store_true", help="tiny CPU check, no timing")
    a = p.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if a.precision == "bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    model = DDPMUNet()
    model([tf.zeros((1, 32, 32, 3)), tf.zeros((1,), tf.int32)])
    n_params = model.count_params()
    print(f"config: precision={a.precision} xla={a.xla} batch={a.batch}")
    print(f"parameters: {n_params:,} (paper reports 35.7M)")

    optimizer = tf.keras.optimizers.Adam(2e-4)
    ema_vars = [tf.Variable(v, trainable=False) for v in model.trainable_variables]
    step = make_train_step(model, optimizer, ema_vars, bool(a.xla))

    if a.smoke:
        x = tf.random.normal((2, 32, 32, 3))
        print("smoke loss:", float(step(x)))
        return

    x = tf.random.normal((a.batch, 32, 32, 3))
    t0 = time.time()
    for _ in range(a.warmup):
        loss = step(x)
    float(loss)
    print(f"warmup ({a.warmup} steps, includes compile): {time.time() - t0:.1f}s")

    t0 = time.time()
    for _ in range(a.steps):
        loss = step(x)
    float(loss)
    dt = time.time() - t0
    sec = dt / a.steps
    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**30
    print(f"loss: {float(loss):.4f}  peak GPU memory: {peak:.1f} GiB")
    print(f"speed: {a.steps / dt:.2f} steps/s ({sec * 1000:.0f} ms/step)")
    print(f"estimate for {PAPER_STEPS:,} steps: {PAPER_STEPS * sec / 3600:.1f} hours")


if __name__ == "__main__":
    main()
