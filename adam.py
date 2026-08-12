import jax
import jax.numpy as jnp

from config import WARMUP_STEPS, PEAK_LR, WEIGHT_DECAY, MAX_GRAD_NORM, B1, B2


EPSILON = 1e-8


def init_adam(params):
    return {
        "m": jax.tree.map(jnp.zeros_like, params),
        "v": jax.tree.map(jnp.zeros_like, params),
        "t": jnp.array(0)
    }


def adam(params, grads, state):
    leaves = jax.tree.leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
    scale_factor = jnp.minimum(1.0, MAX_GRAD_NORM / (global_norm + EPSILON))
    normed_grads = jax.tree.map(lambda g: g * scale_factor, grads)

    t = state["t"] + 1

    lr = PEAK_LR * jnp.minimum(1.0, t / WARMUP_STEPS)

    m = jax.tree.map(lambda m, g: B1 * m + (1 - B1) * g, state["m"], normed_grads)
    v = jax.tree.map(lambda v, g: B2 * v + (1 - B2) * g * g, state["v"], normed_grads)
    scale = lr * jnp.sqrt(1 - B2 ** t) / (1 - B1 ** t)
    new_params = jax.tree.map(lambda p, m, v: p - scale * m / (jnp.sqrt(v) + EPSILON) - lr * WEIGHT_DECAY * (p if p.ndim >= 2 else 0), params, m, v)
    return new_params, {"m": m, "v": v, "t": t}
