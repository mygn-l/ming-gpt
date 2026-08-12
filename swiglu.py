import math

import jax
import jax.numpy as jnp


def init_swiglu(key, d):
    keys = jax.random.split(key, 3)

    mid_dim = int(8 / 3 * d)
    gate_matrix = jax.random.normal(keys[0], (mid_dim, d)) / math.sqrt(d)
    up_matrix = jax.random.normal(keys[1], (mid_dim, d)) / math.sqrt(d)
    down_matrix = jax.random.normal(keys[2], (d, mid_dim)) / math.sqrt(mid_dim)
    return {"gate matrix": gate_matrix, "up matrix": up_matrix, "down matrix": down_matrix}


def swiglu(params, inputs):
    gate = jnp.einsum("ij, kj -> ki", params["gate matrix"], inputs)
    silu = jax.nn.sigmoid(gate) * gate
    mid = silu * jnp.einsum("ij, kj -> ki", params["up matrix"], inputs)
    output = jnp.einsum("ij, kj -> ki", params["down matrix"], mid)
    return output