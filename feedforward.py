import math

import jax
import jax.numpy as jnp


def init_feedforward(key, d):
    keys = jax.random.split(key, 2)

    up_matrix = jax.random.normal(keys[0], (4 * d, d)) / math.sqrt(d)
    up_bias = jnp.zeros(4 * d)
    down_matrix = jax.random.normal(keys[1], (d, 4 * d)) / math.sqrt(4 * d)
    down_bias = jnp.zeros(d)
    return {"up matrix": up_matrix, "up bias": up_bias, "down matrix": down_matrix, "down bias": down_bias}


def feedforward(params, inputs):
    mid = jax.nn.gelu(jnp.einsum("ij, kj -> ki", params["up matrix"], inputs) + params["up bias"])
    output = jnp.einsum("ij, kj -> ki", params["down matrix"], mid) + params["down bias"]
    return output