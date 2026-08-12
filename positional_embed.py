import math

import jax.numpy as jnp


def init_pos_embed(c, d):
    PE = jnp.array([[c / (10000 ** (i / d)) for i in range(d)] for c in range(c)])
    PE = PE.at[:, 0::2].set(jnp.sin(PE[:, 0::2]))
    PE = PE.at[:, 1::2].set(jnp.cos(PE[:, 1::2]))
    PE = PE / math.sqrt(d)
    return {"PE": PE}


def pos_embed(params, inputs):
    return inputs + params["PE"][:inputs.shape[0]]