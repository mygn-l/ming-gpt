import jax.numpy as jnp


EPSILON = 1e-9 # prevent division by zero


def init_layer_norm(d):
    gamma = jnp.ones(d)
    beta = jnp.zeros(d)
    return {"gamma": gamma, "beta": beta}


def layer_norm(params, inputs):
    means = jnp.mean(inputs, axis=1, keepdims=True)
    centered = inputs - means

    variance = jnp.var(inputs, axis=1, keepdims=True)
    normalized = centered / jnp.sqrt(variance + EPSILON)

    output = params["gamma"] * normalized + params["beta"]
    return output
