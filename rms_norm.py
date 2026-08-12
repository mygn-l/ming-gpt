import jax.numpy as jnp


EPSILON = 1e-9 # prevent division by zero


def init_rms_norm(d):
    gamma = jnp.ones(d)
    return {"gamma": gamma}


def rms_norm(params, inputs):
    inputs = inputs.astype(jnp.float32)

    rms = jnp.sqrt(jnp.mean(jnp.square(inputs), axis=1, keepdims=True) + EPSILON)
    output = inputs / rms * params["gamma"]

    output = output.astype(jnp.bfloat16)
    return output
