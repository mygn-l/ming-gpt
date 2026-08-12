import jax
import jax.numpy as jnp


def CE_loss(unembeddings, target_indices, mask=None):
    unembeddings = unembeddings.astype(jnp.float32)
    per_token = jax.nn.logsumexp(unembeddings, axis=1) - unembeddings[jnp.arange(unembeddings.shape[0]), target_indices]
    if mask is None:
        return jnp.mean(per_token)
    return jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)
