import math

import numpy as np
import jax
import jax.numpy as jnp


def init_embed(key, v, d):
    embed_matrix = jax.random.normal(key, (v, d)) / math.sqrt(d)
    return {"embed matrix": embed_matrix}


def get_embed_indices(sorted_vocabulary, tokens):
    lookup = {token: i for i, token in enumerate(sorted_vocabulary)}
    return np.array([lookup[token] for token in tokens])


def get_embeddings(params, indices):
    return params["embed matrix"][indices]


def get_unembedding(params, outputs):
    return jnp.einsum("ij, kj -> ki", params["embed matrix"], outputs)


def sample_word(key, sorted_vocabulary, unembedding, temp, top_k):
    temped_unembedding = unembedding / temp

    temped_unembedding = temped_unembedding.astype(jnp.float32)

    prob_dist = jax.nn.softmax(temped_unembedding)
    probs, indices = jax.lax.top_k(prob_dist, top_k)
    probs = probs / jnp.sum(probs)
    predicted_index = jax.random.choice(key, indices, p=probs)
    return sorted_vocabulary[predicted_index]