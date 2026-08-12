import math

import jax
import jax.numpy as jnp

from dropout import dropout


def init_attention(key, h, d, dh):
    keys = jax.random.split(key, 4)

    query_matrix = jax.random.normal(keys[0], (h, dh, d)) / math.sqrt(d)
    key_matrix = jax.random.normal(keys[1], (h, dh, d)) / math.sqrt(d)
    value_matrix = jax.random.normal(keys[2], (h, dh, d)) / math.sqrt(d)
    output_matrix = jax.random.normal(keys[3], (d, h, dh)) / math.sqrt(h * dh)
    return {"big query matrix": query_matrix, "big key matrix": key_matrix, "big value matrix": value_matrix, "big output matrix": output_matrix}


def attention(key, params, inputs, drop_rate):
    Q = jnp.einsum("ijk, lk -> lij", params["big query matrix"], inputs) #(C, H, DH)
    K = jnp.einsum("ijk, lk -> lij", params["big key matrix"], inputs) #(C, H, DH)
    V = jnp.einsum("ijk, lk -> lij", params["big value matrix"], inputs) #(C, H, DH)

    QKT = jnp.einsum("ijk, ljk -> jil", Q, K) #(H, C, C)
    sqrtdk = math.sqrt(K.shape[2])
    QKT_sqrtdk = QKT / sqrtdk

    QKT_sqrtdk = QKT_sqrtdk.astype(jnp.float32)

    mask = jnp.triu(jnp.ones(QKT.shape[1:3]), k=1).astype(bool)
    QKT_sqrtdk_m = jnp.where(mask, -jnp.inf, QKT_sqrtdk)

    softed = jax.nn.softmax(QKT_sqrtdk_m, axis=2)

    softed = softed.astype(jnp.bfloat16)

    attention = jnp.einsum("ijk, kil -> jil", softed, V) #(C, H, DH)

    attention = dropout(key, attention, drop_rate)

    output = jnp.einsum("ijk, ljk -> li", params["big output matrix"], attention)

    return output