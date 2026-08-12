import json
import random
import os

import jax
import jax.numpy as jnp
from jax import checkpoint

from swiglu import init_swiglu, swiglu
from attention import init_attention, attention
from rms_norm import init_rms_norm, rms_norm
from token_embed import init_embed, get_embed_indices, get_embeddings, get_unembedding, sample_word
from loss import CE_loss
from tokenizer import tokenize
from vocabularizer import USER_TOKEN, ASS_TOKEN, STOP_TOKEN
from positional_embed import init_pos_embed, pos_embed
from adam import init_adam, adam
from config import C, D, H, DH, L, DROP_RATE, TEMP, TOP_K, SAVE_PATH, VOCAB_SIZE


@jax.jit
def forward(key, params, indices, PE):
    keys = jax.random.split(key, L)

    params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    PE = jax.tree.map(lambda x: x.astype(jnp.bfloat16), PE)

    current_layer = pos_embed(PE, get_embeddings(params["embed"], indices))

    @checkpoint
    def layer_fn(current_layer, param_i, key):
        current_layer = current_layer + attention(
            key, param_i["attention"],
            rms_norm(param_i["rms norm 1"], current_layer),
            drop_rate=DROP_RATE)
        current_layer = current_layer + swiglu(
            param_i["swiglu"],
            rms_norm(param_i["rms norm 2"], current_layer))
        return current_layer

    for i in range(len(params["layers"])):
        current_layer = layer_fn(current_layer, params["layers"][i], keys[i])

    return get_unembedding(params["embed"], rms_norm(params["last rms norm"], current_layer))


def loss(key, params, indices, target_indices, PE):
    return CE_loss(forward(key, params, indices, PE), target_indices)


def batch_loss(key, params, indices, target_indices, PE):
    keys = jax.random.split(key, indices.shape[0])
    losses = jax.vmap(loss, in_axes=(0, None, 0, 0, None))(keys, params, indices, target_indices, PE)
    return jnp.mean(losses)


@jax.jit
def jitted_param_to_param(key, params, adam_state, indices, target_indices, PE):
    l, grads = jax.value_and_grad(batch_loss, argnums=1)(key, params, indices, target_indices, PE)
    new_params, new_adam_state = adam(params, grads, adam_state)
    return new_params, new_adam_state, l


def masked_loss(key, params, indices, target_indices, mask, PE):
    return CE_loss(forward(key, params, indices, PE), target_indices, mask)


def masked_batch_loss(key, params, indices, target_indices, mask, PE):
    keys = jax.random.split(key, indices.shape[0])
    losses = jax.vmap(masked_loss, in_axes=(0, None, 0, 0, 0, None))(keys, params, indices, target_indices, mask, PE)
    return jnp.mean(losses)


@jax.jit
def jitted_finetune(key, params, adam_state, indices, target_indices, mask, PE):
    l, grads = jax.value_and_grad(masked_batch_loss, argnums=1)(key, params, indices, target_indices, mask, PE)
    new_params, new_adam_state = adam(params, grads, adam_state)
    return new_params, new_adam_state, l

class MingGPT:
    def __init__(self, key=None):
        self.vocabulary = sorted(json.loads(open(os.path.join(SAVE_PATH, "vocabulary.json")).read()), key=len, reverse=True)

        if key is not None:
            keys = jax.random.split(key, 2 * L)

            self.params = {"layers": [], "embed": {}}
            for i in range(L):
                self.params["layers"].append({
                    "rms norm 1": init_rms_norm(D),
                    "attention": init_attention(keys[2 * i], H, D, DH),
                    "rms norm 2": init_rms_norm(D),
                    "swiglu": init_swiglu(keys[2 * i + 1], D)
                })
            self.params["last rms norm"] = init_rms_norm(D)
            self.params["embed"] = init_embed(key, VOCAB_SIZE, D)

            self.adam_state = init_adam(self.params)

        self.PE = init_pos_embed(C, D)

    def infer(self, key, indices):
        keys = jax.random.split(key)

        # empty tokens are zero
        n = indices.shape[0]
        padded = jnp.zeros(C, dtype=indices.dtype).at[:n].set(indices)

        unembedding = forward(keys[0], self.params, padded, self.PE)[n - 1]

        return sample_word(keys[1], self.vocabulary, unembedding, TEMP, TOP_K)

    def train(self, key, indices, target_indices):
        self.params, self.adam_state, l = jitted_param_to_param(key, self.params, self.adam_state, indices, target_indices, self.PE)
        return l

    def finetune(self, key, indices, target_indices, mask):
        self.params, self.adam_state, l = jitted_finetune(key, self.params, self.adam_state, indices, target_indices, mask, self.PE)
        return l

    def infer_prompt(self, prompt):
        input_text = f"{USER_TOKEN} {prompt} {ASS_TOKEN}".lower()
        output_text = ""
        for i in range(1000):
            key = jax.random.key(random.randint(0, 1000))

            # get last C tokens
            tokens = tokenize(self.vocabulary, input_text)
            if len(tokens) > C:
                tokens = tokens[len(tokens) - C :]

            indices = jnp.array(get_embed_indices(self.vocabulary, tokens))

            token = self.infer(key, indices)
            if token == STOP_TOKEN:
                break
            input_text += token
            output_text += token
        else:
            return "Error: generated response too long"

        return output_text
