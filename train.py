import os

import jax
import jax.numpy as jnp
import numpy as np

from gpt import MingGPT
from utils import load_into, save
from config import TOTAL_STEPS, BATCH_SIZE, SEED, SAVE_STEPS, C, L, D, DH, H, DROP_RATE, VOCAB_SIZE, VOCAB_PATH, WARMUP_STEPS, PEAK_LR, SAVE_PATH, INDICES_PATH, BASE_NAME, TRAIN_TEXT_PATH


my_model = MingGPT(jax.random.key(SEED))

if os.path.isfile(os.path.join(SAVE_PATH, BASE_NAME)):
    load_into(my_model, BASE_NAME)


indices = jnp.array(np.load(INDICES_PATH))

print(f"Started training for MING-GPT")
print("___")
print(f"VOCABULARY")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Text used: {VOCAB_PATH}")
print("___")
print(f"ARCHITECTURE")
print(f"Context size: {C}")
print(f"Layers: {L}")
print(f"d_model: {D}")
print(f"Heads: {H}")
print(f"dim head: {DH}")
print("___")
print("TRAINING SETTINGS")
print(f"Seed: {SEED}")
print(f"Text: {TRAIN_TEXT_PATH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total steps: {TOTAL_STEPS}")
print(f"Warmup steps: {WARMUP_STEPS}")
print(f"Peak learning rate: {PEAK_LR}")
print(f"Dropout rate: {DROP_RATE}")
print("___")
print(f"Model checkpoint will be saved at every {SAVE_STEPS} steps")

for i in range(TOTAL_STEPS):
    key = jax.random.key(i)

    starts = jax.random.randint(key, (BATCH_SIZE, 1), 0, len(indices) - C - 1)
    furthers = jnp.arange(C)
    input_indices = indices[starts + furthers]
    target_indices = indices[starts + furthers + 1]

    my_model.train(key, input_indices, target_indices)

    if i % 100 == 0:
        print(f"Current steps: {i}")

    if i % SAVE_STEPS == 0:
        save(my_model, BASE_NAME)
        print(f"Model checkpoint saved")
save(my_model, BASE_NAME) #final save

print(f"Training done: {TOTAL_STEPS} steps")
