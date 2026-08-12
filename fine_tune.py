import jax
import jax.numpy as jnp
import numpy as np

from gpt import MingGPT
from adam import init_adam
from vocabularizer import USER_TOKEN, ASS_TOKEN, STOP_TOKEN
from utils import load_into, save
from config import TOTAL_STEPS, BATCH_SIZE, SEED, SAVE_STEPS, C, BASE_NAME, ASS_INDICES_PATH


my_model = MingGPT(jax.random.key(SEED))
load_into(my_model, BASE_NAME)
my_model.adam_state = init_adam(my_model.params)


indices = np.load(ASS_INDICES_PATH)

lookup = {tok: i for i, tok in enumerate(my_model.vocabulary)}
user_id = lookup[USER_TOKEN]
ass_id = lookup[ASS_TOKEN]
stop_id = lookup[STOP_TOKEN]

mask = np.zeros(len(indices))
in_response = False
for i in range(len(indices)):
    token = indices[i]
    if token == user_id:
        in_response = False
    elif token == ass_id:
        in_response = True
    elif token == stop_id:
        mask[i] = 1
        in_response = False
    elif in_response:
        mask[i] = 1

indices = jnp.array(indices)
mask = jnp.array(mask)

for step in range(TOTAL_STEPS):
    key = jax.random.key(step)

    starts = jax.random.randint(key, (BATCH_SIZE, 1), 0, len(indices) - C - 1)
    furthers = jnp.arange(C)
    input_indices = indices[starts + furthers]
    target_indices = indices[starts + furthers + 1]
    target_masks = mask[starts + furthers + 1]

    my_model.finetune(key, input_indices, target_indices, target_masks)

    if step % 100 == 0:
        print(f"Current steps: {step}")

    if step % SAVE_STEPS == 0:
        save(my_model)
        print(f"Model checkpoint saved")
save(my_model) #final save

print(f"Fine-tuning done: {TOTAL_STEPS} steps")
