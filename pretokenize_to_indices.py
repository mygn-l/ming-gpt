import numpy as np

from gpt import MingGPT
from tokenizer import tokenize
from token_embed import get_embed_indices
from config import TRAIN_TEXT_PATH, INDICES_PATH


vocab = MingGPT().vocabulary

print("Tokenizing...")

indices_parts = []
remaining = ""
with open(TRAIN_TEXT_PATH) as f:
    while True:
        chunk = f.read(100_000_000)
        if not chunk:
            break
        remaining += chunk

        cut = remaining.rfind("\n")
        if cut == -1:
            continue

        tokens = tokenize(vocab, remaining[:cut + 1])
        indices_parts.append(np.array(get_embed_indices(vocab, tokens), dtype=np.uint16))

        remaining = remaining[cut + 1:]
if remaining:
    tokens = tokenize(vocab, remaining)
    indices_parts.append(np.array(get_embed_indices(vocab, tokens), dtype=np.uint16))

indices = np.concatenate(indices_parts)
np.save(INDICES_PATH, indices)
print(f"Saved indices to {INDICES_PATH}")
