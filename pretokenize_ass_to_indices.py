import numpy as np

from gpt import MingGPT
from tokenizer import tokenize
from token_embed import get_embed_indices
from config import ASS_TEXT_PATH, ASS_INDICES_PATH


vocab = MingGPT().vocabulary
text = open(ASS_TEXT_PATH).read()

print("Tokenizing...")
tokens = tokenize(vocab, text)

indices = np.array(get_embed_indices(vocab, tokens), dtype=np.uint16)

np.save(ASS_INDICES_PATH, indices)
print(f"Saved indices to {ASS_INDICES_PATH}")
