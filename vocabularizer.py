import os
import json

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

from config import VOCAB_SIZE, VOCAB_PATH, SAVE_PATH


UNK_TOKEN = "<unk>"
STOP_TOKEN = "<stop>"
USER_TOKEN = "<user>"
ASS_TOKEN = "<ass>"

SPECIAL_TOKENS = [UNK_TOKEN, STOP_TOKEN, USER_TOKEN, ASS_TOKEN]


"""
    BPE vocabulary with HuggingFace. By Claude.
"""
def vocabularizer():
    alphabet = set()
    with open(VOCAB_PATH, encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            alphabet.update(chunk.lower())

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=sorted(alphabet),
        show_progress=True,
    )
    tokenizer.train([VOCAB_PATH], trainer)

    vocabulary = list(tokenizer.get_vocab().keys())
    vocabulary.sort(key=len, reverse=True)
    return vocabulary


if __name__ == "__main__":
    print("Starting BPE vocabulary generation")

    final_vocabulary = vocabularizer()

    print(f"Done. Vocab size: {VOCAB_SIZE}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    vocab_save_path = os.path.join(SAVE_PATH, "vocabulary.json")
    with open(vocab_save_path, "w") as file:
        file.write(json.dumps(final_vocabulary))

    print(f"Saved JSON at {vocab_save_path}")
