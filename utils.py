import pickle
import os

from config import SAVE_PATH, MODEL_NAME


def save(model, name=MODEL_NAME):
    with open(os.path.join(SAVE_PATH, name), "wb") as f:
        pickle.dump({
            "params": model.params,
            "adam state": model.adam_state
        }, f)

def load_into(model, name=MODEL_NAME):
    with open(os.path.join(SAVE_PATH, name), "rb") as f:
        data = pickle.load(f)
        model.params = data["params"]
        model.adam_state = data["adam state"]
