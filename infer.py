from gpt import MingGPT
from utils import load_into

model = MingGPT()
load_into(model)

def get_response(prompt):
    print(model.infer_prompt(prompt))

if __name__ == "__main__":
    get_response("How are you doing?")
