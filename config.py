"""
    Currently using RMSNorm and SwiGLU.
"""

C = 2048 # context size
D = 1024 # d_model
H = 16   # number of attention heads
DH = 64  # head dim
L = 24   # number of layers

DROP_RATE = 0.1

VOCAB_SIZE = 32000
VOCAB_PATH = "./vocab-slice.txt"

SEED = 42

BATCH_SIZE = 8

TOTAL_STEPS = 300000                        # TODO: 2000 steps for fine-tuning
WARMUP_STEPS = int(0.02 * TOTAL_STEPS)
SAVE_STEPS = 5000                           # TODO: 500 steps for fine-tuning
PEAK_LR = 3e-4                              # TODO: 3e-5 for fine-tuning
MIN_LR = PEAK_LR / 10
TRAIN_TEXT_PATH = "./train-text.txt"
ASS_TEXT_PATH = "./ass-text.txt"
INDICES_PATH = "./train-text-indices.npy"   # pretokenized indices, produced by pretokenize_to_indices.py
ASS_INDICES_PATH = "./ass-text-indices.npy" # produced by pretokenize_ass_to_indices.py

SAVE_PATH = "./ming-gpt"
MODEL_NAME = "model"
BASE_NAME = "base"

WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0 # grad clipping
B1 = 0.9
B2 = 0.999

TEMP = 0.5 # temperature, only used in inference
TOP_K = 10 # only choose the top K words according to their probability, only used in inference
