import torch

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS_STAGE1 = 5

EPOCHS_STAGE2 = 10

LR_STAGE1 = 3e-4

LR_STAGE2 = 1e-5

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PIN_MEMORY = True

MODEL_PATH = "checkpoints/best_model.pth"