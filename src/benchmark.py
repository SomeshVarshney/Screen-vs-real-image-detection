import os
import time
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import config
from utils import load_dataset
from dataset import ScreenDataset
from model import ScreenClassifier

# -------------------------------------------------
# Device
# -------------------------------------------------

device = config.DEVICE

print("=" * 70)
print("DEVICE :", device)
print("=" * 70)

# -------------------------------------------------
# Dataset
# -------------------------------------------------

paths, labels = load_dataset("data/original")

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=config.SEED
)

_, val_idx = next(splitter.split(paths, labels))

val_paths = [paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

dataset = ScreenDataset(
    val_paths,
    val_labels,
    train=False
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# -------------------------------------------------
# Model
# -------------------------------------------------

start = time.perf_counter()

model = ScreenClassifier().to(device)

checkpoint = torch.load(
    config.MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

if device == "cuda":
    torch.cuda.synchronize()

load_time = time.perf_counter() - start

# -------------------------------------------------
# Model Stats
# -------------------------------------------------

params = sum(p.numel() for p in model.parameters())

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

model_size = os.path.getsize(
    config.MODEL_PATH
) / (1024 * 1024)

# -------------------------------------------------
# Warmup
# -------------------------------------------------

dummy = next(iter(loader))[0].to(device)

with torch.no_grad():

    for _ in range(20):

        _ = model(dummy)

if device == "cuda":
    torch.cuda.synchronize()

# -------------------------------------------------
# Benchmark
# -------------------------------------------------

times = []

predictions = []
targets = []
probabilities = []

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

total_start = time.perf_counter()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        outputs = model(images)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append(end - start)

        probs = torch.softmax(outputs, 1)[:, 1]

        preds = torch.argmax(outputs, 1)

        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())
        targets.extend(labels.numpy())

total_time = time.perf_counter() - total_start

# -------------------------------------------------
# Metrics
# -------------------------------------------------

accuracy = accuracy_score(targets, predictions)

precision = precision_score(targets, predictions)

recall = recall_score(targets, predictions)

f1 = f1_score(targets, predictions)

auc = roc_auc_score(targets, probabilities)

avg_time = np.mean(times)

fps = len(dataset) / total_time

# -------------------------------------------------
# GPU Memory
# -------------------------------------------------

if device == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

else:

    peak_memory = 0

# -------------------------------------------------
# Report
# -------------------------------------------------

print()

print("=" * 70)
print("MODEL BENCHMARK")
print("=" * 70)

print(f"Model               : EfficientNet-B0")
print(f"Device              : {device}")

print()

print(f"Images Tested       : {len(dataset)}")

print(f"Model Size          : {model_size:.2f} MB")

print(f"Parameters          : {params:,}")

print(f"Trainable Params    : {trainable:,}")

print()

print(f"Model Load Time     : {load_time*1000:.2f} ms")

print(f"Average Latency     : {avg_time*1000:.2f} ms/image")

print(f"Total Time          : {total_time:.2f} sec")

print(f"Throughput          : {fps:.2f} FPS")

print()

print(f"Peak GPU Memory     : {peak_memory:.2f} MB")

print()

print(f"Accuracy            : {accuracy:.4f}")

print(f"Precision           : {precision:.4f}")

print(f"Recall              : {recall:.4f}")

print(f"F1 Score            : {f1:.4f}")

print(f"ROC-AUC             : {auc:.4f}")

print("=" * 70)