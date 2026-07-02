import os
import shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ScreenDataset
from model import ScreenClassifier
from utils import load_dataset


device = config.DEVICE

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

paths, labels = load_dataset("data/original")

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=config.SEED
)

_, val_idx = next(splitter.split(paths, labels))

val_paths = [paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

val_dataset = ScreenDataset(
    val_paths,
    val_labels,
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = ScreenClassifier().to(device)

checkpoint = torch.load(
    config.MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# -------------------------------------------------------
# Predict Probabilities
# -------------------------------------------------------

probs = []
targets = []

with torch.no_grad():

    for images, labels in tqdm(val_loader):

        images = images.to(device)

        outputs = model(images)

        p = torch.softmax(outputs, dim=1)[:,1]

        probs.extend(
            p.cpu().numpy()
        )

        targets.extend(
            labels.numpy()
        )

probs = np.array(probs)
targets = np.array(targets)

# -------------------------------------------------------
# Search Best Threshold
# -------------------------------------------------------

best_threshold = 0.50
best_f1 = 0

print("\nSearching Best Threshold...\n")

for threshold in np.arange(0.20,0.81,0.01):

    predictions = (probs >= threshold).astype(int)

    score = f1_score(
        targets,
        predictions
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = threshold

predictions = (
    probs >= best_threshold
).astype(int)

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

accuracy = accuracy_score(
    targets,
    predictions
)

precision = precision_score(
    targets,
    predictions
)

recall = recall_score(
    targets,
    predictions
)

f1 = f1_score(
    targets,
    predictions
)

auc = roc_auc_score(
    targets,
    probs
)

print("="*60)

print(f"Best Threshold : {best_threshold:.2f}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC        : {auc:.4f}")

print("="*60)

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

cm = confusion_matrix(
    targets,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real","Screen"]
)

disp.plot(cmap="Blues")

plt.savefig(
    "checkpoints/confusion_matrix.png",
    dpi=300
)

plt.close()

# -------------------------------------------------------
# ROC Curve
# -------------------------------------------------------

fpr,tpr,_ = roc_curve(
    targets,
    probs
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.grid(True)

plt.savefig(
    "checkpoints/roc_curve.png",
    dpi=300
)

plt.close()

# -------------------------------------------------------
# Save Misclassified Images
# -------------------------------------------------------

real_folder = Path(
    "checkpoints/misclassified/real_as_screen"
)

screen_folder = Path(
    "checkpoints/misclassified/screen_as_real"
)

real_folder.mkdir(
    parents=True,
    exist_ok=True
)

screen_folder.mkdir(
    parents=True,
    exist_ok=True
)

real_wrong = 0
screen_wrong = 0

for path,true,pred in zip(
    val_paths,
    targets,
    predictions
):

    if true == pred:
        continue

    filename = Path(path).name

    if true == 0:

        shutil.copy(
            path,
            real_folder / filename
        )

        real_wrong += 1

    else:

        shutil.copy(
            path,
            screen_folder / filename
        )

        screen_wrong += 1

print()

print("Misclassified Images")

print("--------------------------")

print("Real -> Screen :", real_wrong)

print("Screen -> Real :", screen_wrong)

print()

print("Evaluation Complete.")