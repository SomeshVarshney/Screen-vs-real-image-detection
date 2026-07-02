import os
import json
import random
import numpy as np
import torch

from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.amp import GradScaler

import config

from utils import load_dataset
from dataset import ScreenDataset
from model import ScreenClassifier
from engine import train_one_epoch, validate


# ---------------------------------------
# Reproducibility
# ---------------------------------------

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

device = config.DEVICE

print("=" * 60)
print("Device :", device)
print("=" * 60)


# ---------------------------------------
# Dataset
# ---------------------------------------

paths, labels = load_dataset("data/original")

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=config.SEED
)

train_idx, val_idx = next(
    splitter.split(paths, labels)
)

train_paths = [paths[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]

val_paths = [paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]


train_dataset = ScreenDataset(
    train_paths,
    train_labels,
    train=True
)

val_dataset = ScreenDataset(
    val_paths,
    val_labels,
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")


# ---------------------------------------
# Model
# ---------------------------------------

model = ScreenClassifier().to(device)

weights = torch.tensor(
    [1.0,1.17],
    device=device
)

criterion = torch.nn.CrossEntropyLoss(
    weight=weights
)

optimizer = AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=config.LR_STAGE1,

    weight_decay=config.WEIGHT_DECAY

)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config.EPOCHS_STAGE1
)

scaler = GradScaler("cuda")

best_f1 = 0
best_auc = 0

history = []

patience = 5
patience_counter = 0

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# =====================================================
# STAGE 1 - Train Classifier Only
# =====================================================
# ==========================================================
# STAGE 1
# ==========================================================

print("\n" + "=" * 60)
print("STAGE 1 : TRAINING CLASSIFIER")
print("=" * 60)

for epoch in range(config.EPOCHS_STAGE1):

    print(f"\nEpoch {epoch+1}/{config.EPOCHS_STAGE1}")

    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        device=device
    )

    metrics = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device
    )

    scheduler.step()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {metrics['loss']:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1 Score   : {metrics['f1']:.4f}")
    print(f"ROC AUC    : {metrics['roc_auc']:.4f}")

    history.append({

        "stage":1,

        "epoch":epoch+1,

        "train_loss":train_loss,

        **metrics

    })

    save_model = False

    if metrics["f1"] > best_f1:

        save_model = True

    elif abs(metrics["f1"]-best_f1) < 1e-4:

        if metrics["roc_auc"] > best_auc:

            save_model = True

    if save_model:

        best_f1 = metrics["f1"]
        best_auc = metrics["roc_auc"]

        patience_counter = 0

        torch.save(

            {

                "model_state_dict":model.state_dict(),

                "f1":best_f1,

                "roc_auc":best_auc

            },

            config.MODEL_PATH

        )

        print(

            f"\n✅ Best Model Saved "

            f"(F1={best_f1:.4f} "

            f"ROC={best_auc:.4f})"

        )

    else:

        patience_counter += 1

        print(

            f"Patience : "

            f"{patience_counter}/{patience}"

        )

    if patience_counter >= patience:

        print("\nEarly Stopping Stage 1")

        break
    
 # ==========================================================
# LOAD BEST STAGE-1 MODEL
# ==========================================================

print("\n")
print("=" * 60)
print("LOADING BEST STAGE-1 MODEL")
print("=" * 60)

checkpoint = torch.load(
    config.MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_f1 = checkpoint["f1"]
best_auc = checkpoint["roc_auc"]

print(f"Loaded Best Stage-1 Model (F1={best_f1:.4f})")

# ==========================================================
# UNFREEZE BACKBONE
# ==========================================================

print("\n")
print("=" * 60)
print("UNFREEZING BACKBONE")
print("=" * 60)

model.unfreeze()

optimizer = AdamW(

    model.parameters(),

    lr=config.LR_STAGE2,

    weight_decay=config.WEIGHT_DECAY

)

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=config.EPOCHS_STAGE2

)

patience_counter = 0

print("\nStarting Fine Tuning...\n")

# ==========================================================
# STAGE 2 TRAINING
# ==========================================================

for epoch in range(config.EPOCHS_STAGE2):

    print(f"\nEpoch {epoch+1}/{config.EPOCHS_STAGE2}")

    train_loss = train_one_epoch(

        model=model,

        loader=train_loader,

        optimizer=optimizer,

        criterion=criterion,

        scaler=scaler,

        device=device

    )

    metrics = validate(

        model=model,

        loader=val_loader,

        criterion=criterion,

        device=device

    )

    scheduler.step()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {metrics['loss']:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1 Score   : {metrics['f1']:.4f}")
    print(f"ROC AUC    : {metrics['roc_auc']:.4f}")

    history.append({

        "stage":2,

        "epoch":config.EPOCHS_STAGE1 + epoch + 1,

        "train_loss":train_loss,

        **metrics

    })

    save_model = False

    if metrics["f1"] > best_f1:

        save_model = True

    elif abs(metrics["f1"] - best_f1) < 1e-4:

        if metrics["roc_auc"] > best_auc:

            save_model = True

    if save_model:

        best_f1 = metrics["f1"]
        best_auc = metrics["roc_auc"]

        patience_counter = 0

        torch.save(

            {

                "model_state_dict":model.state_dict(),

                "f1":best_f1,

                "roc_auc":best_auc

            },

            config.MODEL_PATH

        )

        print(

            f"\n✅ Best Model Saved "

            f"(F1={best_f1:.4f} "

            f"ROC={best_auc:.4f})"

        )

    else:

        patience_counter += 1

        print(f"Patience : {patience_counter}/{patience}")

    if patience_counter >= patience:

        print("\nEarly Stopping Stage 2")

        break
# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

with open(
    "checkpoints/history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(f"Best F1 Score : {best_f1:.4f}")
print(f"Best ROC AUC  : {best_auc:.4f}")

print(f"Model Saved   : {config.MODEL_PATH}")

print("=" * 60)

print("\nTraining history saved to:")
print("checkpoints/history.json")