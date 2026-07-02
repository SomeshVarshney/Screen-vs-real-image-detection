import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    running_loss = 0.0

    progress = tqdm(loader, leave=False)

    for images, labels in progress:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device,
            enabled=(device == "cuda")
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    loss_sum = 0.0

    predictions = []

    probabilities = []

    targets = []

    with torch.no_grad():

        for images, labels in tqdm(loader, leave=False):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss_sum += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())

            probabilities.extend(probs.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    metrics = {

        "loss": loss_sum / len(loader),

        "accuracy": accuracy_score(targets, predictions),

        "precision": precision_score(
            targets,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            targets,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            targets,
            predictions,
            zero_division=0
        )
    }

    try:

        metrics["roc_auc"] = roc_auc_score(
            targets,
            probabilities
        )

    except:

        metrics["roc_auc"] = 0.0

    return metrics