import os
import sys
import time
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import ScreenClassifier


# --------------------------------------------------------
# Device
# --------------------------------------------------------

device = config.DEVICE

print(f"\nUsing Device : {device.upper()}\n")


# --------------------------------------------------------
# Transform
# --------------------------------------------------------

transform = A.Compose([

    A.Resize(
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()

])


# --------------------------------------------------------
# Load Model
# --------------------------------------------------------

print("Loading model...")

model = ScreenClassifier().to(device)

checkpoint = torch.load(
    config.MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

model.eval()

print("Model Loaded Successfully!\n")


# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

def predict(image_path, threshold=0.5):

    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    image = transform(
        image=image
    )["image"]

    image = image.unsqueeze(0).to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    if device == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() - start
    ) * 1000

    real_prob = probabilities[0].item()
    screen_prob = probabilities[1].item()

    prediction = (
        "SCREEN IMAGE"
        if screen_prob >= threshold
        else "REAL IMAGE"
    )

    confidence = max(
        real_prob,
        screen_prob
    )

    return {

        "prediction": prediction,

        "confidence": confidence,

        "real_probability": real_prob,

        "screen_probability": screen_prob,

        "time": inference_time

    }


# --------------------------------------------------------
# Pretty Output
# --------------------------------------------------------

def show_result(image_path, result):

    print("\n" + "="*60)

    print("Prediction Result")

    print("="*60)

    print(f"Image               : {image_path}")

    print(f"Prediction          : {result['prediction']}")

    print(f"Confidence          : {result['confidence']*100:.2f}%")

    print(f"Real Probability    : {result['real_probability']*100:.2f}%")

    print(f"Screen Probability  : {result['screen_probability']*100:.2f}%")

    print(f"Inference Time      : {result['time']:.2f} ms")

    print("="*60)

def show_image(image_path, result):

    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image)
    ax.axis("off")

    if result["prediction"] == "REAL IMAGE":
        color = "green"
        emoji = "🟢"
    else:
        color = "red"
        emoji = "🔴"

    title = (
        f"{emoji} {result['prediction']}\n"
        f"Confidence : {result['confidence']*100:.2f}%\n"
        f"Inference Time : {result['time']:.2f} ms"
    )

    ax.set_title(
        title,
        fontsize=16,
        color=color,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    os.makedirs("predictions", exist_ok=True)

    save_path = os.path.join(
        "predictions",
        os.path.basename(image_path)
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"\nPrediction image saved to:\n{save_path}")

    plt.show()
# --------------------------------------------------------
# Main
# --------------------------------------------------------

def interactive():

    while True:

        image_path = input(
            "\nEnter Image Path (q to quit): "
        ).strip().strip('"').strip("'")

        image_path = os.path.normpath(image_path)

        if image_path.lower() == "q":

            print("\nGoodbye!\n")

            break

        if not os.path.exists(image_path):

            print("\nImage not found!\n")

            continue

        try:

            result = predict(image_path)
            show_result(image_path, result)
            show_image(image_path, result)

        except Exception as e:

            print("\nError :", e)


if __name__ == "__main__":

    if len(sys.argv) == 2:

        path = sys.argv[1]

        if not os.path.exists(path):

            print("\nImage not found.")

            sys.exit()

        result = predict(path)

        show_result(path, result)

    else:

        interactive()