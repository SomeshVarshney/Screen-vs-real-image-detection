from pathlib import Path

EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def load_dataset(root):
    image_paths = []
    labels = []

    real = Path(root) / "real"
    screen = Path(root) / "screen"

    for ext in EXTENSIONS:
        for img in real.glob(f"*{ext}"):
            image_paths.append(str(img))
            labels.append(0)

    for ext in EXTENSIONS:
        for img in screen.glob(f"*{ext}"):
            image_paths.append(str(img))
            labels.append(1)

    return image_paths, labels