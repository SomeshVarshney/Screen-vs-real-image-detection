# 🖥️ Screen vs Real Image Detection

A deep learning-based computer vision project that classifies whether an input image is:

- 📷 A Real Photograph
- 🖥️ A Photograph of a Digital Screen

The project uses **Transfer Learning** with **EfficientNet-B0** and a custom-built dataset to achieve high classification performance while maintaining fast inference suitable for real-time applications.

---

# Features

- EfficientNet-B0 Transfer Learning
- Two-stage Fine-Tuning
- Automatic Prediction
- Confidence Score
- ROC-AUC Evaluation
- Threshold Optimization
- Benchmarking Script
- GPU/CPU Support
- Interactive Prediction Tool

---

# Model

Backbone

- EfficientNet-B0 (ImageNet Pretrained)

Classifier

- Dropout
- Fully Connected Layer
- Softmax Output

Loss

- CrossEntropy Loss

Optimizer

- AdamW

Learning Rate Scheduler

- Cosine Annealing LR

---

# Dataset

Custom Dataset

Classes

- Real Images
- Screen Images

Dataset Statistics

| Class  | Images |
| ------ | ------ |
| Real   | 244    |
| Screen | 209    |
| Total  | 453    |

Images were collected under various lighting conditions, camera angles, distances, reflections, and display devices to improve generalization.

---

# Training Strategy

### Stage 1

- Freeze EfficientNet backbone
- Train classifier head

### Stage 2

- Unfreeze backbone
- Fine-tune complete network

---

# Evaluation

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 89.01% |
| Precision | 97.06% |
| Recall    | 87.57% |
| F1 Score  | 88.84% |
| ROC-AUC   | 95.80% |

---

# Benchmark

Average GPU inference time

8–10 ms

Throughput

100+ FPS (hardware dependent)

Framework

PyTorch

---

# Project Structure

```text
spot_fake/

├── checkpoints/
│   └── best_model.pth

├── src/
│   ├── benchmark.py
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py

├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

```bash
git clone https://github.com/YOUR_USERNAME/screen-vs-real-image-detection.git

cd screen-vs-real-image-detection

pip install -r requirements.txt
```

---

# Usage

Train

```bash
python src/train.py
```

Evaluate

```bash
python src/evaluate.py
```

Predict

```bash
python src/predict.py
```

Benchmark

```bash
python src/benchmark.py
```

---

# Example Prediction

```
Prediction : SCREEN IMAGE

Confidence : 97.12 %

Inference Time : 8.31 ms
```

---

# Future Improvements

- Larger and more diverse dataset
- ConvNeXt / EfficientNet-B3 backbone
- ONNX export
- TensorRT acceleration
- Mobile deployment
- Web application using Streamlit or FastAPI

---

# Tech Stack

- Python
- PyTorch
- TorchVision
- Albumentations
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

---
# Screenshots
![alt text](<images (1).jpeg>)
![alt text](<WhatsApp Image 2026-07-02 at 3.37.32 PM.jpeg>)


# Author

**Somesh Varshney**
