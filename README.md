# Screen vs Real Image Detection

## Overview

This project detects whether an input image is:

- Real Photograph
- Photograph of a Screen

The model is trained using transfer learning with EfficientNet-B0 and fine-tuned on a custom dataset.

---

## Features

- EfficientNet-B0 Transfer Learning
- Two-Stage Fine Tuning
- Automatic Prediction
- Confidence Score
- Benchmarking
- Confusion Matrix
- Threshold Optimization

---

## Project Structure

```text
spot_fake/
│
├── data/
├── checkpoints/
├── benchmark/
├── predictions/
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── benchmark.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── model.py
│   ├── engine.py
│   ├── config.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

Classes:

- Real Images
- Screen Images

Dataset Size:

- Real: 244
- Screen: 209

Total Images:

453

---

## Training

Model: EfficientNet-B0

Optimizer: AdamW

Scheduler: CosineAnnealingLR

Loss: CrossEntropyLoss

Training Strategy:

Stage 1

- Train classifier head

Stage 2

- Fine tune backbone

---

## Evaluation Metrics

- Accuracy: 89.01%
- Precision: 97.06%
- Recall: 87.57%
- F1 Score: 88.84%
- ROC-AUC: 95.80%

---

## Running

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

## Example Prediction

```text
Prediction : SCREEN IMAGE

Confidence : 97.12 %

Inference : 8.31 ms
```

---

## Future Improvements

- Larger Dataset
- Mobile Deployment
- ONNX Export
- TensorRT Optimization
- Multi-class Classification

---

## Author

Somesh Varshney