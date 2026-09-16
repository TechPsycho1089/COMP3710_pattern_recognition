# COMP3710 Task 3.1 - LFW Facial Pattern Recognition CNN

This repository contains a Convolutional Neural Network (CNN) classifier built using TensorFlow / Keras for facial pattern recognition on the Labeled Faces in the Wild (LFW) dataset.

---

## Directory Structure

```text
CNN/
├── lfw-deepfunneled/     # LFW dataset images across 7 identity classes
├── models/                # Saved best model checkpoint (best_model.keras) & class_names.json
├── logs/                  # Training history CSV (training.csv)
├── train.py               # Dataset processing, model training, & evaluation
├── inference.py           # Command-line inference script for test images
└── README.md              # Project documentation
```

---

## Model Architecture & Design

The network is designed under strict task constraints (exactly 2 Convolutional layers of 32 filters with $3\times3$ kernels, and 1 Dense output layer), while incorporating `MaxPooling2D` and `Dropout` to prevent parameter explosion and overfitting:

```text
Input (250x250x3 RGB) 
  │
  ├──► Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
  │
  ├──► Conv2D (32 filters, 3x3, ReLU)  ──► MaxPooling2D (2x2)
  │
  ├──► Conv2D (32 filters, 3x3, ReLU)  ──► MaxPooling2D (2x2)
  │
  ├──► Flatten (61x61x32 = 119,072 features)
  │
  ├──► Dropout (rate = 0.5)
  │
  └──► Dense (7 classes, Softmax)
```

### Parameter Efficiency
* **Original unregularized dense parameters**: ~13.55 Million
* **Optimized parameters with Max Pooling**: **833,511** (~94% reduction)

---

## Dataset Classes (7 Identities)

1. `Ariel_Sharon`
2. `Colin_Powell`
3. `Donald_Rumsfeld`
4. `George_W_Bush`
5. `Gerhard_Schroeder`
6. `Hugo_Chavez`
7. `Tony_Blair`

---

## Performance Benchmark

* **Validation Accuracy**: **~94.2%**
* **Test Accuracy**: **~92.6%**
* **Early Stopping**: Triggered automatically when validation loss reached global minimum (`val_loss = 0.2180`), saving optimal weights to `models/best_model.keras`.

---

## How to Run

### 1. Training the Model
```bash
python train.py
```

### 2. Running Inference on Test Images
```bash
# Predict a specific image file
python inference.py lfw-deepfunneled/Ariel_Sharon/Ariel_Sharon_0001.jpg
```

**Example Output**:
```text
Prediction
----------
Image     : lfw-deepfunneled/Ariel_Sharon/Ariel_Sharon_0001.jpg
Class     : Ariel_Sharon
Confidence: 0.9970

Probabilities:
Ariel_Sharon        : 0.9970
Colin_Powell        : 0.0000
Donald_Rumsfeld     : 0.0001
George_W_Bush       : 0.0029
Gerhard_Schroeder   : 0.0000
Hugo_Chavez         : 0.0000
Tony_Blair          : 0.0000
```
