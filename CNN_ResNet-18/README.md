# COMP3710 Task 3.2 - DAWNBench Challenge (ResNet-18 on CIFAR-10)

This repository contains an optimized **ResNet-18** deep learning implementation for CIFAR-10 classification, built using **PyTorch** and tailored for fast GPU training on the UQ Rangpur HPC cluster.

---

## Directory Structure

```text
CNN_ResNet-18/
├── data/                  # Auto-downloaded CIFAR-10 dataset
├── models/                # Saved PyTorch model checkpoints (best_resnet18.pth, demo_resnet18.pth)
├── logs/                  # Training history & timing logs (training_log.csv, timing_summary.json)
├── resnet18_cifar10.py    # Main training & timing evaluation script (PyTorch)
├── inference_resnet18.py  # Model evaluation & single-image inference script (PyTorch)
├── job_resnet18.sh        # Slurm batch execution script (5-minute limit)
├── .gitignore             # Excludes large dataset/models from git commits
└── README.md              # Project documentation
```

---

## Features & Optimizations

1. **PyTorch ResNet-18 Residual Connections**: Full ResNet-18 architecture with 8 basic residual blocks (`BasicBlock`) and $1 \times 1$ conv projection shortcuts.
2. **CIFAR-10 Stem Adaptation**: Modified input stem ($3\times3$ conv, stride 1, padding 1, no initial maxpooling) to preserve spatial features for $32\times32$ images.
3. **Dataset Segregation (Zero Data Leakage)**:
   * **Train Set**: 45,000 images
   * **Validation Set**: 5,000 images (monitored by validation loop during training)
   * **Held-out Test Set**: 10,000 images (strictly unseen data for final evaluation)
4. **PyTorch AMP (`torch.amp`) & GPU Cutout**: Leverages PyTorch automatic mixed precision and tensor-native Cutout ($16 \times 16$ masking) for high speed and accuracy.
5. **Execution Timing Tracking**: High-precision `time.perf_counter()` records total training duration and per-epoch timing into `logs/timing_summary.json`.

---

## Execution Guide

### 1. Full Offline GPU Training (Rangpur)
Submit the Slurm GPU batch job on Rangpur to train the full 30 epochs and achieve **>94% test accuracy**:
```bash
sbatch job_resnet18.sh
```

---

### 2. Live Demonstration Instructions (Tutor Marking)

#### A. Demonstrate 1-Epoch Training on Rangpur
To satisfy **Requirement 2 (1 Mark)** without overwriting your 94%+ pre-trained model:
```bash
python resnet18_cifar10.py --demo
```
* Runs **1 single epoch**.
* Saves checkpoint to `models/demo_resnet18.pth`.
* **Protects `models/best_resnet18.pth` from being overwritten!**

#### B. Demonstrate 94%+ Accuracy on Unseen Test Data
Run instant evaluation across all 10,000 held-out test images with Test-Time Augmentation (TTA):
```bash
python inference_resnet18.py --eval
```

**Output**:
```text
==================================================
DEMO TEST EVALUATION (10,000 UNSEEN TEST IMAGES)
==================================================
Single-Pass Test Accuracy: 93.68%
TTA (Flip) Test Accuracy : 94.24%
RESULT: Target accuracy >= 94% met on unseen test dataset!
==================================================
```

#### C. Predict a Single Image (Random or Specified)
```bash
# Predict a random test image
python inference_resnet18.py

# Predict a specific image file
python inference_resnet18.py path/to/image.jpg
```
