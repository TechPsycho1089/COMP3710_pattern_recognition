# COMP3710 Task 3.2 - DAWNBench Challenge (ResNet-18 on CIFAR-10)

This repository contains the ResNet-18 model implementation for CIFAR-10 classification optimized for fast GPU execution on the UQ Rangpur HPC cluster.

---

## Directory Structure

```text
CNN_ResNet-18/
├── data/                  # Auto-downloaded CIFAR-10 dataset
├── models/                # Saved best model checkpoint (best_resnet18.keras)
├── logs/                  # Training logs & timing metrics (training.csv, training_summary.json)
├── resnet18_cifar10.py    # Main model, timing, and dataset script
├── job_resnet18.sh        # Slurm batch execution script (5-minute limit)
├── .gitignore             # Excludes large binaries and dataset from git
└── README.md              # Documentation
```

---

## Features & Optimizations

1. **Residual Connections**: Full ResNet-18 implementation with 8 basic residual blocks (`BasicBlock`) and $1 \times 1$ conv shortcut projections.
2. **CIFAR-10 Adaptation**: Modified stem ($3\times3$ conv, stride 1, no stem maxpooling) to preserve spatial information for $32\times32$ images.
3. **Automated Dataset Handling**: `resnet18_cifar10.py` automatically downloads and extracts the CIFAR-10 dataset into `./data/` if missing.
4. **Mixed Precision (`float16`)**: Leverages Nvidia A100 Tensor Cores for ~2.5x speedup and reduced VRAM.
5. **High-Precision Timing**: Custom `TimingCallback` records total training duration, per-epoch timing, and average epoch seconds into `logs/training_summary.json`.

---

## How to Run on Rangpur Cluster

1. **Clone your repository into your home directory**:
   ```bash
   cd $HOME/COMP3710_pattern_recognition
   git clone <your_github_repo_url> CNN_ResNet-18
   cd CNN_ResNet-18
   ```

2. **Submit the Slurm GPU Job**:
   ```bash
   sbatch job_resnet18.sh
   ```

3. **Check Job Status & View Results**:
   ```bash
   squeue -u $USER
   cat resnet18_*.out
   ```
