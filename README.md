# COMP3710 Pattern Recognition - Demonstration & Recognition Portfolio

**Author:** Harshit Aggarwal  
**Course staff:** Shekhar "Shakes" Chandra (course coordinator, task specifications)  
**Course:** COMP3710 - Pattern Analysis (The University of Queensland)  
**Frameworks:** PyTorch, Scikit-Learn, NumPy, Matplotlib, UMAP

---

## Repository Architecture

This repository contains my implementations of Parts 3 and 4 of **Lab Demonstration 2**:

```text
COMP3710_pattern_recognition/
├── CNN/                           # Part 3.1: 2-Layer CNN on LFW Face Dataset
│   ├── train.py                   # PyTorch CNN training pipeline
│   ├── inference.py               # Evaluation & test inference script
│   └── README.md                  # CNN architecture documentation
├── CNN_ResNet-18/                 # Part 3.2: Fast ResNet-18 DAWNBench Challenge
│   ├── resnet18_cifar10.py        # Fast CIFAR-10 classification with Mixed Precision
│   ├── inference_resnet18.py      # Rangpur cluster evaluation script
│   ├── job_resnet18.sh            # SLURM execution script for Rangpur GPU cluster
│   └── README.md                  # DAWNBench performance & benchmarking guide
└── OASIS_VAE/                     # Part 4: Variational Autoencoder on OASIS Brain MRI
    ├── train_vae.py               # VAE architecture, ELBO loss, & manifold visualization
    └── README.md                  # VAE latent space & UMAP manifold analysis
```

---

## Summary of Parts & Achievements

| Section | Topic | Key Implementation | Metrics & Highlights |
| :--- | :--- | :--- | :--- |
| **Part 3.1**| **CNN Classifier** | 2-Layer Conv2D ($3\times3$, 32 filters) with Adam & Cross-Entropy on LFW | End-to-end representation learning on the LFW face dataset. |
| **Part 3.2**| **DAWNBench ResNet-18** | Fast CIFAR-10 training with ResNet-18, PyTorch AMP (Mixed Precision) | **>94% test accuracy** on CIFAR-10, trained on the Rangpur HPC GPU cluster. |
| **Part 4** | **OASIS Brain VAE & Git** | PyTorch VAE, Reparameterization Trick, ELBO Loss, UMAP 2D Manifold Plot | Decoded continuous 2D manifold of brain MRI slices. Clean Git commit workflow. |

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TechPsycho1089/COMP3710_pattern_recognition.git
   cd COMP3710_pattern_recognition
   ```

2. **Set up Environment & Dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install torch torchvision numpy matplotlib scikit-learn scipy umap-learn
   ```

3. **Download the LFW dataset** (not stored in this repo): download `lfw-deepfunneled` from the [LFW website](http://vis-www.cs.umass.edu/lfw/) and extract it into `CNN/lfw-deepfunneled/`. CIFAR-10 downloads automatically through `torchvision`; the OASIS data lives on the Rangpur cluster.

4. **Running Tasks:**
   * **Part 3.1 CNN:** `python CNN/train.py`
   * **Part 3.2 ResNet-18:** `sbatch CNN_ResNet-18/job_resnet18.sh` (Rangpur)
   * **Part 4 VAE:** `python OASIS_VAE/train_vae.py`
