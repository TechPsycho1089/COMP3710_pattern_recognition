# COMP3710 Pattern Recognition - Demonstration & Recognition Portfolio

**Author:** Shekhar "Shakes" Chandra / COMP3710 Student Portfolio  
**Course:** COMP3710 - Pattern Analysis (The University of Queensland)  
**Frameworks:** PyTorch, Scikit-Learn, NumPy, Matplotlib, UMAP

---

## Repository Architecture

This repository contains the complete implementation across all four parts of **Lab Demonstration 2**:

```text
COMP3710_pattern_recognition/
├── Part1_DFT/
│   ├── part1_dft.py               # Fourier square wave & PyTorch GPU/CPU matrix DFT
│   └── time_vector_demo.py        # In-depth time vector & endpoint=False demonstration
├── Part2_Eigenfaces/
│   └── part2_eigenfaces.py        # SVD PCA face space & Random Forest classifier
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
| **Part 1** | **Discrete Fourier Transform** | Fourier series square wave, PyTorch $O(N^2)$ Tensor Matrix DFT ($W \cdot x$) vs $O(N \log N)$ FFT | Numerical correctness verified (`np.allclose` = `True`). Benchmarked vector speedups across $N$. |
| **Part 2** | **Eigenfaces & PCA** | SVD Eigen-decomposition ($150$ components), Compactness plot, Random Forest | Captured **94.65%** variance while compressing dimensions by 92%. Classifier accuracy: **65.53%**. |
| **Part 3.1**| **CNN Classifier** | 2-Layer Conv2D ($3\times3$, 32 filters) with Adam & Cross-Entropy on LFW | End-to-end representation learning outperforming classical PCA + Random Forest. |
| **Part 3.2**| **DAWNBench ResNet-18** | Fast CIFAR-10 training with ResNet-18, PyTorch AMP (Mixed Precision) | **>90% accuracy** in under 30 minutes on Rangpur HPC cluster. |
| **Part 4** | **OASIS Brain VAE & Git** | PyTorch VAE, Reparameterization Trick, ELBO Loss, UMAP 2D Manifold Plot | Decoded continuous 2D manifold of brain MRI slices. Clean Git commit workflow. |

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/COMP3710_pattern_recognition.git
   cd COMP3710_pattern_recognition
   ```

2. **Set up Environment & Dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install torch torchvision numpy matplotlib scikit-learn scipy umap-learn
   ```

3. **Running Tasks:**
   * **Part 1 DFT:** `python Part1_DFT/part1_dft.py`
   * **Part 2 Eigenfaces:** `python Part2_Eigenfaces/part2_eigenfaces.py`
   * **Part 3 CNN:** `python CNN/train.py`
   * **Part 4 VAE:** `python OASIS_VAE/train_vae.py`
