# Task 4.2 / 4.4: Multi-GPU Variational Autoencoder (VAE) on OASIS Brain MRI

**Author:** Shekhar "Shakes" Chandra / COMP3710 Team  
**Dataset Base Path:** `/home/groups/comp3710/OASIS/` (Rangpur HPC Cluster)  
**Target Architecture:** Multi-GPU PyTorch VAE ($d=16$) with `nn.DataParallel` across 2 GPUs (`partition=comp3710`)

---

## 📌 Task Overview & Theory

This module implements **Task 1 of Section 4.4 (Variational Autoencoder - 3 Marks)**:

1. **Dataset Split Strategy:**
   * **Train Set:** `/home/groups/comp3710/OASIS/keras_png_slices_train`
   * **Validation Set:** `/home/groups/comp3710/OASIS/keras_png_slices_validate` (Evaluated per epoch to track `val_loss` and save the best checkpoint `models/oasis_vae_best.pt`).
   * **Test Set:** `/home/groups/comp3710/OASIS/keras_png_slices_test` (Evaluated for final visual reconstructions & UMAP manifold plotting).

2. **Reparameterization Trick ($d=16$):**  
   The latent bottleneck uses $d=16$ dimensions:
   $$z = \mathbf{\mu} + \mathbf{\sigma} \odot \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

3. **ELBO Loss Function (Evidence Lower Bound):**  
   $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{BCE}}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q(z|x) \parallel p(z))$$

4. **Multi-GPU Acceleration:**  
   Parallelized across **2 NVIDIA GPUs** using PyTorch `nn.DataParallel(model)` with CUDA streams.

5. **UMAP Latent Manifold Projection:**  
   Extracts 16D latent vectors $z$ from held-out test slices and maps them to a 2D continuous space using **UMAP**.

---

## 📁 Repository Structure

```text
OASIS_VAE/
├── train_vae.py         # PyTorch 2-GPU VAE script with train/val/test loaders & UMAP projection
├── job_vae.sh           # Slurm submission script requesting 2 GPUs on comp3710 partition
├── README.md            # Technical documentation & execution guide
├── models/              # Best saved checkpoint (oasis_vae_best.pt)
└── results/             # Output plots (oasis_vae_reconstructions.png, oasis_vae_umap.png)
```

---

## 🚀 Running on Rangpur HPC Cluster

1. **Submit Slurm Job:**
   ```bash
   cd OASIS_VAE
   sbatch job_vae.sh
   ```

2. **Check Status:**
   ```bash
   squeue -u $USER
   ```

3. **Stream Output Logs:**
   ```bash
   tail -f logs/oasis_vae_*.out
   ```
