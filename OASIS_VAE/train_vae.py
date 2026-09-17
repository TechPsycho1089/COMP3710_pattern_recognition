"""
COMP3710 Pattern Recognition - Task 4.2 / 4.4: Multi-GPU Variational Autoencoder (VAE)
Target HPC Cluster: Rangpur GPU Cluster (The University of Queensland)
Dataset Base Path: /home/groups/comp3710/OASIS/
Subfolders:
  - Train:      /home/groups/comp3710/OASIS/keras_png_slices_train
  - Validation: /home/groups/comp3710/OASIS/keras_png_slices_validate
  - Test:       /home/groups/comp3710/OASIS/keras_png_slices_test

Author: Shekhar "Shakes" Chandra / COMP3710 Team

Multi-GPU Features & Architecture:
- PyTorch VAE with Latent Dimension d = 16.
- Multi-GPU Parallelization via PyTorch nn.DataParallel across 2 GPUs (comp3710 partition).
- Dedicated Train, Validation (per-epoch monitoring & best checkpoint saving), and Test loaders.
- Reparameterization Trick & ELBO Loss (BCE Reconstruction + KL Divergence).
- UMAP 2D Manifold Projection & Reconstruction Visualization on held-out test set.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import umap

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# 1. OASIS BRAIN MRI DATASET CLASS
# =====================================================================

class OASISSliceDataset(Dataset):
    """
    PyTorch Dataset for Preprocessed OASIS Brain MRI image slices.
    Targeting Rangpur Cluster folders:
      - keras_png_slices_train
      - keras_png_slices_validate
      - keras_png_slices_test
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []

        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.nii')):
                    self.image_paths.append(os.path.join(folder_path, f))
            print(f"Loaded {len(self.image_paths)} slices from '{folder_path}'.")
        else:
            print(f"WARNING: Directory '{folder_path}' not found locally. Using synthetic fallback for local code verification.")

    def __len__(self):
        if len(self.image_paths) == 0:
            return 500  # Fallback length for local testing
        return len(self.image_paths)

    def __getitem__(self, idx):
        if len(self.image_paths) > 0:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("L")  # Grayscale 1-channel
            if self.transform:
                image = self.transform(image)
            return image
        else:
            # Fallback synthetic brain slice for local verification
            np.random.seed(idx)
            img = np.zeros((64, 64), dtype=np.float32)
            y, x = np.ogrid[:64, :64]
            dist = (((x - 32)/20)**2 + ((y - 32)/25)**2) <= 1.0
            img[dist] = 0.7 + np.random.uniform(-0.1, 0.1)
            tensor_img = torch.from_numpy(img).unsqueeze(0)
            return tensor_img


# =====================================================================
# 2. VAE ARCHITECTURE (LATENT DIMENSION d = 16)
# =====================================================================

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Latent Space projections (d=16)
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder projections
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)

        # Decoder: 128x8x8 -> 64x16x16 -> 32x32x32 -> 1x64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization Trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 128, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# =====================================================================
# 3. ELBO LOSS FUNCTION
# =====================================================================

def vae_loss_function(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss, recon_loss, kld_loss


# =====================================================================
# 4. MULTI-GPU TRAINING & VALIDATION PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU OASIS VAE Training with Validation Set")
    parser.add_argument("--base_dir", type=str, default="/home/groups/comp3710/OASIS/", help="Base path to OASIS on Rangpur")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Global batch size across GPUs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimensionality of latent space z")
    args = parser.parse_args()

    # Image Transform Pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("==================================================================")
    print("      COMP3710 Multi-GPU VAE Training & Validation - Rangpur     ")
    print("==================================================================")

    # Set up paths for train, validate, and test folders
    train_path = os.path.join(args.base_dir, "keras_png_slices_train")
    val_path   = os.path.join(args.base_dir, "keras_png_slices_validate")
    test_path  = os.path.join(args.base_dir, "keras_png_slices_test")

    train_dataset = OASISSliceDataset(folder_path=train_path, transform=transform)
    val_dataset   = OASISSliceDataset(folder_path=val_path, transform=transform)
    test_dataset  = OASISSliceDataset(folder_path=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

    # Device & Multi-GPU Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    print(f"Target Compute Device: {device}")
    print(f"Available GPUs Detected: {num_gpus}")

    vae_model = VAE(latent_dim=args.latent_dim).to(device)

    # Multi-GPU DataParallel wrapping
    if num_gpus > 1:
        print(f"Enabling PyTorch nn.DataParallel across {num_gpus} GPUs!")
        vae_model = nn.DataParallel(vae_model)

    optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_val_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs (Batch Size: {args.batch_size}, Latent Dim: {args.latent_dim})...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- TRAIN PHASE ---
        vae_model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kld = 0.0

        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(data)

            loss, recon, kld = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon.item()
            train_kld += kld.item()

        avg_train_loss = train_loss / len(train_dataset)

        # --- VALIDATION PHASE ---
        vae_model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kld = 0.0

        with torch.no_grad():
            for data in val_loader:
                if isinstance(data, list):
                    data = data[0]
                data = data.to(device)

                recon_batch, mu, logvar = vae_model(data)
                loss, recon, kld = vae_loss_function(recon_batch, data, mu, logvar)

                val_loss += loss.item()
                val_recon += recon.item()
                val_kld += kld.item()

        avg_val_loss = val_loss / len(val_dataset)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | Val BCE: {val_recon/len(val_dataset):.2f} | Val KLD: {val_kld/len(val_dataset):.2f}")

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            raw_model = vae_model.module if isinstance(vae_model, nn.DataParallel) else vae_model
            torch.save(raw_model.state_dict(), "models/oasis_vae_best.pt")
            print(f"  --> Saved new best checkpoint to 'models/oasis_vae_best.pt' (Val Loss: {avg_val_loss:.2f})")

    total_duration = time.time() - start_time
    print(f"\nTraining & Validation completed in {total_duration:.2f} seconds across {num_gpus} GPU(s)!")

    # --- TEST SET EVALUATION & RECONSTRUCTION ARTIFACT ---
    print("\n--- Generating Reconstruction Plot on Held-Out Test Set ---")
    raw_model = vae_model.module if isinstance(vae_model, nn.DataParallel) else vae_model
    raw_model.load_state_dict(torch.load("models/oasis_vae_best.pt"))
    raw_model.eval()

    with torch.no_grad():
        test_batch = next(iter(test_loader))
        if isinstance(test_batch, list):
            test_batch = test_batch[0]
        test_batch = test_batch[:8].to(device)
        recon_test, _, _ = raw_model(test_batch)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(test_batch[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title("Test Target")
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_test[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title("VAE Recon")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("results/oasis_vae_reconstructions.png")
    plt.close()
    print("Saved 'results/oasis_vae_reconstructions.png'")

    # --- UMAP MANIFOLD VISUALIZATION ON TEST SET ---
    print("\nExtracting Test Set Latent Representations for UMAP 2D Manifold...")
    all_mus = []
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            mu, _ = raw_model.encode(data)
            all_mus.append(mu.cpu().numpy())

    all_mus = np.concatenate(all_mus, axis=0)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(all_mus)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='purple', alpha=0.6, edgecolors='none', s=20)
    plt.title("UMAP 2D Projection of OASIS Test Set Latent Manifold (d=16 VAE)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/oasis_vae_umap.png")
    plt.close()
    print("Saved 'results/oasis_vae_umap.png'")
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
