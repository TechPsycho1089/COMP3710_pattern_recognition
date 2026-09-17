#!/bin/bash
#SBATCH --job-name=oasis_vae_2gpu
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/oasis_vae_%j.out
#SBATCH --error=logs/oasis_vae_%j.err

# COMP3710 - Task 4.2: 2-GPU VAE Training on Rangpur HPC Cluster
# Author: Shekhar "Shakes" Chandra / COMP3710 Team

echo "=========================================================="
echo " Starting 2-GPU OASIS VAE Job on Rangpur Cluster"
echo " Node ID: $SLURMD_NODENAME"
echo " Partition: comp3710 | Account: comp3710"
echo " Dataset: /home/groups/comp3710/OASIS/"
echo " Start Time: $(date)"
echo "=========================================================="

# Create required directories
mkdir -p logs
mkdir -p models
mkdir -p results

# Activate miniconda3 torch environment on Rangpur
source $HOME/miniconda3/bin/activate
conda activate torch

# Verify GPU environment and PyTorch availability
echo "--- GPU Status ---"
nvidia-smi

echo "--- Python & PyTorch Test ---"
python -c "import torch; print('PyTorch Version:', torch.__version__); print('Available GPUs:', torch.cuda.device_count())"

# Execute Multi-GPU VAE Training targeting Rangpur OASIS dataset path
python train_vae.py \
    --data_dir "/home/groups/comp3710/OASIS/" \
    --epochs 30 \
    --batch_size 128 \
    --latent_dim 16

echo "=========================================================="
echo " Job Completed Successfully at $(date)"
echo "=========================================================="
