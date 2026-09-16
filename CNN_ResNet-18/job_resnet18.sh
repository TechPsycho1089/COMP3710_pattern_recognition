#!/bin/bash
#SBATCH --job-name=resnet18-cifar10
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=resnet18_%j.out
#SBATCH --error=resnet18_%j.err

# ------------------------------------------------------------
# COMP3710 DAWNBench Challenge - ResNet-18 on CIFAR-10
# ------------------------------------------------------------

# Navigate to project directory on Rangpur
cd $HOME/COMP3710_pattern_recognition/CNN_ResNet-18

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate keras

# Print GPU information
nvidia-smi

# Execute the ResNet-18 training script
python resnet18_cifar10.py
