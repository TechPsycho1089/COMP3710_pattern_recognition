#!/bin/bash
#SBATCH --job-name=infer-resnet18
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=infer_resnet18_%j.out
#SBATCH --error=infer_resnet18_%j.err

# ------------------------------------------------------------
# COMP3710 DAWNBench Challenge - ResNet-18 Inference Job
# ------------------------------------------------------------

# Force instant unbuffered Python print log streaming
export PYTHONUNBUFFERED=1

# Navigate to project directory on Rangpur
cd $HOME/COMP3710_pattern_recognition/CNN_ResNet-18

# Activate conda environment with PyTorch installed
source $HOME/miniconda3/bin/activate
conda activate pytorch 2>/dev/null || conda activate keras 2>/dev/null || true

# Print GPU information
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# Execute PyTorch inference script on full 10,000 unseen test set
echo "Inference job $SLURM_JOB_ID on $(hostname), started $(date)"
python inference_resnet18.py --eval
