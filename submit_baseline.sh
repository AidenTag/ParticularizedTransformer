#!/bin/bash
#SBATCH --job-name=base_listops
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/baseline_%j.out
#SBATCH --error=slurm_logs/baseline_%j.err

set -e

# 1. Environment Setup
source /home/ataghinia27/ParticularizedTransformer/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"

# Create logs directory
mkdir -p slurm_logs

# 2. Run Training
# Running ListOps Baseline (Standard Attention)
python -m src.train \
    --data_dir data/listops \
    --exp_name "baseline_run_$SLURM_JOB_ID" \
    --disable_dalex \
    --max_epochs 50 \
    --batch_size 128 \
    --n_layer 4 \
    --n_head 8 \
    --n_embd 128 \
    --lr 5e-4 \
    --max_length 2048 \
    --gpus 1

echo "End Time: $(date)"
