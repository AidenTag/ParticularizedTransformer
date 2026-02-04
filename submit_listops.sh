#!/bin/bash
#SBATCH --job-name=dalex_listops
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/dalex_%j.out
#SBATCH --error=slurm_logs/dalex_%j.err

set -e

# 1. Environment Setup
# Adjust the venv path to the local project venv
source /home/ataghinia27/ParticularizedTransformer/venv/bin/activate

# Optional: Load CUDA modules if not handled by the host environment
# module load cuda/12.1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"

# Create logs directory
mkdir -p slurm_logs

# 2. Run Training
# Running ListOps Mechanism Check (DALex Pressure 0.5)

# Note: We use -m src.train to run as a module from the project root
python -m src.train \
    --data_dir data/listops \
    --exp_name "dalex_p0.5_run_$SLURM_JOB_ID" \
    --dalex_pressure 0.5 \
    --max_epochs 50 \
    --batch_size 128 \
    --n_layer 4 \
    --n_head 8 \
    --n_embd 128 \
    --lr 5e-4 \
    --max_length 2048 \
    --gpus 1

# 3. (Optional) Run Baseline comparison immediately after
# python -m src.train \
#     --data_dir data/listops \
#     --disable_dalex \
#     --max_epochs 50 \
#     --batch_size 128 \
#     --n_layer 4 \
#     --n_head 8 \
#     --n_embd 128 \
#     --lr 5e-4

echo "End Time: $(date)"
