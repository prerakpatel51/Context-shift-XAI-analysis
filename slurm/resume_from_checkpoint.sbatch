#!/bin/bash
#SBATCH --job-name=nico_resume
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
python scripts/train.py --config configs/train_deit_from_checkpoint.yaml --data_root data/processed/nico --split_dir data/splits --output_dir outputs
