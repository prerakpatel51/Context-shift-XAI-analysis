#!/bin/bash
#SBATCH --job-name=nico_pretrained
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu2
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
python scripts/train.py --config configs/train_deit_pretrained.yaml --data_root data/processed/nico --split_dir data/splits --run_name deit_pretrained --output_dir outputs
