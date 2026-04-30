#!/bin/bash
#SBATCH --job-name=eval_pretrained
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu2
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
python scripts/evaluate.py --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --config configs/train_deit_pretrained.yaml --data_root data/processed/nico --split_dir data/splits --split test --output_dir outputs/metrics/deit_pretrained
