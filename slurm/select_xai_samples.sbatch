#!/bin/bash
#SBATCH --job-name=xai_select
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu2
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
python scripts/select_xai_samples.py --checkpoints outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --names pretrained_best --data_root data/processed/nico --split_dir data/splits --output_csv outputs/metrics/xai/xai_selected_samples.csv --num_samples 200
