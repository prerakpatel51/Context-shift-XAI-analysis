#!/bin/bash
#SBATCH --job-name=xai_tokctx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu2
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export XDG_CACHE_HOME=/tmp/${USER}-cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python scripts/run_xai.py --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --checkpoint_name pretrained_best --config configs/xai.yaml --selected_samples outputs/metrics/xai/xai_selected_samples.csv --data_root data/processed/nico --split_dir data/splits --output_dir outputs/explanations/pretrained_best --methods token_transformation context_swap
