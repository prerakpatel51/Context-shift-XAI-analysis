#!/bin/bash
#SBATCH --job-name=nico_full_xai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu2
#SBATCH --time=24:00:00

set -euo pipefail

cd /home1/ppatel2025/xai_project_4/nico-vit-xai

export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export XDG_CACHE_HOME=/tmp/${USER}-cache
export HF_HOME=/home1/ppatel2025/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home1/ppatel2025/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "[$(date)] Cleaning previous outputs and split files"
rm -rf outputs
rm -f data/splits/train.csv data/splits/val.csv data/splits/test.csv

echo "[$(date)] Creating output directories"
bash scripts/setup_project.sh

echo "[$(date)] Checking cached CLIP model"
python - <<'PY'
from transformers import CLIPModel, CLIPProcessor
model_path = "/home1/ppatel2025/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIPModel.from_pretrained(model_path, local_files_only=True)
CLIPProcessor.from_pretrained(model_path, local_files_only=True)
print(f"CLIP cache ready: {model_path}")
PY

echo "[$(date)] Inspecting dataset"
python scripts/inspect_dataset.py --data_root data/processed/nico

echo "[$(date)] Preparing NICO++ splits"
python scripts/prepare_nico.py --config configs/dataset.yaml --data_root data/processed/nico --split_dir data/splits

echo "[$(date)] Training pretrained DeiT"
python scripts/train.py --config configs/train_deit_pretrained.yaml --data_root data/processed/nico --split_dir data/splits --run_name deit_pretrained --output_dir outputs

echo "[$(date)] Evaluating best checkpoint"
python scripts/evaluate.py --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --config configs/train_deit_pretrained.yaml --data_root data/processed/nico --split_dir data/splits --split test --output_dir outputs/metrics/deit_pretrained

echo "[$(date)] Plotting training metrics"
python scripts/plot_training_metrics.py --metrics_csv outputs/logs/deit_pretrained/metrics_epoch.csv --output_dir outputs/figures/training

echo "[$(date)] Selecting XAI samples"
python scripts/select_xai_samples.py --checkpoints outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --names pretrained_best --data_root data/processed/nico --split_dir data/splits --output_csv outputs/metrics/xai/xai_selected_samples.csv --num_samples 200

echo "[$(date)] Running XAI methods"
python scripts/run_xai.py --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --checkpoint_name pretrained_best --config configs/xai.yaml --selected_samples outputs/metrics/xai/xai_selected_samples.csv --data_root data/processed/nico --split_dir data/splits --output_dir outputs/explanations/pretrained_best

echo "[$(date)] Plotting XAI summaries"
python scripts/plot_xai_results.py --explanation_dirs outputs/explanations/pretrained_best --names pretrained_best --selected_samples outputs/metrics/xai/xai_selected_samples.csv --output_dir outputs/figures/xai

echo "[$(date)] Full pretrained XAI pipeline complete"
