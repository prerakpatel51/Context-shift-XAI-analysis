#!/usr/bin/env bash
set -euo pipefail
python scripts/inspect_dataset.py --data_root data/processed/nico
python scripts/prepare_nico.py --config configs/dataset.yaml --data_root data/processed/nico --split_dir data/splits
python scripts/train.py --config configs/train_deit_pretrained.yaml --data_root data/processed/nico --split_dir data/splits --output_dir outputs
python scripts/evaluate.py --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --config configs/train_deit_pretrained.yaml --data_root data/processed/nico --split_dir data/splits --output_dir outputs/metrics/deit_pretrained
python scripts/select_xai_samples.py --checkpoints outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --names pretrained_best --data_root data/processed/nico --split_dir data/splits --output_csv outputs/metrics/xai/xai_selected_samples.csv
python scripts/run_xai.py --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt --checkpoint_name pretrained_best --config configs/xai.yaml --selected_samples outputs/metrics/xai/xai_selected_samples.csv --data_root data/processed/nico --split_dir data/splits --output_dir outputs/explanations/pretrained_best
python scripts/plot_training_metrics.py
python scripts/plot_xai_results.py --explanation_dirs outputs/explanations/pretrained_best --names pretrained_best --selected_samples outputs/metrics/xai/xai_selected_samples.csv --output_dir outputs/figures/xai
