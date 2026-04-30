#!/bin/bash
#SBATCH --job-name=xai_plots
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export XDG_CACHE_HOME=/tmp/${USER}-cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python scripts/plot_xai_results.py --explanation_dirs outputs/explanations/pretrained_best --names pretrained_best --selected_samples outputs/metrics/xai/xai_selected_samples.csv --output_dir outputs/figures/xai
