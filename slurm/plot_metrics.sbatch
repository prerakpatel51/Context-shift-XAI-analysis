#!/bin/bash
#SBATCH --job-name=plot_metrics
cd /home1/ppatel2025/xai_project_4/nico-vit-xai
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export XDG_CACHE_HOME=/tmp/${USER}-cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python scripts/plot_training_metrics.py
