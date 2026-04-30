#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/checkpoints/deit_pretrained
mkdir -p outputs/logs/deit_pretrained
mkdir -p outputs/metrics/{deit_pretrained,xai}
mkdir -p outputs/figures/{training,xai}
mkdir -p data/{raw,processed,splits}
echo "Project directories are ready."
