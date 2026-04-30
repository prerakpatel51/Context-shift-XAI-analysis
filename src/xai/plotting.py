from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.io import ensure_dir


def denorm(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr, 0, 1)


def save_overlay(image_t, heatmap, path, title=None):
    ensure_dir(Path(path).parent)
    img = denorm(image_t)
    heatmap = np.asarray(heatmap, dtype=float)
    if heatmap.ndim == 2:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.imshow(heatmap, cmap="magma", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_heatmap(heatmap, path):
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="magma")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
