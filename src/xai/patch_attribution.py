from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.xai.explanation_metrics import entropy, topk_concentration
from src.xai.plotting import save_overlay
from src.utils.io import ensure_dir


def gradient_x_input(model, image, label, sample_id, out_dir, patch_size=16, output_mode="both"):
    """Grad x input attribution.

    output_mode:
      - 'patch': average pixel attribution into patch grid (legacy).
      - 'pixel': keep full HxW pixel-level map.
      - 'both' (default): save both, return patch map.
    """
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device).requires_grad_(True)
    logit = model(x)[0, int(label)]
    model.zero_grad(set_to_none=True)
    logit.backward()
    pix = (x.grad * x).abs().sum(1)[0].detach().cpu().numpy()
    H, W = pix.shape
    grid = pix.reshape(H // patch_size, patch_size, W // patch_size, patch_size).mean((1, 3))
    out = ensure_dir(Path(out_dir) / "patch_attribution")
    rows = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            rows.append({"sample_id": sample_id, "patch_row": r, "patch_col": c, "importance": float(grid[r, c])})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "importance": float(entropy(grid)), "metric": "entropy"})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "importance": float(topk_concentration(grid)), "metric": "topk_concentration"})
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_patch_importance.csv", index=False)

    if output_mode in ("patch", "both"):
        save_overlay(image, grid, out / f"{sample_id}_patch_heatmap.png")
        np.save(out / f"{sample_id}_patch_grid.npy", grid)
    if output_mode in ("pixel", "both"):
        save_overlay(image, pix, out / f"{sample_id}_pixel_heatmap.png")
        np.save(out / f"{sample_id}_pixel_grid.npy", pix)

    return pix if output_mode == "pixel" else grid
