from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.io import ensure_dir
from src.xai.plotting import save_overlay


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def token_transformation_attribution(model, image, sample_id, out_dir):
    """Patch-token attribution from L2 representation changes across blocks."""
    device = next(model.parameters()).device
    records = []
    handles = []

    def hook(layer_idx):
        def _hook(_module, inputs, output):
            x_in = inputs[0].detach()
            x_out = output.detach()
            in_patch = x_in[:, 1:, :]
            out_patch = x_out[:, 1:, :]
            effect = (out_patch - in_patch).norm(dim=-1)
            records.append({
                "layer": layer_idx,
                "effect": effect.cpu(),
            })
        return _hook

    for idx, block in enumerate(getattr(model, "blocks", [])):
        handles.append(block.register_forward_hook(hook(idx)))

    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))

    for h in handles:
        h.remove()

    if not records:
        raise RuntimeError("No ViT block activations were captured for token transformation attribution.")

    effects = torch.stack([r["effect"][0] for r in records])
    heat_1d = _normalize(effects.sum(dim=0)).numpy()
    side = int(heat_1d.size ** 0.5)
    heat = heat_1d.reshape(side, side)

    rows = []
    for r in records:
        for patch_idx, value in enumerate(r["effect"][0].numpy()):
            rows.append({
                "sample_id": sample_id,
                "layer": r["layer"],
                "patch_idx": patch_idx,
                "token_transformation_effect": float(value),
            })

    out = ensure_dir(Path(out_dir) / "token_transformation")
    np.save(out / f"{sample_id}_token_transformation.npy", heat)
    pd.DataFrame(heat).to_csv(out / f"{sample_id}_token_transformation_grid.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_token_transformation_layers.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_token_transformation_overlay.png")
    return heat
