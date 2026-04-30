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
    """TokenTM-style attribution from per-block token representation changes.

    This is a lightweight implementation inspired by TokenTM's main idea:
    attention alone misses the effect of token transformations. For each ViT
    block, we score every patch token by both representation length change and
    direction change from block input to block output, then aggregate scores
    across layers.
    """
    device = next(model.parameters()).device
    records = []
    handles = []

    def hook(layer_idx):
        def _hook(_module, inputs, output):
            x_in = inputs[0].detach()
            x_out = output.detach()
            in_patch = x_in[:, 1:, :]
            out_patch = x_out[:, 1:, :]
            length_change = (out_patch.norm(dim=-1) - in_patch.norm(dim=-1)).abs()
            direction_change = 1.0 - torch.nn.functional.cosine_similarity(in_patch, out_patch, dim=-1)
            effect = _normalize(length_change[0]) * _normalize(direction_change[0])
            records.append({
                "layer": layer_idx,
                "length_change": length_change[0].cpu(),
                "direction_change": direction_change[0].cpu(),
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

    effects = torch.stack([r["effect"] for r in records])
    heat_1d = _normalize(effects.mean(dim=0)).numpy()
    side = int(heat_1d.size ** 0.5)
    heat = heat_1d.reshape(side, side)

    rows = []
    for r in records:
        for patch_idx, value in enumerate(r["effect"].numpy()):
            rows.append({
                "sample_id": sample_id,
                "layer": r["layer"],
                "patch_idx": patch_idx,
                "length_change": float(r["length_change"][patch_idx]),
                "direction_change": float(r["direction_change"][patch_idx]),
                "token_transformation_effect": float(value),
            })

    out = ensure_dir(Path(out_dir) / "token_transformation")
    np.save(out / f"{sample_id}_token_transformation.npy", heat)
    pd.DataFrame(heat).to_csv(out / f"{sample_id}_token_transformation_grid.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_token_transformation_layers.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_token_transformation_overlay.png")
    return heat
