from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.attention_hooks import get_attention_grads, get_attentions, patch_timm_attention
from src.xai.plotting import save_overlay
from src.utils.io import ensure_dir


def tam_map(model, image, target, sample_id, out_dir):
    patch_timm_attention(model)
    model.zero_grad(set_to_none=True)
    logits = model(image.unsqueeze(0).to(next(model.parameters()).device))
    logits[0, int(target)].backward()
    attns, grads = get_attentions(model), get_attention_grads(model)
    result = None
    for a, g in zip(attns, grads):
        w = torch.relu(a[0] * g[0]).mean(0)
        eye = torch.eye(w.shape[-1])
        w = w + eye
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        result = w if result is None else w @ result
    grid = result[0, 1:]
    side = int(grid.numel() ** 0.5)
    heat = grid.reshape(side, side).detach().numpy()
    out = ensure_dir(Path(out_dir) / "tam")
    np.save(out / f"{sample_id}_tam.npy", heat)
    pd.DataFrame(heat).to_csv(out / f"{sample_id}_tam.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_tam_overlay.png")
    return heat
