from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.attention_hooks import get_attentions, patch_timm_attention
from src.xai.plotting import save_overlay
from src.utils.io import ensure_dir


def rollout_from_attentions(attns):
    result = None
    for a in attns:
        a = a[0].mean(0)
        eye = torch.eye(a.shape[-1])
        a = (a + eye) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    grid = result[0, 1:]
    side = int(grid.numel() ** 0.5)
    return grid.reshape(side, side).numpy()


def attention_rollout(model, image, sample_id, out_dir):
    patch_timm_attention(model)
    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0).to(next(model.parameters()).device))
    heat = rollout_from_attentions(get_attentions(model))
    out = ensure_dir(Path(out_dir) / "attention_rollout")
    np.save(out / f"{sample_id}_rollout.npy", heat)
    pd.DataFrame(heat).to_csv(out / f"{sample_id}_rollout.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_rollout_overlay.png")
    return heat
