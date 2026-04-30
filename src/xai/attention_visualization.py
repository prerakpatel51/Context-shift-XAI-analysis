from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.attention_hooks import get_attentions, patch_timm_attention
from src.xai.plotting import save_overlay
from src.utils.io import ensure_dir


def attention_visualization(model, image, sample_id, out_dir):
    patch_timm_attention(model)
    model.eval()
    model(image.unsqueeze(0).to(next(model.parameters()).device))
    attns = get_attentions(model)
    out = ensure_dir(Path(out_dir) / "attention_weights")
    rows = []
    for li, a in enumerate(attns):
        arr = a[0].numpy()
        for h in range(arr.shape[0]):
            p = arr[h].ravel()
            rows.append({"sample_id": sample_id, "layer": li, "head": h, "attention_entropy": float(-(p * np.log(p + 1e-12)).sum())})
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_attention_entropy.csv", index=False)
    last = attns[-1][0].mean(0)[0, 1:].numpy()
    side = int(last.size ** 0.5)
    save_overlay(image, last.reshape(side, side), out / f"{sample_id}_last_layer_cls_attention.png")
