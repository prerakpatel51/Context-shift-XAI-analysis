from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.xai.plotting import save_overlay
from src.xai.explanation_metrics import entropy
from src.utils.io import ensure_dir


def causal_patch_impact(model, image, label, sample_id, out_dir, patch_size=16, context_name=None):
    # Practical fallback: input-space causal patch ablation, clearly distinct from exact latent intervention.
    model.eval()
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    target = int(label)
    orig_logit, orig_prob = float(logits[0, target]), float(probs[0, target])
    _, _, H, W = x.shape
    heat = np.zeros((H // patch_size, W // patch_size), dtype=float)
    rows = []
    base = x.mean(dim=(2, 3), keepdim=True)
    for r in range(H // patch_size):
        for c in range(W // patch_size):
            xm = x.clone()
            y, x0 = r * patch_size, c * patch_size
            xm[:, :, y:y + patch_size, x0:x0 + patch_size] = base
            with torch.no_grad():
                ml = model(xm)
                mp = F.softmax(ml, dim=1)
                kl = F.kl_div(mp.log(), probs, reduction="batchmean")
            heat[r, c] = orig_prob - float(mp[0, target])
            rows.append({"sample_id": sample_id, "patch_row": r, "patch_col": c, "patch_logit_drop": orig_logit - float(ml[0, target]),
                         "patch_prob_drop": heat[r, c], "patch_kl_divergence": float(kl), "context_name": context_name})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "causal_impact_entropy": entropy(heat)})
    out = ensure_dir(Path(out_dir) / "causal_patch_impact")
    np.save(out / f"{sample_id}_causal_impact.npy", heat)
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_causal_impact.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_causal_impact_overlay.png")
    return heat
