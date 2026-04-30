from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.xai.plotting import save_overlay
from src.utils.io import ensure_dir


def occlusion_sensitivity(model, image, label, sample_id, out_dir, patch_size=16, stride=16, baseline="mean", context_name=None):
    model.eval()
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    pred = int(probs.argmax(1).item())
    target = int(label)
    orig_logit, orig_prob = float(logits[0, target]), float(probs[0, target])
    _, _, H, W = x.shape
    heat = np.zeros((H // patch_size, W // patch_size), dtype=float)
    rows = []
    base_value = 0.0 if baseline in {"zero", "gray"} else x.mean(dim=(2, 3), keepdim=True)
    for r, y in enumerate(range(0, H - patch_size + 1, stride)):
        for c, x0 in enumerate(range(0, W - patch_size + 1, stride)):
            xm = x.clone()
            xm[:, :, y:y + patch_size, x0:x0 + patch_size] = base_value
            with torch.no_grad():
                ml = model(xm)
                mp = F.softmax(ml, dim=1)
            drop_l = orig_logit - float(ml[0, target])
            drop_p = orig_prob - float(mp[0, target])
            heat[r, c] = drop_p
            rows.append({"sample_id": sample_id, "patch_row": r, "patch_col": c, "x1": x0, "y1": y, "x2": x0 + patch_size, "y2": y + patch_size,
                         "original_logit": orig_logit, "masked_logit": float(ml[0, target]), "logit_drop": drop_l,
                         "original_prob": orig_prob, "masked_prob": float(mp[0, target]), "prob_drop": drop_p,
                         "target_class": target, "true_class": target, "predicted_class": pred, "context_name": context_name})
    out = ensure_dir(Path(out_dir) / "occlusion")
    np.save(out / f"{sample_id}_occlusion.npy", heat)
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_occlusion.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_occlusion_overlay.png")
    return heat
