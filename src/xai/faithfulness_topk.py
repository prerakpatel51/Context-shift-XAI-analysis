from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.xai.explanation_metrics import deletion_auc
from src.utils.io import ensure_dir


def faithfulness_topk(model, image, label, heatmap, sample_id, out_dir, topk_values=(0.05, 0.1, 0.2, 0.3, 0.5), patch_size=16, method_name="default"):
    device = next(model.parameters()).device
    x0 = image.unsqueeze(0).to(device)
    flat_order = np.abs(np.asarray(heatmap, dtype=float)).ravel().argsort()[::-1]
    rows = []
    H, W = image.shape[1:]
    n_cols = W // patch_size
    base = x0.mean(dim=(2, 3), keepdim=True)
    for mode in ["top", "bottom", "random"]:
        if mode == "bottom":
            order = flat_order[::-1]
        elif mode == "random":
            order = np.random.default_rng(0).permutation(len(flat_order))
        else:
            order = flat_order
        probs = []
        for frac in topk_values:
            x = x0.clone()
            k = max(1, int(len(order) * frac))
            for idx in order[:k]:
                r, c = divmod(int(idx), n_cols)
                x[:, :, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = base
            with torch.no_grad():
                p = F.softmax(model(x), dim=1)[0]
            rows.append({"sample_id": sample_id, "method": method_name, "mask_mode": mode, "fraction": frac,
                         "target_prob": float(p[int(label)]), "predicted_class": int(p.argmax())})
            if mode == "top":
                probs.append(float(p[int(label)]))
        if mode == "top":
            rows.append({"sample_id": sample_id, "method": method_name, "mask_mode": "top", "fraction": "auc",
                         "target_prob": deletion_auc(topk_values, probs), "predicted_class": -1})
    out = ensure_dir(Path(out_dir) / "faithfulness")
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_{method_name}_deletion_curve.csv", index=False)
    auc_rows = [r for r in rows if r["fraction"] == "auc"]
    agg = out / "aggregate_deletion_auc.csv"
    pd.DataFrame(auc_rows).to_csv(agg, mode="a", header=not agg.exists(), index=False)
