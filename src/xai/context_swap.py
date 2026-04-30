from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.io import ensure_dir
from src.xai.plotting import denorm, save_overlay


def _prob_row(model, image):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probs = logits.softmax(dim=-1)[0].detach().cpu()
    pred = int(probs.argmax())
    return pred, float(probs[pred]), probs


def _patch_indices(heat, fraction, largest=False):
    flat = np.asarray(heat).reshape(-1)
    k = max(1, int(round(flat.size * fraction)))
    order = np.argsort(flat)
    return order[-k:] if largest else order[:k]


def _swap_patches(image, partner, patch_indices, patch_size):
    swapped = image.clone()
    _, height, width = swapped.shape
    patches_per_row = width // patch_size
    for idx in patch_indices:
        row, col = divmod(int(idx), patches_per_row)
        y0, x0 = row * patch_size, col * patch_size
        if y0 + patch_size <= height and x0 + patch_size <= width:
            swapped[:, y0:y0 + patch_size, x0:x0 + patch_size] = partner[:, y0:y0 + patch_size, x0:x0 + patch_size]
    return swapped


def _save_image(image, path):
    import matplotlib.pyplot as plt

    arr = denorm(image)
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def context_swap_attribution(model, ds, index_by_sample_id, sample_id, image, label, heat, out_dir, cfg):
    """Swap low-attribution same-class/different-context patches as context evidence.

    Without object masks, low-attribution patches are used as a conservative
    proxy for background/context. A same-class image from a different context
    provides the donor patches, so the class label is controlled while context
    changes.
    """
    row = ds.df.iloc[index_by_sample_id[str(sample_id)]]
    class_name = str(row.class_name)
    context_name = None if pd.isna(row.get("context_name", None)) else str(row.context_name)
    candidates = ds.df[(ds.df["class_name"].astype(str) == class_name) & (ds.df["sample_id"].astype(str) != str(sample_id))]
    if context_name is not None and "context_name" in candidates:
        candidates = candidates[candidates["context_name"].astype(str) != context_name]
    if candidates.empty:
        return None

    partner_row = candidates.sample(n=1, random_state=abs(hash(str(sample_id))) % (2**32)).iloc[0]
    partner = ds[int(partner_row.name)]["image"]
    patch_size = int(cfg.get("patch_size", 16))
    swap_fraction = float(cfg.get("context_swap", {}).get("swap_fraction", 0.25))
    patch_idxs = _patch_indices(heat, swap_fraction, largest=False)
    swapped = _swap_patches(image, partner, patch_idxs, patch_size)

    orig_pred, orig_conf, orig_probs = _prob_row(model, image)
    swap_pred, swap_conf, swap_probs = _prob_row(model, swapped)
    target = int(label)
    target_drop = float(orig_probs[target] - swap_probs[target])
    pred_drop = float(orig_conf - swap_conf)

    out = ensure_dir(Path(out_dir) / "context_swap")
    rows = [{
        "sample_id": sample_id,
        "class_name": class_name,
        "source_context": context_name,
        "partner_sample_id": str(partner_row.sample_id),
        "partner_context": None if pd.isna(partner_row.get("context_name", None)) else str(partner_row.context_name),
        "swap_fraction": swap_fraction,
        "num_swapped_patches": int(len(patch_idxs)),
        "true_label": target,
        "original_prediction": orig_pred,
        "swapped_prediction": swap_pred,
        "original_confidence": orig_conf,
        "swapped_confidence": swap_conf,
        "true_class_probability_before": float(orig_probs[target]),
        "true_class_probability_after": float(swap_probs[target]),
        "true_class_probability_drop": target_drop,
        "predicted_confidence_drop": pred_drop,
        "prediction_changed": bool(orig_pred != swap_pred),
        "swapped_patch_indices": " ".join(map(str, patch_idxs.tolist())),
    }]
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_context_swap.csv", index=False)
    save_overlay(image, heat, out / f"{sample_id}_context_proxy_overlay.png")
    _save_image(swapped, out / f"{sample_id}_context_swapped.png")
    return rows[0]
