from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.io import ensure_dir
from src.xai.plotting import denorm, save_overlay


def total_variation(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, 1:, :] - x[:, :-1, :]).abs().mean()


def _patch_mask_from_heat(heat, image_shape, patch_size, top_fraction, device):
    """Restrict perturbation to top-attribution patches (semantic CF)."""
    C, H, W = image_shape
    if heat is None:
        return torch.ones(1, C, H, W, device=device)
    h = np.asarray(heat).ravel()
    n_cols = W // patch_size
    k = max(1, int(round(h.size * top_fraction)))
    keep = set(h.argsort()[-k:].tolist())
    mask = torch.zeros(1, 1, H, W, device=device)
    for idx in keep:
        r, c = divmod(int(idx), n_cols)
        y0, x0 = r * patch_size, c * patch_size
        mask[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = 1.0
    return mask.expand(1, C, H, W)


def counterfactual(
    model,
    image,
    label,
    sample_id,
    out_dir,
    steps=50,
    lr=0.03,
    lambda_l2=0.01,
    lambda_tv=0.001,
    lambda_l1=0.001,
    lambda_linf=0.0,
    linf_budget=None,
    heat=None,
    patch_size=16,
    semantic_top_fraction=0.0,
):
    """Wachter-style counterfactual with L1 sparsity, L∞ penalty/budget, and
    optional semantic patch mask restricting perturbation to top-attribution
    regions.

    - lambda_l1 > 0: encourages sparse pixel-level changes.
    - lambda_linf > 0: soft hinge penalty pushing max |delta| below `linf_budget`.
    - linf_budget (float, in normalized-image units): hard projected clamp on |delta| each step.
    - semantic_top_fraction in (0,1]: restrict delta to the top-K attribution
      patches from `heat`. 0 = no mask (free perturbation).
    """
    device = next(model.parameters()).device
    x = image.to(device)
    with torch.no_grad():
        p0 = F.softmax(model(x.unsqueeze(0)), dim=1)[0]
    target = int(torch.topk(p0, 2).indices[1])

    region_mask = (
        _patch_mask_from_heat(heat, x.shape, patch_size, semantic_top_fraction, device)
        if semantic_top_fraction and semantic_top_fraction > 0.0
        else None
    )

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    flipped, step_flip = False, None
    for step in range(steps):
        opt.zero_grad()
        d = delta
        if region_mask is not None:
            d = delta * region_mask[0]
        cf = torch.clamp(x + d, -3, 3)
        logits = model(cf.unsqueeze(0))
        loss_ce = F.cross_entropy(logits, torch.tensor([target], device=device))
        loss_l2 = lambda_l2 * d.pow(2).mean()
        loss_tv = lambda_tv * total_variation(d)
        loss_l1 = lambda_l1 * d.abs().mean() if lambda_l1 > 0 else torch.zeros((), device=device)
        if lambda_linf > 0 and linf_budget is not None:
            loss_linf = lambda_linf * F.relu(d.abs().max() - float(linf_budget))
        else:
            loss_linf = lambda_linf * d.abs().max() if lambda_linf > 0 else torch.zeros((), device=device)
        loss = loss_ce + loss_l2 + loss_tv + loss_l1 + loss_linf
        loss.backward()
        opt.step()
        with torch.no_grad():
            if linf_budget is not None:
                delta.clamp_(min=-float(linf_budget), max=float(linf_budget))
            if region_mask is not None:
                delta.mul_(region_mask[0])
        if int(logits.argmax(1)) == target and not flipped:
            flipped, step_flip = True, step + 1
            break

    out = ensure_dir(Path(out_dir) / "counterfactual")
    with torch.no_grad():
        d_final = delta * region_mask[0] if region_mask is not None else delta
        cf_final = torch.clamp(x + d_final, -3, 3)
        p1 = F.softmax(model(cf_final.unsqueeze(0)), dim=1)[0]

    sparsity = float((d_final.abs() > 1e-4).float().mean())
    metrics = {
        "sample_id": sample_id,
        "original_prediction": int(p0.argmax()),
        "counterfactual_prediction": int(p1.argmax()),
        "target_class": target,
        "flipped": flipped,
        "steps_to_flip": step_flip,
        "l1_norm": float(d_final.abs().sum()),
        "l2_norm": float(d_final.norm()),
        "linf_norm": float(d_final.abs().max()),
        "linf_budget": linf_budget,
        "sparsity_fraction_changed": sparsity,
        "semantic_top_fraction": semantic_top_fraction,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "lambda_tv": lambda_tv,
        "lambda_linf": lambda_linf,
        "confidence_before": float(p0.max()),
        "confidence_after": float(p1.max()),
    }
    (out / f"{sample_id}_counterfactual_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"class_id": i, "original_prob": float(p0[i]), "counterfactual_prob": float(p1[i])}
        for i in range(len(p0))
    ]).to_csv(out / f"{sample_id}_counterfactual_probs.csv", index=False)
    delta_np = d_final.detach().cpu().numpy()
    np.save(out / f"{sample_id}_counterfactual_delta.npy", delta_np)

    import matplotlib.pyplot as plt

    cf_img = denorm(cf_final.detach().cpu())
    orig_img = denorm(x.detach().cpu())
    delta_signed = delta_np.sum(0)
    vmax = float(np.abs(delta_signed).max() + 1e-12)
    delta_mag = np.abs(delta_np).sum(0)

    plt.figure(figsize=(4, 4)); plt.imshow(orig_img); plt.axis("off"); plt.tight_layout()
    plt.savefig(out / f"{sample_id}_original.png", dpi=160); plt.close()

    plt.figure(figsize=(4, 4)); plt.imshow(np.clip(cf_img, 0, 1)); plt.axis("off"); plt.tight_layout()
    plt.savefig(out / f"{sample_id}_counterfactual.png", dpi=160); plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow(delta_signed, cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.colorbar(); plt.axis("off"); plt.tight_layout()
    plt.savefig(out / f"{sample_id}_delta_signed.png", dpi=160); plt.close()

    save_overlay(image, delta_mag, out / f"{sample_id}_delta_overlay.png")
