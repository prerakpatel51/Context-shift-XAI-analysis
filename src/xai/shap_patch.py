from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.io import ensure_dir
from src.xai.explanation_metrics import entropy, topk_concentration
from src.xai.plotting import save_overlay


def _mask_image(x, mask_flat, patch_size, baseline):
    _, _, H, W = x.shape
    n_cols = W // patch_size
    out = x.clone()
    idx = np.where(mask_flat == 0)[0]
    for k in idx:
        r, c = divmod(int(k), n_cols)
        y0, x0 = r * patch_size, c * patch_size
        out[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = baseline
    return out


def _kernel_weight(M, s):
    if s == 0 or s == M:
        return 1e6
    from math import comb
    return (M - 1) / (comb(M, s) * s * (M - s))


def kernel_shap_patch(
    model,
    image,
    label,
    sample_id,
    out_dir,
    patch_size=16,
    n_samples=512,
    baseline="mean",
    seed=0,
):
    """KernelSHAP over patch coalitions.

    Coalition vector z in {0,1}^M (M = #patches). z=1 keeps patch, z=0 replaces
    with baseline. Predictor f(z) = P(target | masked_image). Solve weighted
    linear regression with SHAP kernel weights -> per-patch Shapley values.
    """
    model.eval()
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device)
    _, _, H, W = x.shape
    n_rows, n_cols = H // patch_size, W // patch_size
    M = n_rows * n_cols
    target = int(label)

    base = x.mean(dim=(2, 3), keepdim=True) if baseline == "mean" else torch.zeros_like(x[:, :, :1, :1])

    with torch.no_grad():
        p_full = float(F.softmax(model(x), dim=1)[0, target])
        p_empty = float(F.softmax(model(_mask_image(x, np.zeros(M), patch_size, base)), dim=1)[0, target])

    rng = np.random.default_rng(seed)
    Z = np.zeros((n_samples, M), dtype=np.float32)
    w = np.zeros(n_samples, dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)

    sizes = rng.integers(1, M, size=n_samples)
    for i, s in enumerate(sizes):
        idx = rng.choice(M, size=int(s), replace=False)
        Z[i, idx] = 1.0
        w[i] = _kernel_weight(M, int(s))
        with torch.no_grad():
            xm = _mask_image(x, Z[i], patch_size, base)
            y[i] = float(F.softmax(model(xm), dim=1)[0, target])

    y_centered = y - p_empty
    target_diff = p_full - p_empty
    A = np.hstack([Z, np.ones((n_samples, 1))])
    A_eq = np.zeros((1, M + 1))
    A_eq[0, :M] = 1.0
    A_eq[0, M] = 0.0
    W = np.diag(w)
    big = 1e8
    AtWA = A.T @ W @ A + big * (A_eq.T @ A_eq)
    AtWy = A.T @ W @ y_centered + big * A_eq.T.flatten() * target_diff
    try:
        sol = np.linalg.solve(AtWA, AtWy)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(AtWA, AtWy, rcond=None)
    phi = sol[:M]

    heat = phi.reshape(n_rows, n_cols)

    out = ensure_dir(Path(out_dir) / "shap")
    np.save(out / f"{sample_id}_shap.npy", heat)
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            rows.append({"sample_id": sample_id, "patch_row": r, "patch_col": c, "shap_value": float(heat[r, c])})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "shap_value": float(entropy(np.abs(heat))), "metric": "entropy_abs"})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "shap_value": float(topk_concentration(np.abs(heat))), "metric": "topk_concentration_abs"})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "shap_value": float(phi.sum()), "metric": "sum_phi"})
    rows.append({"sample_id": sample_id, "patch_row": -1, "patch_col": -1, "shap_value": float(target_diff), "metric": "target_diff_full_minus_empty"})
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_shap.csv", index=False)
    save_overlay(image, np.abs(heat), out / f"{sample_id}_shap_overlay.png")
    return heat
