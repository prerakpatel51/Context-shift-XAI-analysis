from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.xai.explanation_metrics import topk_overlap


def _spearman(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    da, db = ra - ra.mean(), rb - rb.mean()
    denom = (np.linalg.norm(da) * np.linalg.norm(db)) + 1e-12
    return float((da * db).sum() / denom)


def _pearson(a, b):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    denom = (np.linalg.norm(da) * np.linalg.norm(db)) + 1e-12
    return float((da * db).sum() / denom)


def _resize_to(a, target_shape):
    a = np.asarray(a, dtype=float)
    if a.shape == target_shape:
        return a
    from numpy import linspace
    H0, W0 = a.shape
    H1, W1 = target_shape
    ys = np.clip(np.round(linspace(0, H0 - 1, H1)).astype(int), 0, H0 - 1)
    xs = np.clip(np.round(linspace(0, W0 - 1, W1)).astype(int), 0, W0 - 1)
    return a[ys][:, xs]


def method_agreement(heatmaps: dict, sample_id, out_dir, topk_frac=0.1):
    """Compute pairwise Spearman, Pearson, top-k overlap across method heatmaps.

    Heatmaps may have different spatial sizes; smaller is resized to the
    larger via nearest-neighbor before comparison.
    """
    items = [(name, np.abs(np.asarray(h, dtype=float))) for name, h in heatmaps.items() if h is not None]
    rows = []
    for (na, ha), (nb, hb) in combinations(items, 2):
        target = ha.shape if ha.size >= hb.size else hb.shape
        a = _resize_to(ha, target)
        b = _resize_to(hb, target)
        rows.append({
            "sample_id": sample_id,
            "method_a": na,
            "method_b": nb,
            "spearman": _spearman(a, b),
            "pearson": _pearson(a, b),
            "topk_overlap": topk_overlap(a, b, k=topk_frac),
            "topk_frac": topk_frac,
            "shape_a": "x".join(map(str, ha.shape)),
            "shape_b": "x".join(map(str, hb.shape)),
        })
    if not rows:
        return None
    out = ensure_dir(Path(out_dir) / "method_agreement")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{sample_id}_method_agreement.csv", index=False)
    agg_path = out / "aggregate_method_agreement.csv"
    df.to_csv(agg_path, mode="a", header=not agg_path.exists(), index=False)
    return df
