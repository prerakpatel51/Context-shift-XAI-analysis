from __future__ import annotations

import numpy as np


def entropy(arr) -> float:
    x = np.asarray(arr, dtype=float).ravel()
    x = x - x.min()
    p = x / (x.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def topk_concentration(arr, k=0.1) -> float:
    x = np.asarray(arr, dtype=float).ravel()
    kk = max(1, int(len(x) * k))
    return float(np.sort(x)[-kk:].sum() / (x.sum() + 1e-12))


def deletion_auc(fracs, probs) -> float:
    return float(np.trapz(np.asarray(probs, dtype=float), np.asarray(fracs, dtype=float)))


def topk_overlap(a, b, k=0.1) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    kk = max(1, int(len(a) * k))
    sa, sb = set(a.argsort()[-kk:]), set(b.argsort()[-kk:])
    return len(sa & sb) / max(1, len(sa | sb))
