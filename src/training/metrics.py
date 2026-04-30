from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _safe(v):
    try:
        if np.isnan(v):
            return None
    except TypeError:
        pass
    return float(v) if isinstance(v, (np.floating, float)) else v


def classification_metrics(y_true, y_pred, contexts=None, groups=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": _safe(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": _safe(balanced_accuracy_score(y_true, y_pred)),
        "per_class_accuracy": {},
    }
    for c in sorted(np.unique(y_true)):
        m = y_true == c
        out["per_class_accuracy"][str(int(c))] = _safe((y_pred[m] == y_true[m]).mean())
    if contexts is not None:
        contexts = np.asarray(contexts)
        valid = contexts >= 0
        ctx_acc = {}
        for c in sorted(np.unique(contexts[valid])):
            m = contexts == c
            ctx_acc[str(int(c))] = _safe((y_pred[m] == y_true[m]).mean())
        out["per_context_accuracy"] = ctx_acc
        out["worst_context_accuracy"] = min(ctx_acc.values()) if ctx_acc else None
    if groups is not None:
        groups = np.asarray(groups)
        vals = [g for g in np.unique(groups) if str(g)]
        grp_acc = {}
        for g in vals:
            m = groups == g
            grp_acc[str(g)] = _safe((y_pred[m] == y_true[m]).mean())
        out["per_group_accuracy"] = grp_acc
        out["worst_group_accuracy"] = min(grp_acc.values()) if grp_acc else None
    return out


@torch.no_grad()
def predict_epoch(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    ys, preds, probs, paths, sample_ids, ctxs, groups = [], [], [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += float(criterion(logits, y).item()) * len(y)
        p = torch.softmax(logits, dim=1)
        ys.extend(y.cpu().tolist())
        preds.extend(p.argmax(1).cpu().tolist())
        probs.extend(p.cpu().tolist())
        paths.extend(batch["image_path"])
        sample_ids.extend(batch["sample_id"])
        ctxs.extend(batch["context"].cpu().tolist() if torch.is_tensor(batch["context"]) else batch["context"])
        groups.extend(batch["group_id"])
    metrics = classification_metrics(ys, preds, ctxs, groups)
    metrics["loss"] = total_loss / max(1, len(ys)) if criterion is not None else None
    return metrics, {"y_true": ys, "y_pred": preds, "probs": probs, "paths": paths, "sample_ids": sample_ids, "contexts": ctxs, "groups": groups}
