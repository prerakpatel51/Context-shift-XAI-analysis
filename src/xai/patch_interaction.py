from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.attention_hooks import get_attentions, patch_timm_attention
from src.utils.io import ensure_dir


def patch_interaction(model, image, sample_id, out_dir, top_edges=30):
    patch_timm_attention(model)
    model.eval()
    model(image.unsqueeze(0).to(next(model.parameters()).device))
    a = get_attentions(model)[-1][0].mean(0).numpy()[1:, 1:]
    rows = []
    flat = a.ravel().argsort()[-top_edges:][::-1]
    n = a.shape[0]
    for idx in flat:
        i, j = divmod(int(idx), n)
        rows.append({"sample_id": sample_id, "source_patch": i, "target_patch": j, "weight": float(a[i, j])})
    out = ensure_dir(Path(out_dir) / "patch_interaction")
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_interaction_edges.csv", index=False)
    ent = [{"sample_id": sample_id, "attention_entropy": float(-(a.ravel() * np.log(a.ravel() + 1e-12)).sum())}]
    pd.DataFrame(ent).to_csv(out / f"{sample_id}_attention_entropy.csv", index=False)
