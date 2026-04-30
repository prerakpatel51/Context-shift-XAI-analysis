from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_FIELDS = ["image_path", "filename", "path", "img_path", "file"]
CLASS_FIELDS = ["label", "class", "object", "category", "concept", "y"]
CONTEXT_FIELDS = ["context", "domain", "environment", "env", "background"]
SPLIT_FIELDS = ["split", "partition", "train_val_test"]
CONTEXT_HINTS = {
    "grass", "snow", "water", "forest", "street", "indoor", "outdoor", "sky", "beach", "mountain",
    "field", "road", "desert", "city", "home", "park", "zoo", "farm", "river", "lake", "ocean",
    "autumn", "dim", "rock", "season", "day", "night",
}


def _first_col(cols: list[str], names: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def find_metadata(root: str | Path) -> Path | None:
    root = Path(root)
    for name in ["metadata.csv", "meta.csv", "annotations.csv"]:
        p = root / name
        if p.exists():
            return p
    found = list(root.rglob("metadata.csv"))
    return found[0] if found else None


def _folder_samples(root: Path) -> list[dict[str, Any]]:
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    top = {p.name for p in root.iterdir() if p.is_dir()}
    has_split_top = bool(top & {"train", "val", "valid", "validation", "test"})
    probe_parts = []
    for p in images[:200]:
        parts = p.relative_to(root).parts
        if has_split_top and parts[0] in {"train", "val", "valid", "validation", "test"}:
            parts = parts[1:]
        if len(parts) >= 3:
            probe_parts.append((parts[0].lower(), parts[1].lower()))
    layout_b_votes = sum(1 for a, _ in probe_parts if a in CONTEXT_HINTS)
    first_level = {a for a, _ in probe_parts}
    second_level = {b for _, b in probe_parts}
    layout_b = layout_b_votes > max(0, len(probe_parts) // 3) or (
        len(first_level) > 1 and len(second_level) > 1 and len(first_level) < len(second_level)
    )
    rows = []
    for i, p in enumerate(sorted(images)):
        rel = p.relative_to(root)
        parts = rel.parts
        split = None
        parts_no_split = parts
        if has_split_top and parts[0] in {"train", "val", "valid", "validation", "test"}:
            split = "val" if parts[0] in {"valid", "validation"} else parts[0]
            parts_no_split = parts[1:]
        if len(parts_no_split) >= 3 and layout_b:
            context_name, class_name = parts_no_split[0], parts_no_split[1]
        else:
            class_name = parts_no_split[0] if len(parts_no_split) >= 2 else p.parent.name
            context_name = parts_no_split[1] if len(parts_no_split) >= 3 else None
        rows.append({
            "sample_id": f"sample_{i:08d}",
            "image_path": str(p),
            "class_name": class_name,
            "context_name": context_name,
            "split": split,
        })
    return rows


def discover_samples(root: str | Path, prefer_metadata: bool = True) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    meta = find_metadata(root) if prefer_metadata else None
    if meta:
        df = pd.read_csv(meta)
        path_col = _first_col(list(df.columns), PATH_FIELDS)
        class_col = _first_col(list(df.columns), CLASS_FIELDS)
        ctx_col = _first_col(list(df.columns), CONTEXT_FIELDS)
        split_col = _first_col(list(df.columns), SPLIT_FIELDS)
        if path_col is None or class_col is None:
            raise ValueError(f"{meta} must contain image path and class fields. Columns: {list(df.columns)}")
        rows = []
        for i, r in df.iterrows():
            img = Path(str(r[path_col]))
            if not img.is_absolute():
                img = root / img
            rows.append({
                "sample_id": str(r.get("sample_id", f"sample_{i:08d}")),
                "image_path": str(img),
                "class_name": str(r[class_col]),
                "context_name": None if ctx_col is None or pd.isna(r[ctx_col]) else str(r[ctx_col]),
                "split": None if split_col is None or pd.isna(r[split_col]) else str(r[split_col]).lower(),
                "group_id": r.get("group_id", None),
            })
        out = pd.DataFrame(rows)
    else:
        out = pd.DataFrame(_folder_samples(root))
    if out.empty:
        raise ValueError(f"No images found under {root}")
    out["class_name"] = out["class_name"].astype(str)
    classes = {c: i for i, c in enumerate(sorted(out["class_name"].unique()))}
    out["class_id"] = out["class_name"].map(classes)
    if "context_name" in out and out["context_name"].notna().any():
        ctxs = {c: i for i, c in enumerate(sorted(out["context_name"].dropna().astype(str).unique()))}
        out["context_id"] = out["context_name"].map(ctxs)
        out["group_id"] = out["class_name"].astype(str) + "__" + out["context_name"].astype(str)
    else:
        warnings.warn("No context labels detected; context-wise metrics and context-based XAI will be limited.")
        out["context_name"] = None
        out["context_id"] = None
        out["group_id"] = None
    return out


class NICODataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.class_to_idx = {c: int(i) for c, i in sorted(zip(self.df.class_name, self.df.class_id), key=lambda x: x[1])}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if "context_name" in self.df and self.df["context_name"].notna().any():
            pairs = self.df.dropna(subset=["context_name", "context_id"])[["context_name", "context_id"]].drop_duplicates()
            self.context_to_idx = {str(r.context_name): int(r.context_id) for r in pairs.itertuples()}
        else:
            self.context_to_idx = {}
        self.idx_to_context = {v: k for k, v in self.context_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        context_id = -1 if pd.isna(row.get("context_id", None)) else int(row.context_id)
        group_id = "" if pd.isna(row.get("group_id", None)) else str(row.group_id)
        return {
            "image": image,
            "label": int(row.class_id),
            "context": context_id,
            "group_id": group_id,
            "image_path": str(row.image_path),
            "sample_id": str(row.sample_id),
        }
