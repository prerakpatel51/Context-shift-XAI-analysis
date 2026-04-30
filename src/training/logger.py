from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import append_jsonl, ensure_dir


class MetricLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = ensure_dir(log_dir)
        self.rows = []

    def log(self, row: dict):
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.log_dir / "metrics_epoch.csv", index=False)
        append_jsonl(row, self.log_dir / "metrics_epoch.jsonl")
