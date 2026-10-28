# preprocessing/eda/eda_utils.py

import json
import os
from typing import Dict, List, Optional
import pandas as pd


# =========================
# File & directory helpers
# =========================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


# =========================
# Numeric statistics (for text length only)
# =========================

def numeric_summary(
    series: pd.Series,
    quantiles: Optional[List[float]] = None
) -> Dict:
    if series is None:
        return {}

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}

    stats = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
    }

    if quantiles:
        for q in quantiles:
            stats[f"p{int(q * 100)}"] = float(s.quantile(q))

    return stats


# =========================
# Time utilities
# =========================

def parse_datetime(
    series: pd.Series,
    date_format: Optional[str] = None
) -> pd.Series:
    return pd.to_datetime(series, format=date_format, errors="coerce")