# infer/eda_loader.py

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from configs.config_infer import EDA_FILES, MAX_EXPLAIN_PHRASES


class EDAStats:
    """
    Load and compare EDA statistics for post-hoc explanation during inference.

    NOTE:
    - Designed for BINARY classification: real / fake
    - Explanations are statistical context, NOT decision rules
    """

    def __init__(self):
        self.label_distribution = self._load_csv(
            EDA_FILES.get("label_distribution")
        )
        self.label_binary_stats = self._load_json(
            EDA_FILES.get("label_binary_stats")
        )
        self.label_text_length_stats = self._load_json(
            EDA_FILES.get("label_text_length_stats")
        )

    # =====================================================
    # LOADERS
    # =====================================================

    def _load_json(self, path) -> Dict[str, Any]:
        if path is None:
            return {}
        path = Path(path)
        if not path.exists():
            print(f"[Warning] EDA JSON file not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_csv(self, path) -> pd.DataFrame:
        if path is None:
            return pd.DataFrame()
        path = Path(path)
        if not path.exists():
            print(f"[Warning] EDA CSV file not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    # =====================================================
    # LABEL PRIOR
    # =====================================================

    def get_label_prior(self, label: str) -> float:
        if self.label_distribution.empty:
            return 0.0
        row = self.label_distribution[self.label_distribution["label"] == label]
        if row.empty:
            return 0.0
        return float(row["ratio"].values[0])

    # =====================================================
    # TEXT LENGTH
    # =====================================================

    def check_text_length(self, label: str, word_count: int) -> List[str]:
        reasons = []
        stats = self.label_text_length_stats.get(label)
        if not stats:
            return reasons

        p10 = stats.get("p10")
        p90 = stats.get("p90")

        if p10 is not None and word_count < p10:
            reasons.append(
                "Văn bản ngắn hơn mức phổ biến trong dữ liệu huấn luyện"
            )
        if p90 is not None and word_count > p90:
            reasons.append(
                "Văn bản dài hơn mức phổ biến trong dữ liệu huấn luyện"
            )
        return reasons

    # =====================================================
    # BINARY FEATURES (FIXED)
    # =====================================================

    def compare_binary_features(
        self, label: str, features: Dict[str, bool]
    ) -> List[str]:
        reasons = []
        label_stats = self.label_binary_stats.get(label)
        if not label_stats:
            return reasons

        for feat, present in features.items():
            if not present:
                continue

            freq_info = label_stats.get(feat, 0.0)

            # FIX: handle both float and dict cases
            if isinstance(freq_info, dict):
                freq = freq_info.get(label, 0.0)
            else:
                freq = freq_info

            if isinstance(freq, (int, float)) and freq >= 0.5:
                reasons.append(
                    f"Đặc trưng '{feat}' thường xuất hiện trong tin {label}"
                )

        return reasons

    # =====================================================
    # MAIN EXPLAIN
    # =====================================================

    def explain(self, label: str, features: Dict[str, Any]) -> Dict[str, Any]:
        reasons: List[str] = []

        reasons += self.compare_binary_features(
            label, features.get("binary", {})
        )
        reasons += self.check_text_length(
            label, features.get("text_length", 0)
        )

        prior = self.get_label_prior(label)

        return {
            "eda_reasons": reasons[:MAX_EXPLAIN_PHRASES],
            "num_matches": len(reasons),
            "label_prior": round(prior, 4),
            "eda_confidence": round(
                min(len(reasons) / MAX_EXPLAIN_PHRASES, 1.0), 3
            ),
        }
