
"""
engine.py - Generalized RuleEngine (FINAL)

- Hỗ trợ luật kết hợp (association rules) sinh từ dữ liệu
- Hỗ trợ nhiều PATTERN TOKENS: medical, policy, finance, cyber, emergency, clickbait
- Pattern đóng vai trò expert soft-rules (statistical boost)
- Soft scoring ∈ [0,1], dùng cho ensemble với CNN + SVM
"""

import json
import os
import re
from typing import List, Dict

# =========================
# OPTIONAL TOKENIZER
# =========================
try:
    from underthesea import word_tokenize as ut_word_tokenize
    USE_UNDERTHESEA = True
except Exception:
    USE_UNDERTHESEA = False

# =========================
# PATTERN TOKENS (SYNC auto_generate_rules)
# =========================
PATTERN_RULES = {
    "__MEDICAL_CLAIM__": [
        r"(chữa|khỏi|điều trị).{0,12}(dứt điểm|hoàn toàn|triệt để)",
        r"(thần dược|kỳ diệu)",
        r"không cần (phẫu thuật|hóa trị|thuốc)"
    ],
    "__POLICY_CLAIM__": [
        r"(bộ|chính phủ|ủy ban).{0,15}(ban hành|quy định|yêu cầu|chỉ đạo)",
        r"(áp dụng|thu hồi).{0,8}(ngay lập tức|toàn quốc)"
    ],
    "__FINANCE_SCAM__": [
        r"(đóng băng|phong tỏa).{0,6}tài khoản",
        r"(cam kết|đảm bảo).{0,10}(lãi|lợi nhuận)",
        r"rút tiền ngay"
    ],
    "__CYBER_THREAT__": [
        r"(hack|xâm nhập|đánh cắp dữ liệu)",
        r"(mã độc|virus|tấn công mạng)"
    ],
    "__EMERGENCY_FAKE__": [
        r"(cảnh báo khẩn cấp|ngay lập tức|ngày mai sẽ)",
        r"(nếu không).{0,10}(sẽ bị|sẽ mất)"
    ],
    "__CLICKBAIT__": [
        r"(sốc|chấn động|không thể tin)",
        r"sự thật.{0,6}(bị che giấu|lộ ra)"
    ]
}

# =========================
# STOPWORDS
# =========================
STOPWORDS = {
    "của","là","và","có","những","cho","rằng","được","trong","với",
    "một","này","khi","đã","nhiều","đến","về","từ","các","cũng",
    "như","theo","vào","đang","người","trên","tại","nếu","lên",
    "sau","qua","ra","để","nhưng","do"
}

# =========================
# HELPERS
# =========================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())

def tokenize_vn(text: str) -> List[str]:
    t = normalize_text(text)
    if USE_UNDERTHESEA:
        try:
            return ut_word_tokenize(t, format="text").split()
        except Exception:
            pass
    return t.split()

def text_contains_phrase(norm_text: str, token: str) -> bool:
    phrase = token.replace("_", " ")
    return re.search(r"\b" + re.escape(phrase) + r"\b", norm_text) is not None

def match_pattern_tokens(text: str) -> List[str]:
    norm = normalize_text(text)
    matched = []
    for tag, patterns in PATTERN_RULES.items():
        for p in patterns:
            if re.search(p, norm, re.I):
                matched.append(tag)
                break
    return matched

# =========================
# RULE ENGINE
# =========================
class RuleEngine:
    def __init__(self, rules_path: str):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f).get("rules", [])

        for r in self.rules:
            r["antecedents"] = [
                str(a).lower().replace(" ", "_")
                for a in r.get("antecedents", [])
            ]
            r["confidence"] = float(r.get("confidence", 0.0))
            r["lift"] = float(r.get("lift", 1.0))

        self.max_weight = max(
            [r["confidence"] * r["lift"] for r in self.rules],
            default=1.0
        )

    # =========================
    # PREPARE
    # =========================
    def _prepare(self, text: str):
        norm = normalize_text(text)
        toks = tokenize_vn(text)
        toks = [
            t.replace(" ", "_")
            for t in toks
            if t not in STOPWORDS and len(t) > 2
        ]
        toks.extend(match_pattern_tokens(text))
        return norm, set(toks)

    # =========================
    # SCORE
    # =========================
    def score(self, text: str) -> float:
        if not self.rules:
            return 0.0

        norm, toks = self._prepare(text)
        weights = []

        for r in self.rules:
            ants = set(r["antecedents"])
            if ants and all(
                (a in toks) or text_contains_phrase(norm, a)
                for a in ants
            ):
                w = r["confidence"] * r["lift"] * (1 + 0.2 * (len(ants) - 1))
                weights.append(w)

        # Expert pattern boost
        pattern_hits = match_pattern_tokens(text)
        if pattern_hits and not weights:
            return 0.35  # đủ mạnh để không rơi vào "uncertain"

        if not weights:
            return 0.05

        avg_w = sum(sorted(weights, reverse=True)[:5]) / min(5, len(weights))
        return min(avg_w / self.max_weight, 1.0)

    # =========================
    # EXPLAIN
    # =========================
    def explain(self, text: str) -> List[Dict]:
        norm, toks = self._prepare(text)
        activated = []

        for r in self.rules:
            ants = r["antecedents"]
            if ants and all(
                (a in toks) or text_contains_phrase(norm, a)
                for a in ants
            ):
                activated.append({
                    "antecedents": ants,
                    "confidence": r["confidence"],
                    "lift": r["lift"],
                    "type": "association_rule"
                })

        for p in match_pattern_tokens(text):
            activated.append({
                "antecedents": [p],
                "type": "expert_pattern"
            })

        return activated
