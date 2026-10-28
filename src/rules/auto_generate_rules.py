
"""
auto_generate_rules.py (GENERALIZED VERSION)

- Sinh luật kết hợp (association rules) cho Fake News đa lĩnh vực
- Phân biệt FAKE vs REAL dựa trên support ratio
- Hỗ trợ token thường + token domain + token pattern (regex)
"""

import os
import json
import re
from collections import Counter
from typing import List, Set
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# =========================
# OPTIONAL NLP
# =========================
try:
    from underthesea import word_tokenize
    USE_UNDERTHESEA = True
except:
    USE_UNDERTHESEA = False

# =========================
# PATH
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "data_processed", "train.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "json", "rules_auto.json")

# =========================
# PARAMS
# =========================
MIN_SUPPORT = 0.04
MIN_CONFIDENCE = 0.7
MIN_LIFT = 2.0
MAX_ANTECEDENTS = 3
FAKE_REAL_RATIO = 2.5
MIN_TOKEN_FREQ = 8

# =========================
# STOPWORDS
# =========================
STOPWORDS = {
    "và","là","có","cho","trong","với","một","những","được","khi","đã",
    "này","đến","từ","các","cũng","như","lại","vẫn","theo","vào","ra",
    "nếu","để","nhưng","do","bởi","trên","dưới","vừa","rằng","nhiều"
}

# =========================
# DOMAIN TOKENS (RISK SIGNALS)
# =========================
DANGER_TOKENS = {

    # ---- GOVERNMENT / POLICY ----
    "chính_phủ","bộ_y_tế","bộ_công_an","ban_hành","thu_hồi",
    "xử_phạt","cấm","bắt_buộc","toàn_bộ","quy_định_mới",

    # ---- FINANCE / INVEST ----
    "lợi_nhuận","cam_kết","đầu_tư","không_rủi_ro",
    "lãi_suất_cao","nhân_đôi","đảm_bảo_lãi",

    # ---- CRYPTO ----
    "coin","token","airdrop","staking","defi","ví_điện_tử",

    # ---- MEDICAL ----
    "thần_dược","chữa_khỏi","không_tác_dụng_phụ","khỏi_100%",

    # ---- CYBER SECURITY ----
    "hack","malware","virus","tải_về","apk","link_lạ",

    # ---- SOCIAL SCAM ----
    "chuyển_khoản","liên_hệ_riêng","inbox","group_kín",

    # ---- CLICKBAIT ----
    "khẩn_cấp","cảnh_báo","sốc","không_thể_tin",
    "sự_thật_bị_che_giấu"
}

# =========================
# PATTERNS → TOKEN
# =========================
PATTERN_TOKENS = {
    "__MEDICAL_CLAIM__": [
        r"(chữa|khỏi).{0,12}(dứt điểm|hoàn toàn)",
        r"(thần dược|kỳ diệu)",
        r"không cần (phẫu thuật|hóa trị)"
    ],
    "__POLICY_CLAIM__": [
        r"(chính phủ|bộ).{0,15}(ban hành|quy định|thu hồi)",
        r"xử phạt.{0,10}(triệu|vĩnh viễn)"
    ],
    "__FINANCE_SCAM__": [
        r"(cam kết|đảm bảo).{0,10}(lãi|lợi nhuận)",
        r"lợi nhuận.{0,6}(cao|khủng)"
    ],
    "__CYBER_THREAT__": [
        r"(phát hiện|cảnh báo).{0,10}(virus|mã độc)",
        r"(hack|xâm nhập|đánh cắp)"
    ],
    "__CLICKBAIT__": [
        r"(sốc|chấn động|không thể tin)",
        r"sự thật.{0,6}(bị che giấu|lộ ra)"
    ]
}

# =========================
# UTILS
# =========================
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text: str) -> List[str]:
    text = normalize(text)
    if USE_UNDERTHESEA:
        return word_tokenize(text, format="text").split()
    return [w.strip(".,!?()[]{}:;\"'") for w in text.split() if len(w) > 2]

def extract_pattern_tokens(text: str) -> List[str]:
    tokens = []
    for tag, patterns in PATTERN_TOKENS.items():
        for p in patterns:
            if re.search(p, text, re.I):
                tokens.append(tag)
                break
    return tokens

def build_transactions(texts: List[str], vocab: Set[str]) -> List[List[str]]:
    transactions = []
    for t in texts:
        toks = tokenize(t)
        items = [
            tok.replace(" ", "_")
            for tok in toks
            if tok not in STOPWORDS and tok in vocab
        ]
        items += extract_pattern_tokens(t)
        if any(i in DANGER_TOKENS or i.startswith("__") for i in items):
            transactions.append(list(set(items)))
    return transactions

def calc_support(itemset, transactions):
    c = 0
    for t in transactions:
        if set(itemset).issubset(t):
            c += 1
    return c / len(transactions) if transactions else 0

# =========================
# MAIN
# =========================
def main():
    print("📥 Load dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].astype(str)
    labels = df["label"].astype(int)

    fake_texts = texts[labels == 1].tolist()
    real_texts = texts[labels == 0].tolist()

    counter = Counter()
    for t in fake_texts:
        counter.update(tokenize(t))

    vocab = {w for w, c in counter.items() if c >= MIN_TOKEN_FREQ}

    tf = build_transactions(fake_texts, vocab)
    tr = build_transactions(real_texts, vocab)

    tf_sets = [set(x) for x in tf]
    tr_sets = [set(x) for x in tr]

    te = TransactionEncoder()
    df_bin = pd.DataFrame(
        te.fit_transform(tf + tr),
        columns=te.columns_
    )

    freq = fpgrowth(
        df_bin,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ANTECEDENTS
    )

    rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)

    final_rules = []
    for _, r in rules.iterrows():
        ants = list(r["antecedents"])
        if len(ants) < 2:
            continue

        s_fake = calc_support(ants, tf_sets)
        s_real = calc_support(ants, tr_sets)

        if s_fake < MIN_SUPPORT:
            continue
        if s_real > 0 and (s_fake / (s_real + 1e-9)) < FAKE_REAL_RATIO:
            continue
        if r["lift"] < MIN_LIFT:
            continue

        final_rules.append({
            "type": "association",
            "antecedents": sorted(ants),
            "consequent": "fake",
            "support_fake": round(s_fake, 4),
            "support_real": round(s_real, 4),
            "confidence": round(float(r["confidence"]), 4),
            "lift": round(float(r["lift"]), 4)
        })

    uniq = {}
    for r in final_rules:
        k = tuple(r["antecedents"])
        if k not in uniq or r["lift"] > uniq[k]["lift"]:
            uniq[k] = r

    final_rules = sorted(
        uniq.values(),
        key=lambda x: (x["support_fake"], x["lift"]),
        reverse=True
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"rules": final_rules}, f, ensure_ascii=False, indent=2)

    print(f"✅ DONE: {len(final_rules)} rules saved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
