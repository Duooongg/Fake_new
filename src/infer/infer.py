#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer.py - NewsInferencer Optimized
"""

import os
import math
import logging
from typing import Dict, Any, List

import torch
import joblib

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# IMPORT ML COMPONENTS
# =========================
from src.model.cnn_model import TextCNN
from src.rules.engine import RuleEngine
from preprocessing.preprocess.tokenizer import SimpleTokenizer, texts_to_sequence

# 🔐 FIX PyTorch 2.6+
torch.serialization.add_safe_globals([SimpleTokenizer])

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsInferencer")

# =========================
# HỖ TRỢ HÀM
# =========================
def sigmoid(x: float) -> float:
    if x < -709: return 0.0
    if x > 709: return 1.0
    return 1.0 / (1.0 + math.exp(-x))

# Từ khóa cảnh báo (Hỗ trợ cả dấu cách và gạch dưới)
FINANCIAL_KEYWORDS = {
    "đóng cửa", "đóng cửa ngân hàng", "đóng băng", "đóng băng tài khoản",
    "rút tiền", "rút hết", "đổi tiền", "khẩn cấp", "cảnh báo khẩn"
}
MEDICAL_KEYWORDS = {
    "chữa dứt điểm", "khỏi hoàn toàn", "không cần hóa trị", "thần dược", "kỳ diệu",
    "nước chanh", "tỏi", "ung thư giai đoạn cuối"
}

class NewsInferencer:
    def __init__(self):
        # 1. LOAD CNN
        checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "cnn_best.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        self.tokenizer = checkpoint["tokenizer"]
        vocab_size = len(self.tokenizer.word2idx)

        self.cnn = TextCNN(vocab_size=vocab_size, embedding_dim=300).to(DEVICE)
        self.cnn.load_state_dict(checkpoint["model_state_dict"])
        self.cnn.eval()

        # 2. LOAD SVM (Sử dụng Pipeline mới đã train)
        svm_path = os.path.join(BASE_DIR, "checkpoints", "svm_model.pkl")
        if not os.path.exists(svm_path):
            raise FileNotFoundError(f"SVM model not found: {svm_path}")
        self.svm = joblib.load(svm_path)

        # 3. LOAD RULE ENGINE
        rules_path = os.path.join(BASE_DIR, "json", "rules_auto.json")
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rules file not found: {rules_path}")
        self.rule_engine = RuleEngine(rules_path)

        # Cấu hình Thresholds
        self.low_confidence_threshold = 0.2
        self.rule_alert_threshold = 0.1
        self.high_confidence_threshold = 0.45

        logger.info(" NewsInferencer initialized with Pipeline-Aware Ensemble")

    def _detect_domain_alerts(self, explanations: List[Dict[str, Any]], text: str) -> Dict[str, bool]:
        med = False
        fin = False
        tnorm = text.lower()

        # Kiểm tra từ luật
        for r in explanations:
            ants = r.get("antecedents", []) or []
            for a in ants:
                a_norm = str(a).lower()
                if any(x in a_norm for x in ["medical", "y_te", "chanh", "toi", "thuoc"]): med = True
                if any(x in a_norm for x in ["finance", "ngan_hang", "tien"]): fin = True

        # Kiểm tra từ khóa bổ trợ (Khớp cả dấu cách và gạch dưới)
        for kw in FINANCIAL_KEYWORDS:
            if kw in tnorm or kw.replace(" ", "_") in tnorm:
                fin = True; break
        for kw in MEDICAL_KEYWORDS:
            if kw in tnorm or kw.replace(" ", "_") in tnorm:
                med = True; break

        return {"medical": med, "financial": fin, "any_alert": (med or fin)}

    def infer(self, news_input: Dict[str, Any]) -> Dict[str, Any]:
        text = news_input.get("text", "").strip()
        if not text: raise ValueError("Empty input text")

        # CNN + Feature Extraction
        seq = texts_to_sequence(self.tokenizer, [text])
        X = torch.tensor(seq, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            cnn_logit, cnn_feat = self.cnn(X)
        cnn_score = sigmoid(float(cnn_logit.item()))

        # SVM INFERENCE (Sử dụng Pipeline)
        svm_input = cnn_feat.cpu().numpy()
        try:
            # Nhờ Pipeline có StandardScaler, svm_score sẽ cực kỳ chính xác
            svm_score = float(self.svm.predict_proba(svm_input)[0][1])
        except:
            svm_score = 0.0

        # RULE ENGINE
        rule_score = float(self.rule_engine.score(text))
        explanations = self.rule_engine.explain(text)
        domain = self._detect_domain_alerts(explanations, text)

        # ENSEMBLE LOGIC
        final_score = 0.0
        triage = False

        # CASE A: Tin rất sạch
        if cnn_score < 0.2 and not domain["any_alert"] and rule_score < 0.1:
            final_score = (cnn_score * 0.7 + svm_score * 0.3)
        
        # CASE B: Cảnh báo nguy hiểm (Y tế/Tài chính)
        elif domain["any_alert"] or rule_score > 0.1:
            triage = True
            # Nếu 1 trong 2 AI nghi ngờ, đẩy mạnh lên Tin Giả
            if cnn_score > 0.35 or svm_score > 0.35:
                final_score = max(0.85, cnn_score, svm_score, rule_score)
            else:
                # Nếu AI chưa chắc nhưng dính từ khóa nhạy cảm -> Cảnh báo trung bình
                final_score = max(0.48, rule_score)

        # CASE C: Bình thường, tin vào AI
        else:
            # Nếu CNN bắt đầu nghi, ưu tiên CNN
            if cnn_score > 0.5:


                final_score = (cnn_score * 0.8 + svm_score * 0.2)
            else:
                final_score = (cnn_score * 0.5 + svm_score * 0.5)

        # Clamp & Labeling
        final_score = max(0.01, min(0.99, final_score))
        is_fake = final_score >= self.high_confidence_threshold

        return {
            "prediction": {
                "label": 1 if is_fake else 0,
                "confidence": round(final_score, 4),
                "cnn_score": round(cnn_score, 4),
                "svm_score": round(svm_score, 4),
                "rule_score": round(rule_score, 4),
            },
            "explanation": {
                "domain_alert": domain,
                "activated_rules": explanations[:5]
            },
            "triage": triage
        }