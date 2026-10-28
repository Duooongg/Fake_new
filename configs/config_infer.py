# src/infer/config.py
"""
Inference configuration
CNN + SVM + Rule-based Ensemble
"""

import torch
from configs.shared import ROOT_DIR

# =========================
# Paths
# =========================

CNN_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "cnn_best.pt"
SVM_MODEL_PATH = ROOT_DIR / "checkpoints" / "svm_model.pkl"
RULES_PATH = ROOT_DIR / "json" / "rules_auto.json"

# =========================
# Device
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Ensemble thresholds
# =========================

CNN_THRESHOLD = 0.5
SVM_THRESHOLD = 0.5
FINAL_DECISION_THRESHOLD = 0.45

# =========================
# Rule-based influence
# =========================

RULE_ALERT_THRESHOLD = 0.1
ENABLE_RULE_EXPLAIN = True
MAX_RULE_EXPLANATIONS = 5