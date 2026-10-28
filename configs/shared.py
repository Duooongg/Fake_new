# configs/shared.py
"""
Shared configuration for Fake News Detection System
Model: CNN + SVM + Rule-based
"""

import sys
from pathlib import Path

# =========================================================
# Project paths
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# =========================================================
# Project metadata
# =========================================================

PROJECT_NAME = "Fake News Detection (CNN + SVM + Rules)"
PROJECT_VERSION = "1.0.0"

# =========================================================
# Labels (GLOBAL)
# =========================================================

LABELS = ["true_news", "fake_news"]

LABEL2ID = {
    "true_news": 0,
    "fake_news": 1
}

ID2LABEL = {
    0: "true_news",
    1: "fake_news"
}

NUM_CLASSES = 2

# =========================================================
# Label semantics (for explanation / UI)
# =========================================================

LABEL_DESCRIPTIONS = {
    "true_news": "Tin tức phản ánh đúng sự thật, có thể kiểm chứng.",
    "fake_news": "Tin tức sai lệch hoặc gây hiểu lầm cho người đọc."
}

# =========================================================
# CNN input configuration (USED)
# =========================================================

CNN_MAX_LEN = 300
CNN_EMBEDDING_DIM = 300

# =========================================================
# Reproducibility
# =========================================================

RANDOM_SEED = 42