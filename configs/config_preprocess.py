# preprocessing/preprocess/config.py
"""
Preprocessing configuration
Used for CNN tokenizer & rule-based inference
"""

from configs.shared import RANDOM_SEED, ROOT_DIR

# =========================
# Modes
# =========================

TRAIN_MODE = "train"
INFER_MODE = "infer"

# =========================
# Cleaning & normalization
# =========================

LOWERCASE = True
REMOVE_URL = True
NORMALIZE_WHITESPACE = True

# =========================
# Token constraints
# =========================

MIN_TOKEN_LENGTH = 1

# =========================
# Rule-support resources
# =========================

PANIC_WORDS_PATH = ROOT_DIR / "json" / "panic_words.json"
SYNONYM_MAP_PATH = ROOT_DIR / "json" / "synonyms.json"