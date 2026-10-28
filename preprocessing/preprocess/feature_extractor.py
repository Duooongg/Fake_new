# preprocessing/preprocess/feature_extractor.py

import json
from pathlib import Path
from configs.config_preprocess import PANIC_WORDS_PATH


def load_panic_words(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return set(json.load(f).get("panic_words", []))


PANIC_WORDS = load_panic_words(PANIC_WORDS_PATH)


def extract_aux_features(text: str) -> dict:
    tokens = text.split()
    panic = sum(1 for t in tokens if t in PANIC_WORDS)

    return {
        "token_count": len(tokens),
        "char_count": len(text),
        "panic_word_count": panic,
        "panic_density": panic / max(len(tokens), 1),
    }