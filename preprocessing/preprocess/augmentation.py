# preprocessing/preprocess/augmentation.py

import json
import random
import re
from configs.config_preprocess import (
    SENTENCE_SHUFFLE_PROB,
    SYNONYM_PROB,
    SYNONYM_MAP_PATH
)


def load_synonyms(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SYNONYM_MAP = load_synonyms(SYNONYM_MAP_PATH)


def shuffle_sentences(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 4:
        return text
    random.shuffle(sents)
    return " ".join(sents)


def apply_synonym_noise(text: str) -> str:
    for w, syns in SYNONYM_MAP.items():
        if w in text and random.random() < SYNONYM_PROB:
            return text.replace(w, random.choice(syns), 1)
    return text


def augment_text(text: str) -> str:
    if random.random() < SENTENCE_SHUFFLE_PROB:
        text = shuffle_sentences(text)
    return apply_synonym_noise(text)