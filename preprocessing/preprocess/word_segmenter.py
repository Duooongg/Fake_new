# preprocessing/preprocess/word_segmenter.py

import re
from pyvi import ViTokenizer

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

CRITICAL_PATTERNS = [URL_PATTERN, EMAIL_PATTERN, IP_PATTERN]


def _protect(text):
    mapping = {}
    idx = 0
    for p in CRITICAL_PATTERNS:
        for m in p.finditer(text):
            key = f"__CRIT{idx}__"
            mapping[key] = m.group()
            text = text.replace(m.group(), key)
            idx += 1
    return text, mapping


def _restore(text, mapping):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def segment_vi(text: str) -> str:
    """
    Vietnamese word segmentation
    for CNN / SVM pipelines.
    """
    text, mapping = _protect(text)
    text = ViTokenizer.tokenize(text)
    text = _restore(text, mapping)
    return re.sub(r"\s+", " ", text).strip()