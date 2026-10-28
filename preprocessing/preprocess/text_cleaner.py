# preprocessing/preprocess/text_cleaner.py

import re
import unicodedata
from configs.config_preprocess import LOWERCASE_MODE, UNICODE_NORMAL_FORM

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")
HTML_PATTERN = re.compile(r"<.*?>")


def _controlled_lowercase(text: str) -> str:
    protected = []

    for pattern in [URL_PATTERN, EMAIL_PATTERN, ALL_CAPS_PATTERN]:
        for m in pattern.finditer(text):
            protected.append((m.start(), m.end()))

    protected = sorted(protected)
    out = []
    last = 0

    for s, e in protected:
        out.append(text[last:s].lower())
        out.append(text[s:e])
        last = e

    out.append(text[last:].lower())
    return "".join(out)


def clean_text(text: str) -> str:
    text = unicodedata.normalize(UNICODE_NORMAL_FORM, text)
    text = HTML_PATTERN.sub(" ", text)

    if LOWERCASE_MODE == "full":
        text = text.lower()
    elif LOWERCASE_MODE == "controlled":
        text = _controlled_lowercase(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text