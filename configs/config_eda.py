# preprocessing/eda/eda_config.py
"""
EDA configuration (optional, NOT part of model pipeline)
"""

from configs.shared import LABELS

LABEL_FIELD = "label"

TEXT_FIELDS = [
    "title",
    "text"
]

EXCLUDED_COLUMNS = [
    "id",
    "url",
    "author",
    LABEL_FIELD
]

QUANTILES = [0.25, 0.5, 0.75]