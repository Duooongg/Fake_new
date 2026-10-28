# preprocessing/preprocess/pipeline.py

import pandas as pd

from .validators import validate_dataframe, validate_text
from .text_cleaner import clean_text
from .word_segmenter import segment_vi
from .augmentation import augment_text
from .feature_extractor import extract_aux_features
from configs.config_preprocess import (
    TRAIN_MODE,
    AUGMENTATION_ENABLED_MODES,
    MIN_TOKEN_LENGTH,
    LABEL2ID,
)


def preprocess_row(row, mode: str):
    """
    Preprocess a single dataframe row
    for SVM + CNN + Rule-based models.
    """

    raw_text = f"{row['title']}. {row['text']}"

    if not validate_text(raw_text):
        return None

    # Clean text
    text = clean_text(raw_text)

    # Word segmentation (ViTokenizer)
    text = segment_vi(text)

    # Token length filtering (AFTER segmentation)
    tokens = text.split()
    if len(tokens) < MIN_TOKEN_LENGTH:
        return None

    # Augmentation ONLY in training modes
    if mode in AUGMENTATION_ENABLED_MODES:
        text = augment_text(text)

    # Auxiliary features for rules / fusion
    aux_features = extract_aux_features(text)

    return {
        "id": row["id"],
        "text": text,
        "label": LABEL2ID[row["label"]],
        **aux_features,
    }


def preprocess_dataframe(df: pd.DataFrame, mode: str = TRAIN_MODE) -> pd.DataFrame:
    validate_dataframe(df)

    processed = []
    for _, row in df.iterrows():
        row_out = preprocess_row(row, mode)
        if row_out is not None:
            processed.append(row_out)

    return pd.DataFrame(processed)