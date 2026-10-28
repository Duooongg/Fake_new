# src/train/config.py
"""
Training configuration
Used for CNN training only
"""

import torch
from configs.shared import (
    ROOT_DIR,
    CNN_EMBEDDING_DIM,
    CNN_MAX_LEN,
    RANDOM_SEED
)

# =========================
# Paths
# =========================

DATA_PATH = ROOT_DIR / "dataset" / "data_raw" / "vietnamese_news_dataset.csv"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# =========================
# CNN training hyperparameters
# =========================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# =========================
# Device
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")