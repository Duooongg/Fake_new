import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.model.cnn_model import TextCNN
from preprocessing.preprocess.tokenizer import SimpleTokenizer


# =========================
# CONFIG
# =========================
DATA_PATH = "dataset/data_raw/vietnamese_news_dataset.csv"

TEXT_COL_MAIN = "text"
TEXT_COL_TITLE = "title"
LABEL_COL = "label"

MAX_VOCAB_SIZE = 30000
MAX_LEN = 300
EMBEDDING_DIM = 300
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

CHECKPOINT_PATH = "checkpoints/cnn_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)


# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Label gốc:", df[LABEL_COL].unique())

    texts = (
        df[TEXT_COL_TITLE].fillna("") + " " +
        df[TEXT_COL_MAIN].fillna("")
    ).astype(str).tolist()

    label_map = {
        "true_news": 0,
        "fake_news": 1
    }

    labels = df[LABEL_COL].map(label_map)

    if labels.isnull().sum() > 0:
        raise ValueError("Có label không hợp lệ")

    labels = labels.astype(int).tolist()
    print("Label sau map:", set(labels))

    return texts, labels


texts, labels = load_data()

train_texts, val_texts, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# =========================
# TOKENIZER
# =========================
tokenizer = SimpleTokenizer(
    max_vocab_size=MAX_VOCAB_SIZE,
    max_len=MAX_LEN
)
tokenizer.fit(train_texts)

X_train = torch.tensor(
    tokenizer.encode_batch(train_texts),
    dtype=torch.long
)
X_val = torch.tensor(
    tokenizer.encode_batch(val_texts),
    dtype=torch.long
)

#  BCEWithLogitsLoss → FLOAT
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)


# =========================
# DATALOADER
# =========================
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)


# =========================
# MODEL
# =========================
model = TextCNN(
    vocab_size=len(tokenizer.word2idx),
    embedding_dim=EMBEDDING_DIM
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# TRAIN LOOP
# =========================
best_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ========== VALIDATION ==========
    model.eval()
    val_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(
        f"Epoch [{epoch}/{EPOCHS}] | "
        f"Train Loss: {np.mean(train_losses):.4f} | "
        f"Val Loss: {np.mean(val_losses):.4f} | "
        f"Acc: {acc:.4f} | F1: {f1:.4f}"
    )

    # ========== SAVE BEST ==========
    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer
            },
            CHECKPOINT_PATH
        )
        print(f"Saved best model (F1={best_f1:.4f})")

print("Training completed")
