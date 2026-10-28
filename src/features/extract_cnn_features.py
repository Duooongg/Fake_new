import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.model.cnn_model import TextCNN
from preprocessing.preprocess.tokenizer import SimpleTokenizer

# =========================
# FIX PyTorch 2.6+
# =========================
torch.serialization.add_safe_globals([SimpleTokenizer])

# =========================
# CONFIG (Khớp với train_cnn.py)
# =========================
DATA_PATH = "dataset/data_raw/vietnamese_news_dataset.csv"
TEXT_COL_MAIN = "text"
TEXT_COL_TITLE = "title"
LABEL_COL = "label"

MAX_LEN = 300
EMBEDDING_DIM = 300  # Khớp với training
BATCH_SIZE = 64

CHECKPOINT_PATH = "checkpoints/cnn_best.pt"
FEATURE_DIR = "src/features/cnn_features"
os.makedirs(FEATURE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)
    texts = (
        df[TEXT_COL_TITLE].fillna("") + " " +
        df[TEXT_COL_MAIN].fillna("")
    ).astype(str).tolist()

    label_map = {"true_news": 0, "fake_news": 1}
    labels = df[LABEL_COL].map(label_map).astype(int).values
    return texts, labels

# =========================
# EXTRACT FUNCTION
# =========================
def extract_features(model, dataloader):
    model.eval()
    features_all = []
    labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            
            # Theo train_cnn: model trả về (logits, features)
            _, feats = model(X_batch) 
            
            # Đảm bảo vector phẳng cho SVM
            if len(feats.shape) > 2:
                feats = feats.view(feats.size(0), -1)

            features_all.append(feats.cpu().numpy())
            labels_all.append(y_batch.numpy())

    return np.vstack(features_all), np.concatenate(labels_all)

# =========================
# MAIN
# =========================
def main():
    print("Loading data...")
    texts, labels = load_data()

    # Split y hệt train_cnn
    train_texts, temp_texts, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, y_val, y_test = train_test_split(
        temp_texts, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Loading CNN checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    tokenizer = checkpoint["tokenizer"]

    # SỬA LỖI TẠI ĐÂY: Chỉ truyền 2 tham số đúng như TextCNN.__init__ yêu cầu
    model = TextCNN(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("CNN loaded successfully")

    # Encode dữ liệu
    print("Encoding texts...")
    X_train = torch.tensor(tokenizer.encode_batch(train_texts), dtype=torch.long)
    X_val   = torch.tensor(tokenizer.encode_batch(val_texts), dtype=torch.long)
    X_test  = torch.tensor(tokenizer.encode_batch(test_texts), dtype=torch.long)

    # 4. Tạo DataLoader
    train_loader = DataLoader(TensorDataset(X_train, torch.tensor(y_train)), batch_size=BATCH_SIZE)
    val_loader   = DataLoader(TensorDataset(X_val, torch.tensor(y_val)), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_test, torch.tensor(y_test)), batch_size=BATCH_SIZE)

    # 5. Extract
    print("Extracting CNN features for SVM...")
    X_train_feat, y_train_feat = extract_features(model, train_loader)
    X_val_feat, y_val_feat     = extract_features(model, val_loader)
    X_test_feat, y_test_feat   = extract_features(model, test_loader)

    # 6. Save
    print("Saving .npy files...")
    np.save(os.path.join(FEATURE_DIR, "X_train.npy"), X_train_feat)
    np.save(os.path.join(FEATURE_DIR, "y_train.npy"), y_train_feat)
    np.save(os.path.join(FEATURE_DIR, "X_val.npy"), X_val_feat)
    np.save(os.path.join(FEATURE_DIR, "y_val.npy"), y_val_feat)
    np.save(os.path.join(FEATURE_DIR, "X_test.npy"), X_test_feat)
    np.save(os.path.join(FEATURE_DIR, "y_test.npy"), y_test_feat)

    print(f"DONE! Features saved to {FEATURE_DIR}")
    print(f"Feature dimension: {X_train_feat.shape[1]}")


if __name__ == "__main__":
    main()