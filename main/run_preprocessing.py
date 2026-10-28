# =========================
# FIX PYTHON PATH (BẮT BUỘC ĐẶT TRÊN CÙNG)
# =========================
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# =========================
# IMPORT SAU KHI FIX PATH
# =========================
import pandas as pd
from preprocessing.preprocess.pipeline import preprocess_dataframe
from configs.config_preprocess import TRAIN_MODE

# =========================
# MAIN FUNCTION
# =========================
def run_preprocessing(input_csv_path: str):
    print("▶ Dữ liệu đầu vào:", input_csv_path)

    OUTPUT_CSV_PATH = os.path.join(
        ROOT_DIR, "dataset", "data_processed", "train_processed.csv"
    )

    # ====== 1. LOAD CSV ======
    df = pd.read_csv(input_csv_path)

    # ====== 2. EDA ======
    from preprocessing.eda.run_eda import run_eda
    print("\n▶ Đang phân tích dữ liệu (EDA)...")
    run_eda(input_csv_path)

    # ====== 3. PREPROCESS ======
    print("\n▶ Đang tiền xử lý văn bản...")
    processed_df = preprocess_dataframe(df, mode=TRAIN_MODE)

    # ====== 4. SAVE ======
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Saved processed data to {OUTPUT_CSV_PATH}")

    # ====== 5. SPLIT DATA ======
    from preprocessing.preprocess.split_data import split
    print("\n▶ Đang chia tập train / val / test...")
    split(OUTPUT_CSV_PATH)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    INPUT_CSV = os.path.join(
        ROOT_DIR, "dataset", "data_raw", "vietnamese_news_dataset.csv"
    )
    run_preprocessing(INPUT_CSV)
