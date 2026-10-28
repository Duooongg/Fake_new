#split_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.preprocess.validators import validate_dataframe


def split(INPUT_PATH, RANDOM_SEED=42):

    preprocess_data_dir = os.path.abspath(os.path.dirname(INPUT_PATH))

    TRAIN_PATH = os.path.join(preprocess_data_dir, "train.csv")
    VAL_PATH   = os.path.join(preprocess_data_dir, "val.csv")
    TEST_PATH  = os.path.join(preprocess_data_dir, "test.csv")

    # Load data
    df = pd.read_csv(INPUT_PATH)

    # Validate schema & labels
    validate_dataframe(df)

    # ===== 1) Train (80%) - Temp (20%) =====
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    # ===== 2) Temp -> Val (15%) & Test (5%) =====
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"]
    )

    # Save files
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Split completed:")
    print(f"Train: {len(train_df)} samples → {TRAIN_PATH}")
    print(f"Val  : {len(val_df)} samples → {VAL_PATH}")
    print(f"Test : {len(test_df)} samples → {TEST_PATH}")