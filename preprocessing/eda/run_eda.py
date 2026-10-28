# preprocessing/eda/run_eda.py

import argparse
import os
import sys
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

from preprocessing.eda.eda_utils import save_json, save_csv
from preprocessing.eda.eda_stats import (
    compute_data_quality,
    compute_label_distribution,
    compute_text_length_stats,
    compute_temporal_stats,
)
from configs.config_eda import LABEL_FIELD, DATE_FIELD


def run_eda(input_csv: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[EDA] Input file not found: {input_csv}")

    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(input_csv), "../..")),
        "dataset/data_eda"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"[EDA] Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    if LABEL_FIELD not in df.columns:
        raise ValueError(f"[EDA] Missing required column: {LABEL_FIELD}")

    # =====================================================
    # Data quality
    # =====================================================
    print("[EDA] Computing data quality...")
    quality = compute_data_quality(df)
    save_json(quality, os.path.join(output_dir, "data_quality.json"))

    # =====================================================
    # Label distribution
    # =====================================================
    print("[EDA] Computing label distribution...")
    label_dist = compute_label_distribution(df)
    save_csv(label_dist, os.path.join(output_dir, "label_distribution.csv"))

    # =====================================================
    # Text length stats
    # =====================================================
    print("[EDA] Computing text length statistics...")
    text_len_stats = compute_text_length_stats(df)
    save_json(
        text_len_stats,
        os.path.join(output_dir, "label_text_length_stats.json")
    )

    # =====================================================
    # Temporal stats (optional)
    # =====================================================
    if DATE_FIELD in df.columns:
        print("[EDA] Computing temporal statistics...")
        temporal_df = compute_temporal_stats(df)
        if not temporal_df.empty:
            save_csv(
                temporal_df,
                os.path.join(output_dir, "label_time_distribution.csv")
            )

    print("[EDA] DONE ✔ All EDA artifacts saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA for Fake News Detection")
    parser.add_argument(
        "--input",
        default=os.path.join(
            root_dir,
            "dataset/data_raw/vietnamese_news_dataset.csv"
        ),
        help="Path to input CSV"
    )
    args = parser.parse_args()
    run_eda(args.input)