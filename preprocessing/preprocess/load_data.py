#load_data.py
import pandas as pd


def load_data(file_path, text_col="text", label_col="label"):
    df = pd.read_csv(file_path)
    return df[text_col].tolist(), df[label_col].tolist()