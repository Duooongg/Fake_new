import pandas as pd
import re

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b[a-zà-ỹ]{2,}\b", text)

df = pd.read_csv("dataset/data_processed/train.csv")

transactions = df["text"].astype(str).apply(tokenize).tolist()

pd.to_pickle(transactions, "rules/transactions.pkl")
print("✅ Saved transactions")