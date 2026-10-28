import re
import joblib
from collections import Counter


class SimpleTokenizer:
    def __init__(self, max_vocab_size=30000, max_len=300):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(
            r"[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
            r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
            r"ùúụủũưừứựửữỳýỵỷỹđ\s]",
            " ",
            text
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            text = self.clean_text(text)
            counter.update(text.split())

        most_common = counter.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str):
        text = self.clean_text(text)
        tokens = text.split()
        ids = [self.word2idx.get(w, 1) for w in tokens]
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def save(self, path):
        joblib.dump(self, path)


def load_tokenizer(path):
    return joblib.load(path)


def texts_to_sequence(tokenizer, texts):
    return tokenizer.encode_batch(texts)
