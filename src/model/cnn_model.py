import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        dropout=0.5
    ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # ⚠️ CHỈ 1 NEURON
        self.fc = nn.Linear(
            len(filter_sizes) * num_filters,
            1
        )

    def forward(self, x):
        """
        return:
            logits: (B, 1)  → dùng cho BCEWithLogitsLoss
            feature: (B, hidden_dim) → dùng cho SVM
        """
        x = self.embedding(x)       # (B, L, D)
        x = x.unsqueeze(1)          # (B, 1, L, D)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x)).squeeze(3)     # (B, F, L')
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outs.append(p)

        feature = torch.cat(conv_outs, dim=1)
        feature = self.dropout(feature)

        logits = self.fc(feature)   # (B, 1)
        return logits, feature