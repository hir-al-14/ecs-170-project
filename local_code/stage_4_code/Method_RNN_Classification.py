import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):

    def __init__(self, gloveW, hidden_SZ=100, dropout=0.4, cell_type="rnn"):
        super().__init__()

        # use GloVe for words
        self.embedding = nn.Embedding.from_pretrained(
            gloveW, freeze=False, padding_idx=0
        )

        # choose model type
        cells = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }

        if cell_type not in cells:
            raise ValueError("Invalid cell type")

        # use same setting for all models
        self.rnn = cells[cell_type](
            gloveW.shape[1],
            hidden_SZ,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_SZ * 2, 2)

    def forward(self, x, lengths):

        x = self.embedding(x)

        # ignore padding
        x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        out, h = self.rnn(x)

        # LSTM has two states
        if isinstance(h, tuple):
            h = h[0]

        # combine both directions
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)

        # return two class scores
        return self.fc(h)
