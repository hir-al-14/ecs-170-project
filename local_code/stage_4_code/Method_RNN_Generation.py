"""Recurrent next-token models for Stage 4 joke generation."""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_dim=128,
        dropout=0.4,
        cell_type="rnn",
    ):
        super().__init__()

        recurrent_layers = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }

        if cell_type not in recurrent_layers:
            raise ValueError("cell_type must be 'rnn', 'lstm', or 'gru'")

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.recurrent = recurrent_layers[cell_type](
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids, lengths):
        embeddings = self.embedding(token_ids)

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.recurrent(packed)

        # LSTM returns (hidden_state, cell_state).
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        final_hidden = hidden[-1]
        return self.output(self.dropout(final_hidden))
