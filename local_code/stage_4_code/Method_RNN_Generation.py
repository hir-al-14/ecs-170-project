"""Recurrent next-token models for Stage 4 joke generation."""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_dim=128,
        dropout=0.4,
        cell_type="rnn",
        tie_embeddings=False,
        embedding_dropout=0.0,
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
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.tie_embeddings = tie_embeddings

        self.recurrent = recurrent_layers[cell_type](
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        if tie_embeddings:
            self.projection = nn.Linear(hidden_dim, embedding_dim)
            self.output = nn.Linear(embedding_dim, vocab_size)
            self.output.weight = self.embedding.weight
        else:
            self.projection = nn.Identity()
            self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids, lengths):
        embeddings = self.embedding_dropout(self.embedding(token_ids))

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
        return self.output(self.projection(self.dropout(final_hidden)))

    def forward_sequence(self, token_ids, lengths):
        """Predict every next token in a batch of complete training examples."""
        embeddings = self.embedding_dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_output, _ = self.recurrent(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=token_ids.size(1),
        )
        return self.output(self.projection(self.dropout(output)))
