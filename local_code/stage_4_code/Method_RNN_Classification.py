import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNNClassifier(nn.Module):

    def __init__(self, gloveW, hidden_SZ=100, dropout=0.4):
        super().__init__()

        # starts from GloVe datasets
        self.embedding = nn.Embedding.from_pretrained(gloveW, freeze=False, padding_idx=0)
        self.rnn = nn.RNN(gloveW.shape[1], hidden_SZ, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # two classes
        self.fc = nn.Linear(hidden_SZ * 2, 2)

    def forward(self, x, lengths):
        # batch, time, 50
        x = self.embedding(x)                 
        # ignore padding at end
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(x)                

        h = torch.cat((h[-2], h[-1]), dim=1)                        
        h = self.dropout(h)
        # return 2 raw scores
        return self.fc(h)