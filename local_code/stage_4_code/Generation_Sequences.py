"""Complete examples for teacher-forced next-token training."""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from local_code.stage_4_code.Text_Preprocessor import BOS_ID, EOS_ID, PAD_ID, encode_text


class JokeSequenceDataset(Dataset):
    def __init__(self, jokes, vocab, context_length):
        self.examples = []
        for joke in jokes:
            tokens = encode_text(joke, vocab)
            if not tokens:
                continue
            sequence = torch.tensor([BOS_ID, *tokens, EOS_ID], dtype=torch.long)
            if len(sequence) - 1 > context_length:
                raise ValueError("Increase context length to fit complete generation examples")
            self.examples.append((sequence[:-1], sequence[1:]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_sequences(batch):
    contexts, targets = zip(*batch)
    lengths = torch.tensor([len(context) for context in contexts], dtype=torch.long)
    return (
        pad_sequence(contexts, batch_first=True, padding_value=PAD_ID),
        lengths,
        pad_sequence(targets, batch_first=True, padding_value=PAD_ID),
    )
