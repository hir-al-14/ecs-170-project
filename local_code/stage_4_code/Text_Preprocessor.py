"""Stage 4 text preprocessing utilities for classification and joke generation.

tokenize — Lowercase, strip HTML, split into words/numbers/punctuation.
detokenize — Reassemble tokens into readable text.
build_vocab — Build token→ID map from training texts.
encode_text — Convert a string to vocab IDs (no padding/BOS/EOS).
ReviewDataset — Dataset of (token_ids, label) for sentiment.
collate_reviews — Pad reviews → (padded, lengths, labels).
JokeDataset — Dataset of (context, next_token) pairs per joke.
collate_jokes — Pad contexts → (padded, lengths, targets).
encode_prompt — Encode 3 seed words as [BOS, w1, w2, w3].
load_glove_matrix — Stream GloVe file into an embedding matrix for vocab.


Exposes PyTorch datasets and collate functions. No file I/O or training.

Vocabulary rules:
    0 = <PAD>  (padding only)
    1 = <UNK>  (out-of-vocabulary words)
    2 = <BOS>  (beginning of a joke)
    3 = <EOS>  (end of a joke)

Build vocabularies using TRAINING data only.
Classification and generation may use different vocabularies
Their word IDs may not be interchangeable.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

_HTML_TAG = re.compile(r"<[^>]*>")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip HTML tags, and preserve words, numbers, punctuation.

    Keep punctuation: tokenizer preserves it for joke generation.
    For example: "Don't!" -> ["don't", "!"].
    """
    text = _HTML_TAG.sub(" ", text).lower()
    return _WORD.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    """Readable (not byte-perfect) text reconstructed from generated tokens."""
    text = " ".join(t for t in tokens if t not in SPECIAL_TOKENS)
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\b(\w+)\s+'(\w+)\b", r"\1'\2", text)
    return text.strip()


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int = 20_000,
    min_freq: int = 1,
) -> dict[str, int]:
    """Build token->ID mapping from training texts only.

    max_vocab_size includes four reserved special tokens. Break ties
    alphabetically for reproducible IDs independent of text order.
    """
    if max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("max_vocab_size must be at least 4")
    if min_freq < 1:
        raise ValueError("min_freq must be >= 1")
    counts = Counter(token for text in texts for token in tokenize(text))
    candidates = sorted(
        ((token, count) for token, count in counts.items()
         if count >= min_freq and token not in SPECIAL_TOKENS),
        key=lambda item: (-item[1], item[0]),
    )
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in candidates[:max_vocab_size - len(SPECIAL_TOKENS)]:
        vocab[token] = len(vocab)
    return vocab


def encode_text(
    text: str,
    vocab: Mapping[str, int],
    max_length: int | None = None,
) -> list[int]:
    """Tokenize and encode one text (no padding or BOS/EOS)."""
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be >= 1 or None")
    tokens = tokenize(text)
    if max_length is not None:
        tokens = tokens[:max_length]
    return [vocab.get(token, UNK_ID) for token in tokens]


class ReviewDataset(Dataset):
    """IMDb reviews with labels 0=negative and 1=positive.

    Returns (token_ids, label); use collate_reviews with a DataLoader to get
    (padded_ids, true_lengths, labels).
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Mapping[str, int],
        max_length: int = 200,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        ids = encode_text(self.texts[index], self.vocab, self.max_length)
        if not ids:
            ids = [UNK_ID]  # pack_padded_sequence cannot handle length 0
        label = int(self.labels[index])
        if label not in (0, 1):
            raise ValueError(f"Expected label 0 or 1, got {label}")
        return torch.tensor(ids, dtype=torch.long), label


def collate_reviews(batch):
    """Return (padded_ids[B,T], lengths[B], labels[B])."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class JokeDataset(Dataset):
    """Next-word prediction examples from *individual* joke strings.

    Each joke is encoded as [BOS, tokens..., EOS]. A rolling window supplies
    up to `context_length` preceding tokens to predict the next token.
    Example: [BOS, 'what', 'did'] -> 'the'. Padding is applied by collate_jokes.

    Important: each entry in `jokes` must be ONE joke, not a whole file full of
    jokes. Dataset_Loader should split the raw data using its actual format.
    """

    def __init__(
        self,
        jokes: Sequence[str],
        vocab: Mapping[str, int],
        context_length: int = 20,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        self.context_length = context_length
        self.examples: list[tuple[list[int], int]] = []
        for joke in jokes:
            token_ids = encode_text(joke, vocab)
            if not token_ids:
                continue
            sequence = [BOS_ID, *token_ids, EOS_ID]
            for target_index in range(1, len(sequence)):
                start = max(0, target_index - context_length)
                self.examples.append((
                    sequence[start:target_index],
                    sequence[target_index],
                ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        context, target = self.examples[index]
        return torch.tensor(context, dtype=torch.long), int(target)


def collate_jokes(batch):
    """Return (padded_contexts[B,T], lengths[B], next_word_targets[B])."""
    contexts, targets = zip(*batch)
    lengths = torch.tensor([len(context) for context in contexts], dtype=torch.long)
    padded = pad_sequence(contexts, batch_first=True, padding_value=PAD_ID)
    return padded, lengths, torch.tensor(targets, dtype=torch.long)


def encode_prompt(three_words: str, vocab: Mapping[str, int]) -> list[int]:
    """Encode exactly three words for autoregressive joke generation.

    Returns [BOS_ID, word1, word2, word3]. Unknown words map to UNK_ID.
    During generation, use the model to predict one next token at a time,
    stop at EOS_ID, and retain at most `context_length` context tokens.
    """
    words = tokenize(three_words)
    if len(words) != 3 or any(not re.fullmatch(r"[a-z]+(?:'[a-z]+)?|\d+", w) for w in words):
        raise ValueError("Prompt must contain exactly three words (no punctuation)")
    return [BOS_ID, *[vocab.get(w, UNK_ID) for w in words]]


def load_glove_matrix(
    glove_path: str | Path,
    vocab: Mapping[str, int],
    embedding_dim: int = 50,
    seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """Load only vectors needed by `vocab` from a plain-text GloVe file.

    Returns (weight_matrix[V,D], matched_word_count). No full GloVe file is
    held in memory. The zip archive must be extracted first. Unmatched rows
    stay randomly initialized; the padding vector is exactly zero.
    For 1.2M-vocabulary files, expect a large one-time sequential disk read.
    """
    path = Path(glove_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(len(vocab), embedding_dim, generator=generator) * 0.05
    weights[PAD_ID].zero_()
    matched = set()
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            # split from right so a vocabulary term containing spaces can work
            parts = line.rstrip().rsplit(" ", embedding_dim)
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            if token not in vocab or token in matched:
                continue
            try:
                vector = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            weights[vocab[token]] = torch.tensor(vector, dtype=torch.float32)
            matched.add(token)
    return weights, len(matched)
