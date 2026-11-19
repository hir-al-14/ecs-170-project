"""Reproducible joke-level splits shared by generation experiments."""

import random

from local_code.stage_4_code.Text_Preprocessor import build_vocab, tokenize


def prepare_generation_data(jokes, seed=42, max_vocab_size=10_000):
    """Deduplicate token-equivalent jokes, then split 80/10/remaining percent.

    Deduplication uses the model's tokenizer, so case/HTML-only differences
    cannot put identical model inputs on opposite sides of a split.
    This removes exact token duplicates, not paraphrases or near duplicates.
    Vocabulary is fit exclusively to the training partition.
    """
    seen = set()
    unique = []
    empty_count = 0
    duplicate_count = 0
    for joke in jokes:
        key = tuple(tokenize(joke))
        if not key:
            empty_count += 1
        elif key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
            unique.append(joke)
    if len(unique) < 10:
        raise ValueError("At least 10 distinct nonempty jokes are required")

    random.Random(seed).shuffle(unique)
    train_end = int(len(unique) * 0.8)
    validation_end = train_end + int(len(unique) * 0.1)
    splits = {
        "train": unique[:train_end],
        "validation": unique[train_end:validation_end],
        "test": unique[validation_end:],
    }
    vocab = build_vocab(splits["train"], max_vocab_size=max_vocab_size)
    return {
        "splits": splits,
        "vocab": vocab,
        "metadata": {
            "seed": seed,
            "source_jokes": len(jokes),
            "unique_jokes": len(unique),
            "duplicates_removed": duplicate_count,
            "empty_token_sequences_removed": empty_count,
            "split_sizes": {name: len(texts) for name, texts in splits.items()},
            "max_vocab_size": max_vocab_size,
            "vocab_size": len(vocab),
        },
    }
