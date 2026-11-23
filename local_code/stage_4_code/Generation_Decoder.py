"""Beam decoding with a repeated-trigram constraint."""

import math

import torch

from local_code.stage_4_code.Text_Preprocessor import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID, detokenize, encode_prompt,
)


def beam_generate(model, vocab, prompt, context_length, max_tokens, beam_size=5):
    if beam_size < 1 or max_tokens < 1 or context_length < 1:
        raise ValueError("Beam size and token limits must be positive")
    start = encode_prompt(prompt, vocab)
    if UNK_ID in start:
        raise ValueError("Choose three prompt words present in the training vocabulary")
    device = next(model.parameters()).device
    beams = [(start, 0.0)]
    completed = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max_tokens):
            candidates = []
            for ids, score in beams:
                context = ids[-context_length:]
                x = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(x, torch.tensor([len(context)]))[0]
                logits[[PAD_ID, UNK_ID, BOS_ID]] = -torch.inf
                # Block a token if it would repeat an earlier trigram.
                prefix = tuple(ids[-2:])
                for index in range(len(ids) - 2):
                    if tuple(ids[index:index + 2]) == prefix:
                        logits[ids[index + 2]] = -torch.inf
                values, indices = logits.log_softmax(dim=0).topk(min(beam_size, len(vocab)))
                for value, token in zip(values.tolist(), indices.tolist()):
                    if not math.isfinite(value):
                        continue
                    candidate = (ids + [token], score + value)
                    if token == EOS_ID:
                        completed.append(candidate)
                    else:
                        candidates.append(candidate)
            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
            if not beams:
                break
    choices = completed or beams
    if not choices:
        raise RuntimeError("No valid continuation found")
    # Length normalization reduces the preference for very short completions.
    ids, _ = max(choices, key=lambda item: item[1] / ((len(item[0]) - len(start) + 5) / 6) ** 0.7)
    inverse = {index: token for token, index in vocab.items()}
    return {"prompt": prompt, "text": detokenize([inverse[index] for index in ids]),
            "ended_with_eos": ids[-1] == EOS_ID, "decoding": "beam",
            "beam_size": beam_size, "no_repeat_ngram_size": 3, "length_penalty": 0.7}
