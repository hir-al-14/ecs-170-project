"""Train a recurrent generator and sample a continuation from three words."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from local_code.stage_4_code.Method_RNN_Generation import RNNGenerator
from local_code.stage_4_code.Config import get_glove_path
from local_code.stage_4_code.Generation_Sequences import JokeSequenceDataset, collate_sequences
from local_code.stage_4_code.Text_Preprocessor import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID, JokeDataset, collate_jokes,
    detokenize, encode_prompt, tokenize, load_glove_matrix, build_vocab,
)

ROOT = Path(__file__).resolve().parents[2]


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    with torch.set_grad_enabled(training):
        for contexts, lengths, targets in loader:
            contexts, targets = contexts.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            if targets.ndim == 2:
                logits = model.forward_sequence(contexts, lengths)
                loss = nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_ID,
                )
                count = (targets != PAD_ID).sum().item()
            else:
                logits = model(contexts, lengths)
                loss = nn.functional.cross_entropy(logits, targets)
                count = targets.size(0)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; try a smaller learning rate")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if model.tie_embeddings:
                    with torch.no_grad():
                        model.embedding.weight[PAD_ID].zero_()
            total_loss += loss.item() * count
            total += count
    if total == 0:
        raise ValueError("No next-token examples in this split")
    return total_loss / total


def generate(model, vocab, prompt, context_length, max_tokens, temperature, seed, top_k=0):
    if context_length < 1 or max_tokens < 1:
        raise ValueError("Context length and maximum tokens must be positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    ids = encode_prompt(prompt, vocab)
    if UNK_ID in ids:
        raise ValueError("Choose three prompt words present in the training vocabulary")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    id_to_token = {index: token for token, index in vocab.items()}
    model.eval()
    ended = False
    with torch.inference_mode():
        for _ in range(max_tokens):
            context = ids[-context_length:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x, torch.tensor([len(context)]))[0] / temperature
            logits[[PAD_ID, UNK_ID, BOS_ID]] = -torch.inf
            if top_k:
                values, indices = logits.topk(min(top_k, len(vocab) - 3))
                filtered = torch.full_like(logits, -torch.inf)
                filtered[indices] = values
                logits = filtered
            next_id = torch.multinomial(
                logits.softmax(dim=0), 1, generator=generator
            ).item()
            ids.append(next_id)
            if next_id == EOS_ID:
                ended = True
                break
    return {
        "prompt": prompt,
        "text": detokenize([id_to_token[i] for i in ids]),
        "ended_with_eos": ended,
        "seed": seed,
        "temperature": temperature,
        "top_k": top_k,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "result/stage_4_result/generation/prepared_data.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this experiment")
    parser.add_argument("--cell", choices=["rnn", "lstm", "gru"], default="rnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="what did the")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--evaluate-test", action="store_true", help="Only use after selecting settings on validation data")
    parser.add_argument("--glove", action="store_true", help="Initialize embeddings with the supplied GloVe file")
    parser.add_argument("--top-k", type=int, default=0, help="Sample from the k most likely tokens; 0 uses all tokens")
    parser.add_argument("--sequence-training", action="store_true", help="Train all positions in each example together")
    parser.add_argument("--tie-embeddings", action="store_true", help="Share input and output word vectors")
    parser.add_argument("--freeze-embeddings", action="store_true", help="Keep pretrained embeddings fixed")
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many unimproved epochs; 0 disables")
    parser.add_argument("--save-every", type=int, default=0, help="Also retain a checkpoint every n epochs")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum training frequency for a vocabulary word")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.embedding_dim,
           args.context_length, args.max_tokens) < 1:
        parser.error("Epochs, batch size, dimensions and lengths must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive")
    if not 0 <= args.dropout < 1:
        parser.error("Dropout must be between 0 (inclusive) and 1 (exclusive)")
    if args.top_k < 0:
        parser.error("top-k must be nonnegative")
    if not 0 <= args.embedding_dropout < 1:
        parser.error("Embedding dropout must be between 0 and 1")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    if args.patience < 0 or args.save_every < 0:
        parser.error("Patience and checkpoint interval must be nonnegative")
    if args.min_freq < 1:
        parser.error("Minimum word frequency must be positive")
    if args.freeze_embeddings and not args.glove:
        parser.error("Frozen embeddings require GloVe initialization")
    raw = args.data.read_bytes()
    prepared = json.loads(raw)
    vocab = prepared["vocab"]
    if args.min_freq > 1:
        vocab = build_vocab(prepared["splits"]["train"], max_vocab_size=len(vocab), min_freq=args.min_freq)
    prompt_ids = encode_prompt(args.prompt, vocab)
    if UNK_ID in prompt_ids:
        parser.error("Prompt words must be present in the training vocabulary")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_config = dict(vocab_size=len(vocab), embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, dropout=args.dropout, cell_type=args.cell,
                        tie_embeddings=args.tie_embeddings, embedding_dropout=args.embedding_dropout)
    model = RNNGenerator(**model_config).to(device)
    if args.glove:
        print("Loading GloVe embeddings...", flush=True)
        weights, matched = load_glove_matrix(get_glove_path(args.embedding_dim), vocab,
                                             args.embedding_dim, seed=args.seed)
        with torch.no_grad():
            model.embedding.weight.copy_(weights.to(device))
        print(f"GloVe coverage: {matched}/{len(vocab)} vocabulary entries", flush=True)
    if args.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset_class = JokeSequenceDataset if args.sequence_training else JokeDataset
    collate = collate_sequences if args.sequence_training else collate_jokes
    loaders = {}
    for name in ("train", "validation"):
        dataset = dataset_class(prepared["splits"][name], vocab, args.context_length)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                                   collate_fn=collate,
                                   generator=torch.Generator().manual_seed(args.seed))
    # A fresh directory prevents experiments from overwriting each other.
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config["data_sha256"] = hashlib.sha256(raw).hexdigest()
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    history = {"train_loss": [], "validation_loss": []}
    best_loss = math.inf
    best_epoch = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        validation_loss = run_epoch(model, loaders["validation"], device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "model_config": model_config,
                        "vocab": vocab, "context_length": args.context_length,
                        "epoch": epoch, "validation_loss": best_loss,
                        "experiment": config}, args.output / "best.pt")
        else:
            stale_epochs += 1
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state": model.state_dict(), "model_config": model_config,
                        "vocab": vocab, "context_length": args.context_length,
                        "epoch": epoch, "validation_loss": validation_loss,
                        "experiment": config}, args.output / f"epoch_{epoch:03d}.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, validation loss {validation_loss:.4f}", flush=True)
        if args.patience and stale_epochs >= args.patience:
            print("Stopped after validation loss stopped improving.", flush=True)
            break
    # Keep the final model as well as the minimum-loss model so qualitative
    # comparisons do not require retraining or silently replacing best.pt.
    torch.save({"model_state": model.state_dict(), "model_config": model_config,
                "vocab": vocab, "context_length": args.context_length,
                "epoch": epoch, "validation_loss": validation_loss,
                "experiment": config}, args.output / "last.pt")
    final_sample = generate(model, vocab, args.prompt, args.context_length,
                            args.max_tokens, args.temperature, args.seed, args.top_k)
    (args.output / "sample_last.json").write_text(json.dumps(final_sample, indent=2))
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    results = {"best_epoch": best_epoch, "validation_loss": best_loss,
               "validation_perplexity": math.exp(min(best_loss, 700))}
    if args.evaluate_test:
        test = dataset_class(prepared["splits"]["test"], vocab, args.context_length)
        loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)
        test_loss = run_epoch(model, loader, device)
        results.update(test_loss=test_loss, test_perplexity=math.exp(min(test_loss, 700)))
    sample = generate(model, vocab, args.prompt, args.context_length,
                      args.max_tokens, args.temperature, args.seed, args.top_k)
    training_tokens = {tuple(tokenize(text)) for text in prepared["splits"]["train"]}
    sample["exact_training_match"] = tuple(tokenize(sample["text"])) in training_tokens
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    (args.output / "sample.json").write_text(json.dumps(sample, indent=2))
    print(json.dumps(results, indent=2))
    print(sample["text"])


if __name__ == "__main__":
    main()
