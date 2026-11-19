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
from local_code.stage_4_code.Text_Preprocessor import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID, JokeDataset, collate_jokes,
    detokenize, encode_prompt, tokenize,
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
            logits = model(contexts, lengths)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; try a smaller learning rate")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
    if total == 0:
        raise ValueError("No next-token examples in this split")
    return total_loss / total


def generate(model, vocab, prompt, context_length, max_tokens, temperature, seed):
    ids = encode_prompt(prompt, vocab)
    if UNK_ID in ids:
        raise ValueError("Choose three prompt words present in the training vocabulary")
    device = next(model.parameters()).device
    random = torch.Generator(device=device).manual_seed(seed)
    inverse = {index: token for token, index in vocab.items()}
    model.eval()
    ended = False
    with torch.inference_mode():
        for _ in range(max_tokens):
            context = ids[-context_length:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x, torch.tensor([len(context)]))[0] / temperature
            logits[[PAD_ID, UNK_ID, BOS_ID]] = -torch.inf
            next_id = torch.multinomial(logits.softmax(dim=0), 1, generator=random).item()
            ids.append(next_id)
            if next_id == EOS_ID:
                ended = True
                break
    return {"prompt": prompt, "text": detokenize([inverse[i] for i in ids]),
            "ended_with_eos": ended, "seed": seed, "temperature": temperature}


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
    raw = args.data.read_bytes()
    prepared = json.loads(raw)
    vocab = prepared["vocab"]
    prompt_ids = encode_prompt(args.prompt, vocab)
    if UNK_ID in prompt_ids:
        parser.error("Prompt words must be present in the training vocabulary")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_config = dict(vocab_size=len(vocab), embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, dropout=args.dropout, cell_type=args.cell)
    model = RNNGenerator(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loaders = {}
    for name in ("train", "validation"):
        dataset = JokeDataset(prepared["splits"][name], vocab, args.context_length)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                                   collate_fn=collate_jokes,
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
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        validation_loss = run_epoch(model, loaders["validation"], device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            torch.save({"model_state": model.state_dict(), "model_config": model_config,
                        "vocab": vocab, "context_length": args.context_length,
                        "epoch": epoch, "validation_loss": best_loss,
                        "experiment": config}, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, validation loss {validation_loss:.4f}", flush=True)
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    results = {"best_epoch": best_epoch, "validation_loss": best_loss,
               "validation_perplexity": math.exp(min(best_loss, 700))}
    if args.evaluate_test:
        test = JokeDataset(prepared["splits"]["test"], vocab, args.context_length)
        loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_jokes)
        test_loss = run_epoch(model, loader, device)
        results.update(test_loss=test_loss, test_perplexity=math.exp(min(test_loss, 700)))
    sample = generate(model, vocab, args.prompt, args.context_length,
                      args.max_tokens, args.temperature, args.seed)
    training_tokens = {tuple(tokenize(text)) for text in prepared["splits"]["train"]}
    sample["exact_training_match"] = tuple(tokenize(sample["text"])) in training_tokens
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    (args.output / "sample.json").write_text(json.dumps(sample, indent=2))
    print(json.dumps(results, indent=2))
    print(sample["text"])


if __name__ == "__main__":
    main()
