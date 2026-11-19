"""Compare classifier settings on a fixed validation split, not the test set."""

import argparse
import csv
import itertools
import json
import math
import shutil
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from local_code.stage_4_code.Config import get_glove_path
from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.Method_RNN_Classification import RNNClassifier
from local_code.stage_4_code.Text_Preprocessor import (
    ReviewDataset, build_vocab, collate_reviews, load_glove_matrix,
)
from local_code.stage_4_code.evaluate_classification import evaluate_classification

ROOT = Path(__file__).resolve().parents[2]
BASELINE = (0.0005, 32, 0.4)


def experiment_settings(full_grid=False):
    learning_rates = [0.0001, 0.0005, 0.001]
    batch_sizes = [16, 32, 64]
    dropouts = [0.2, 0.4, 0.6]
    if full_grid:
        return list(itertools.product(learning_rates, batch_sizes, dropouts))
    # One factor at a time gives interpretable comparisons with seven runs.
    settings = [BASELINE]
    settings += [(lr, 32, 0.4) for lr in learning_rates if lr != BASELINE[0]]
    settings += [(0.0005, batch, 0.4) for batch in batch_sizes if batch != 32]
    settings += [(0.0005, 32, dropout) for dropout in dropouts if dropout != 0.4]
    return settings


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    truth, predicted = [], []
    with torch.set_grad_enabled(training):
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(tokens, lengths)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite classification loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            truth.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if not truth:
        raise ValueError("Cannot evaluate an empty split")
    metrics = evaluate_classification(truth, predicted)
    metrics["loss"] = total_loss / len(truth)
    return metrics


def train_experiment(train, validation, weights, vocab, setting, args, output):
    lr, batch_size, dropout = setting
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # from_pretrained shares its input tensor; clone to keep every run's
    # starting embeddings identical even after another run fine-tunes them.
    model = RNNClassifier(weights.clone(), hidden_SZ=128, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=collate_reviews,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate_reviews)
    output.mkdir()
    history = {"train_loss": [], "validation_loss": [], "train_acc": [], "validation_acc": []}
    best_loss = math.inf
    stale_epochs = 0
    config = {"learning_rate": lr, "batch_size": batch_size, "dropout": dropout,
              "hidden_dim": 128, "embedding_dim": 50, "max_length": 200,
              "seed": args.seed, "max_epochs": args.epochs, "patience": args.patience}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer)
        metrics = run_epoch(model, validation_loader, device)
        history["train_loss"].append(training["loss"])
        history["validation_loss"].append(metrics["loss"])
        history["train_acc"].append(training["accuracy"])
        history["validation_acc"].append(metrics["accuracy"])
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            stale_epochs = 0
            best_metrics = {**metrics, "epoch": epoch}
            torch.save({"model_state": model.state_dict(), "vocab": vocab,
                        "config": config, "epoch": epoch}, output / "best.pt")
            (output / "validation_metrics.json").write_text(json.dumps(best_metrics, indent=2))
        else:
            stale_epochs += 1
        print(f"{output.name} epoch {epoch}: train loss {training['loss']:.4f}, "
              f"validation loss {metrics['loss']:.4f}, accuracy {metrics['accuracy']:.4f}", flush=True)
        if stale_epochs >= args.patience:
            break
    return {"run": output.name, **config, "best_epoch": best_metrics["epoch"],
            "validation_loss": best_metrics["loss"],
            "validation_accuracy": best_metrics["accuracy"],
            "validation_macro_f1": best_metrics["macro_f1"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--plan", choices=["baseline", "one-factor", "grid"], default="one-factor")
    parser.add_argument("--baseline-from", type=Path, help="Reuse a completed baseline directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("Epochs and patience must be positive")
    if args.output.exists():
        parser.error("Choose a new output directory to preserve previous experiments")
    data = Stage4DatasetLoader().load_classification()["train"]
    indices = list(range(len(data["X"])))
    train_ids, validation_ids = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=data["y"],
    )
    train_texts = [data["X"][i] for i in train_ids]
    vocab = build_vocab(train_texts, max_vocab_size=15_000)
    train = ReviewDataset(train_texts, [data["y"][i] for i in train_ids], vocab, 200)
    validation = ReviewDataset([data["X"][i] for i in validation_ids],
                               [data["y"][i] for i in validation_ids], vocab, 200)
    weights, matched = load_glove_matrix(get_glove_path(50), vocab, 50, seed=args.seed)
    args.output.mkdir(parents=True)
    split = {"seed": args.seed, "train_indices": train_ids, "validation_indices": validation_ids,
             "index_order": "Dataset_Loader sorted filenames, negative then positive",
             "vocab_size": len(vocab), "glove_matches": matched}
    (args.output / "split.json").write_text(json.dumps(split, indent=2))
    baseline = None
    if args.baseline_from:
        baseline = json.loads((args.baseline_from / "best_run.json").read_text())
        old_split = json.loads((args.baseline_from / "split.json").read_text())
        checkpoint = torch.load(args.baseline_from / baseline["run"] / "best.pt",
                                map_location="cpu", weights_only=True)
        if old_split != split or checkpoint["vocab"] != vocab:
            parser.error("Baseline split or vocabulary does not match this experiment")
        expected = (BASELINE[0], BASELINE[1], BASELINE[2], args.seed, args.epochs, args.patience)
        actual = tuple(baseline[key] for key in
                       ("learning_rate", "batch_size", "dropout", "seed", "max_epochs", "patience"))
        if actual != expected or args.plan == "grid":
            parser.error("Baseline settings must match; reuse supports baseline or one-factor plans")
    settings = [BASELINE] if args.plan == "baseline" else experiment_settings(args.plan == "grid")
    rows = []
    for index, setting in enumerate(settings, start=1):
        if baseline is not None and setting == BASELINE:
            shutil.copytree(args.baseline_from / baseline["run"], args.output / "run_01")
            result = {**baseline, "run": "run_01"}
            print("Reusing completed baseline", flush=True)
        else:
            result = train_experiment(train, validation, weights, vocab, setting, args,
                                      args.output / f"run_{index:02d}")
        rows.append(result)
        with (args.output / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    best = min(rows, key=lambda row: row["validation_loss"])
    (args.output / "best_run.json").write_text(json.dumps(best, indent=2))
    print("Best validation run:", json.dumps(best, indent=2))
    print("The test set was not used to select settings.")


if __name__ == "__main__":
    main()
