"""Collect team results for the report without mixing test and validation scores."""

import argparse
import csv
import json

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "result" / "stage_4_result"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_results(result_dir):
    classification = []
    for cell in ("rnn", "lstm", "gru"):
        path = result_dir / f"{cell}_classification_metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        classification.append({
            "model": cell, "split": "test", "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"],
            "source": str(path.relative_to(result_dir)),
        })

    generation = []
    for path in sorted((result_dir / "generation").glob("*/metrics.json")):
        folder = path.parent
        metrics = json.loads(path.read_text())
        config = json.loads((folder / "config.json").read_text())
        sample = json.loads((folder / "sample.json").read_text())
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        generation.append({
            "experiment": folder.name, "model": config["cell"], "split": "validation",
            "vocab_size": len(checkpoint["vocab"]), "min_freq": config.get("min_freq", 1),
            "best_epoch": metrics["best_epoch"],
            "loss": metrics["validation_loss"],
            "perplexity": metrics["validation_perplexity"],
            "prompt": sample["prompt"], "sample": sample["text"],
            "manual_review": "required; loss does not establish readability",
            "history": str((folder / "history.json").relative_to(result_dir)),
        })
    return classification, generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=RESULTS / "summary")
    args = parser.parse_args()
    classification, generation = collect_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "classification_test.csv", classification)
    write_csv(args.output / "generation_validation.csv", generation)
    for row in classification:
        print(f"{row['model'].upper():4} test accuracy {row['accuracy']:.2%}; "
              f"macro F1 {row['macro_f1']:.2%}")
    for row in generation:
        print(f"\n{row['experiment']} (manual readability review required):\n{row['sample']}")
    print(f"\nSaved comparison tables to {args.output}")
    print("Classification tuning summaries remain separate under classification_tuning/.")
    print("Generation loss/perplexity comparisons require the same vocabulary and preprocessing.")


if __name__ == "__main__":
    main()
