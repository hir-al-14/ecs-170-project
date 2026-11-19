"""Evaluate a selected tuning checkpoint on the official test split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.Method_RNN_Classification import RNNClassifier
from local_code.stage_4_code.Text_Preprocessor import ReviewDataset, collate_reviews
from script.stage_4_script.tune_classification import run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state = checkpoint["model_state"]
    model = RNNClassifier(state["embedding.weight"], hidden_SZ=config["hidden_dim"],
                          dropout=config["dropout"])
    model.load_state_dict(state)
    data = Stage4DatasetLoader().load_classification()["test"]
    dataset = ReviewDataset(data["X"], data["y"], checkpoint["vocab"], config["max_length"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_reviews)
    metrics = run_epoch(model, loader, torch.device("cpu"))
    output = args.checkpoint.parent / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved test results to {output}")


if __name__ == "__main__":
    main()
