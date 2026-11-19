"""Evaluate the saved Stage 4 RNN classifier without retraining."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.evaluate_classification import (
    evaluate_classification,
)
from local_code.stage_4_code.Method_RNN_Classification import RNNClassifier
from local_code.stage_4_code.Text_Preprocessor import (
    ReviewDataset,
    build_vocab,
    collate_reviews,
)


def main():
    project_root = Path(__file__).resolve().parents[2]
    result_dir = project_root / "result" / "stage_4_result"
    checkpoint_path = result_dir / "rnn_classification.pt"

    # CPU works even when the checkpoint was trained on CUDA.
    device = torch.device("cpu")

    data = Stage4DatasetLoader().load_classification()

    # Reproduce the vocabulary used by the existing training script.
    vocab = build_vocab(
        data["train"]["X"],
        max_vocab_size=15_000,
    )

    test_dataset = ReviewDataset(
        data["test"]["X"],
        data["test"]["y"],
        vocab,
        max_length=200,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_reviews,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # The checkpoint already contains the trained embedding weights.
    # There is no need to read the large GloVe file again.
    model = RNNClassifier(
        state_dict["embedding.weight"],
        hidden_SZ=state_dict["rnn.weight_hh_l0"].shape[1],
        dropout=0.4,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []

    print(f"Evaluating {len(test_dataset):,} test reviews...")

    with torch.inference_mode():
        for token_ids, lengths, labels in test_loader:
            logits = model(token_ids.to(device), lengths)
            predictions = logits.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    metrics = evaluate_classification(y_true, y_pred)

    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = result_dir / "rnn_classification_metrics.json"
    predictions_path = result_dir / "rnn_classification_predictions.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    predictions_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_path.name,
                "label_mapping": {"0": "negative", "1": "positive"},
                "y_true": y_true,
                "y_pred": y_pred,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        print(f"{name}: {metrics[name]:.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    for row in metrics["confusion_matrix"]:
        print(row)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
