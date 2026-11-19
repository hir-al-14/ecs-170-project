"""Check tuning independence and validation checkpoint selection."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from local_code.stage_4_code.Text_Preprocessor import ReviewDataset, build_vocab
from script.stage_4_script.tune_classification import experiment_settings, train_experiment


class ClassificationTuningTests(unittest.TestCase):
    def test_one_factor_plan(self):
        settings = experiment_settings()
        self.assertEqual(len(settings), 7)
        self.assertEqual(len(set(settings)), 7)
        baseline = settings[0]
        for setting in settings[1:]:
            self.assertEqual(sum(a != b for a, b in zip(baseline, setting)), 1)
        self.assertEqual(len(experiment_settings(full_grid=True)), 27)

    def test_independent_runs_and_best_checkpoint(self):
        vocab = build_vocab(["good movie", "bad movie"])
        dataset = ReviewDataset(["good movie", "bad movie"] * 2, [1, 0] * 2, vocab)
        weights = torch.randn(len(vocab), 50)
        weights[0].zero_()
        original = weights.clone()
        args = SimpleNamespace(seed=42, device="cpu", epochs=2, patience=2)
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for name in ("first", "second"):
                output = Path(directory) / name
                results.append(train_experiment(dataset, dataset, weights, vocab,
                                                (0.0005, 2, 0.4), args, output))
                torch.testing.assert_close(weights, original)
                history = json.loads((output / "history.json").read_text())
                checkpoint = torch.load(output / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["epoch"], history["validation_loss"].index(min(history["validation_loss"])) + 1)
                self.assertEqual(checkpoint["vocab"], vocab)
            self.assertEqual(results[0]["validation_loss"], results[1]["validation_loss"])


if __name__ == "__main__":
    unittest.main()
