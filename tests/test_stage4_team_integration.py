"""Check evaluation against the shared classifier's three cell types."""

import unittest
from pathlib import Path

import torch

from local_code.stage_4_code.Method_RNN_Classification import RNNClassifier
from script.stage_4_script.run_evaluation import load_classifier
from script.stage_4_script.summarize_results import collect_results


class TeamIntegrationTests(unittest.TestCase):
    def test_saved_classifiers_load_for_each_cell(self):
        tokens = torch.tensor([[2, 4, 5], [3, 2, 0]])
        lengths = torch.tensor([3, 2])
        for cell in ("rnn", "lstm", "gru"):
            with self.subTest(cell=cell):
                original = RNNClassifier(torch.randn(12, 50), hidden_SZ=8, cell_type=cell).eval()
                restored = load_classifier(original.state_dict(), cell)
                with torch.inference_mode():
                    torch.testing.assert_close(original(tokens, lengths), restored(tokens, lengths))

    def test_summary_keeps_generation_and_classification_splits_separate(self):
        results = Path(__file__).resolve().parents[1] / "result" / "stage_4_result"
        classification, generation = collect_results(results)
        self.assertEqual({row["model"] for row in classification}, {"rnn", "lstm", "gru"})
        self.assertTrue(all(row["split"] == "test" for row in classification))
        self.assertTrue(all(row["split"] == "validation" for row in generation))
        for row in classification:
            self.assertTrue((results / row["source"]).is_file())
        for row in generation:
            self.assertTrue((results / row["history"]).is_file())


if __name__ == "__main__":
    unittest.main()
