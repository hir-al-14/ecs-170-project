"""Regression tests for padding, optimization, and checkpoint sampling."""

import io
import unittest

import torch
from torch.utils.data import DataLoader

from local_code.stage_4_code.Method_RNN_Generation import RNNGenerator
from local_code.stage_4_code.Text_Preprocessor import build_vocab, JokeDataset, collate_jokes
from script.stage_4_script.run_generation import generate, run_epoch


class GenerationModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocab = build_vocab(["what did the chicken say", "what did the dog say"])
        self.dataset = JokeDataset(["what did the chicken say"], self.vocab)

    def test_padding_and_checkpoint_roundtrip_for_each_cell(self):
        x = torch.tensor([[2, 4, 5], [2, 5, 0]])
        lengths = torch.tensor([3, 2])
        for cell in ("rnn", "lstm", "gru"):
            with self.subTest(cell=cell):
                config = dict(vocab_size=len(self.vocab), hidden_dim=8, cell_type=cell)
                model = RNNGenerator(**config).eval()
                expected = model(x, lengths)
                padded = torch.nn.functional.pad(x, (0, 4))
                torch.testing.assert_close(expected, model(padded, lengths))
                torch.testing.assert_close(expected[1], model(x[1:2, :2], lengths[1:2])[0])
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = RNNGenerator(**config).eval()
                restored.load_state_dict(torch.load(checkpoint, weights_only=True))
                torch.testing.assert_close(expected, restored(x, lengths))

    def test_training_updates_weights_but_validation_does_not(self):
        model = RNNGenerator(len(self.vocab), hidden_dim=8)
        loader = DataLoader(self.dataset, batch_size=4, collate_fn=collate_jokes)
        before = model.output.weight.detach().clone()
        loss = run_epoch(model, loader, torch.device("cpu"), torch.optim.Adam(model.parameters()))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.output.weight))
        before = model.output.weight.detach().clone()
        run_epoch(model, loader, torch.device("cpu"))
        torch.testing.assert_close(before, model.output.weight)

    def test_top_k_and_training_overlap(self):
        from script.stage_4_script.review_generation import compare_training_text
        model = RNNGenerator(len(self.vocab), hidden_dim=8)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[0] = 200  # Padding must never be sampled.
            model.output.bias[3] = 100
        sample = generate(model, self.vocab, "what did the", 20, 5, 0.8, 42, top_k=1)
        self.assertTrue(sample["ended_with_eos"])
        self.assertEqual(sample["text"], "what did the")
        with self.assertRaises(ValueError):
            generate(model, self.vocab, "what did the", 20, 5, 0.8, 42, top_k=-1)
        result = compare_training_text("The cat sat.", ["The cat sat.", "Another example."])
        self.assertTrue(result["exact_training_match"])
        self.assertEqual(result["longest_training_token_match"], 4)
        partial = compare_training_text("The cat ran.", ["The cat sat."])
        self.assertFalse(partial["exact_training_match"])
        self.assertEqual(partial["longest_training_token_match"], 2)

    def test_sampling_is_repeatable_and_stops_at_eos(self):
        model = RNNGenerator(len(self.vocab), hidden_dim=8)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-1000)
            model.output.bias[3] = 1000
        args = (model, self.vocab, "what did the", 20, 5, 0.8, 42)
        sample = generate(*args)
        self.assertEqual(sample, generate(*args))
        self.assertTrue(sample["ended_with_eos"])
        self.assertEqual(sample["text"], "what did the")
        with self.assertRaises(ValueError):
            generate(model, self.vocab, "missing word here", 20, 5, 0.8, 42)


if __name__ == "__main__":
    unittest.main()
