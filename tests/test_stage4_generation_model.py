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

    def test_beam_search_uses_model_and_preserves_prompt(self):
        from local_code.stage_4_code.Generation_Decoder import beam_generate
        model = RNNGenerator(len(self.vocab), hidden_dim=8)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-1000)
            model.output.bias[3] = 1000
        sample = beam_generate(model, self.vocab, "what did the", 64, 8, beam_size=3)
        self.assertEqual(sample["text"], "what did the")
        self.assertTrue(sample["ended_with_eos"])
        self.assertEqual(sample, beam_generate(model, self.vocab, "what did the", 64, 8, 3))
        with self.assertRaises(ValueError):
            beam_generate(model, self.vocab, "what did the", 64, 8, 0)

    def test_rare_words_supply_unknown_training_targets(self):
        from local_code.stage_4_code.Generation_Sequences import JokeSequenceDataset
        texts = ["what did the chicken say", "what did the dog say"]
        vocab = build_vocab(texts, min_freq=2)
        dataset = JokeSequenceDataset(texts, vocab, 64)
        self.assertEqual((dataset[0][1] == 1).sum().item(), 1)
        self.assertEqual((dataset[1][1] == 1).sum().item(), 1)

    def test_sequence_training_matches_rolling_context_loss(self):
        from local_code.stage_4_code.Generation_Sequences import JokeSequenceDataset, collate_sequences
        jokes = ["what did the chicken say", "what did the dog say"]
        model = RNNGenerator(len(self.vocab), hidden_dim=8, dropout=0).eval()
        windows = DataLoader(JokeDataset(jokes, self.vocab, 64), batch_size=4, collate_fn=collate_jokes)
        sequences = DataLoader(JokeSequenceDataset(jokes, self.vocab, 64), batch_size=2, collate_fn=collate_sequences)
        self.assertAlmostEqual(run_epoch(model, windows, torch.device("cpu")),
                               run_epoch(model, sequences, torch.device("cpu")), places=5)

    def test_tied_embeddings_and_sequence_logits(self):
        from local_code.stage_4_code.Generation_Sequences import JokeSequenceDataset, collate_sequences
        config = dict(vocab_size=len(self.vocab), hidden_dim=8, tie_embeddings=True)
        model = RNNGenerator(**config).eval()
        self.assertIs(model.embedding.weight, model.output.weight)
        x, lengths, targets = collate_sequences([JokeSequenceDataset(["what did the dog say"], self.vocab, 64)[0]])
        with torch.inference_mode():
            logits = model.forward_sequence(x, lengths)
            for position in range(x.size(1)):
                expected = model(x[:, :position + 1], torch.tensor([position + 1]))
                torch.testing.assert_close(logits[:, position], expected)
        restored = RNNGenerator(**config)
        restored.load_state_dict(model.state_dict())
        self.assertIs(restored.embedding.weight, restored.output.weight)
        loader = DataLoader(JokeSequenceDataset(["what did the dog say"], self.vocab, 64),
                            batch_size=1, collate_fn=collate_sequences)
        run_epoch(restored, loader, torch.device("cpu"), torch.optim.Adam(restored.parameters()))
        self.assertTrue(torch.equal(restored.embedding.weight[0], torch.zeros(50)))

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
