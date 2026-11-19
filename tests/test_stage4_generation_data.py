import tempfile
import unittest
from pathlib import Path

from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.Generation_Data import prepare_generation_data
from local_code.stage_4_code.Text_Preprocessor import tokenize


class GenerationDataTests(unittest.TestCase):
    def test_csv_preserves_joke_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_text('ID,Joke\n1,"Hello, world"\n2,"Line one\nline two"\n')
            loader = Stage4DatasetLoader()
            loader.generation_dir = path
            self.assertEqual(loader.load_generation(), ["Hello, world", "Line one\nline two"])

    def test_reproducible_disjoint_splits_and_training_only_vocab(self):
        jokes = [f"unique word{i}" for i in range(30)]
        prepared = prepare_generation_data(jokes + [jokes[0].upper()])
        self.assertEqual(prepared, prepare_generation_data(jokes + [jokes[0].upper()]))
        self.assertEqual(prepared["metadata"]["duplicates_removed"], 1)
        splits = prepared["splits"]
        keys = {name: {tuple(tokenize(j)) for j in texts} for name, texts in splits.items()}
        self.assertFalse(keys["train"] & keys["validation"])
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["validation"] & keys["test"])
        training_tokens = {token for joke in splits["train"] for token in tokenize(joke)}
        self.assertTrue(set(prepared["vocab"]) - {"<PAD>", "<UNK>", "<BOS>", "<EOS>"} <= training_tokens)


if __name__ == "__main__":
    unittest.main()
