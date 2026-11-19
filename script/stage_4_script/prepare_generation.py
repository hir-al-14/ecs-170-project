"""Prepare and inspect generation data without training a model."""

import json
from pathlib import Path

from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.Generation_Data import prepare_generation_data
from local_code.stage_4_code.Text_Preprocessor import JokeDataset, collate_jokes


def main():
    prepared = prepare_generation_data(Stage4DatasetLoader().load_generation())
    counts = {}
    for name, jokes in prepared["splits"].items():
        dataset = JokeDataset(jokes, prepared["vocab"], context_length=20)
        counts[name] = len(dataset)
        batch = next(iter(DataLoader(dataset, batch_size=32, collate_fn=collate_jokes)))
        print(f"{name}: {len(jokes)} jokes, {len(dataset)} next-token examples; "
              f"batch shape {tuple(batch[0].shape)}")
    prepared["metadata"]["context_length"] = 20
    prepared["metadata"]["next_token_examples"] = counts
    output_dir = Path(__file__).resolve().parents[2] / "result" / "stage_4_result" / "generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "prepared_data.json"
    output.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    print(json.dumps(prepared["metadata"], indent=2))
    print(f"Saved splits and training vocabulary to {output}")


if __name__ == "__main__":
    main()
