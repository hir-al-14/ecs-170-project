"""Generate text from a saved Stage 4 checkpoint without training again."""

import argparse
import json
import math
from pathlib import Path

import torch

from local_code.stage_4_code.Method_RNN_Generation import RNNGenerator
from script.stage_4_script.run_generation import generate
from script.stage_4_script.run_generation import ROOT
from script.stage_4_script.review_generation import compare_training_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--plain", action="store_true", help="Print a readable terminal demonstration")
    parser.add_argument("--compare-training", action="store_true", help="Check overlap with the prepared training examples")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive")
    if args.max_tokens < 1:
        parser.error("Maximum tokens must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RNNGenerator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    try:
        sample = generate(
            model, checkpoint["vocab"], args.prompt,
            checkpoint["context_length"], args.max_tokens,
            args.temperature, args.seed, args.top_k,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.compare_training:
        data = json.loads((ROOT / "result/stage_4_result/generation/prepared_data.json").read_text())
        sample.update(compare_training_text(sample["text"], data["splits"]["train"]))
    if args.plain:
        print("Stage 4 — Text generation")
        print(f"Model: {checkpoint['model_config']['cell_type'].upper()}")
        print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")
        print(f"Prompt: {sample['prompt']}")
        print(f"Temperature: {args.temperature}; top-k: {args.top_k}; seed: {args.seed}")
        print(f"\nGenerated text:\n{sample['text']}")
        if args.compare_training:
            print(f"\nExact training-text match: {sample['exact_training_match']}")
            print(f"Longest matching span: {sample['longest_training_token_match']} tokens")
    else:
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
