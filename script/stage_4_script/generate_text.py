"""Generate text from a saved Stage 4 checkpoint without training again."""

import argparse
import json
import math
from pathlib import Path

import torch

from local_code.stage_4_code.Method_RNN_Generation import RNNGenerator
from script.stage_4_script.run_generation import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
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
            args.temperature, args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
