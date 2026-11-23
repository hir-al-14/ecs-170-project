"""Generate a fixed prompt panel and compare each output with training text."""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import torch

from local_code.stage_4_code.Method_RNN_Generation import RNNGenerator
from local_code.stage_4_code.Text_Preprocessor import tokenize
from script.stage_4_script.run_generation import ROOT, generate


def compare_training_text(text, training_texts):
    tokens = tokenize(text)
    longest = 0
    closest = None
    exact = False
    for training_text in training_texts:
        reference = tokenize(training_text)
        exact = exact or tokens == reference
        match = SequenceMatcher(None, tokens, reference, autojunk=False).find_longest_match()
        if match.size > longest:
            longest = match.size
            closest = training_text
    return {
        "exact_training_match": exact,
        "longest_training_token_match": longest,
        "matching_training_example": closest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "result/stage_4_result/generation/prepared_data.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    parser.add_argument("--prompts", nargs="+", default=["why did the", "what do you", "what did the"])
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RNNGenerator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    training = json.loads(args.data.read_text())["splits"]["train"]
    samples = []
    for prompt in args.prompts:
        for seed in args.seeds:
            sample = generate(model, checkpoint["vocab"], prompt, checkpoint["context_length"],
                              60, args.temperature, seed, args.top_k)
            sample.update(compare_training_text(sample["text"], training))
            sample["manual_review"] = {"grammar": None, "coherence": None, "notes": "Not yet reviewed"}
            samples.append(sample)
            print(f"\nPrompt: {prompt}; seed: {seed}\n{sample['text']}")
            print(f"Exact training match: {sample['exact_training_match']}; "
                  f"longest matching span: {sample['longest_training_token_match']} tokens")
    report = {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
              "cell": checkpoint["model_config"]["cell_type"], "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
