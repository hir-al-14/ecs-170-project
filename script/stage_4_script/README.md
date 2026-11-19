# Stage 4 classification evaluation

Run from the repository root with the existing project environment:

```bash
.venv/bin/python -m script.stage_4_script.run_evaluation
```

Loads the existing final RNN checkpoint on CPU without retraining or reading
GloVe. Rebuilds its vocabulary from the same training reviews using the original
15,000-token limit; uses the original 200-token review limit. The checkpoint
contains weights only, so changing the training data or tokenizer would invalidate
this reconstruction. Hidden size is inferred from checkpoint weights.

Outputs under `result/stage_4_result/`:
- `rnn_classification_metrics.json`: accuracy, macro precision/recall/F1,
  per-class scores, confusion matrix (actual rows, predicted columns; order 0, 1).
- `rnn_classification_predictions.json`: test labels and predictions in loader order.

`run_classification` also uses the shared metrics function and writes its final
checkpoint, history, and metrics to this result directory. Running that training
script overwrites the baseline files. Use the evaluation command above to evaluate
without training. Legacy module entry points under `local_code.stage_4_code`
remain supported. LSTM/GRU can reuse the metrics function with their predictions;
the checkpoint-loading script currently supports the existing RNN architecture.

Check metric calculations:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_stage4_evaluation.py'
```

## Generation data

```bash
.venv/bin/python -m script.stage_4_script.prepare_generation
```

Reads the instructor's extensionless `ID,Joke` CSV with proper CSV quoting.
Removes duplicate token sequences before splitting jokes with seed 42 into
80% training, 10% validation, and the remainder test (integer rounding).
Builds vocabulary exclusively from training jokes. Splits are made before
creating rolling next-token windows, avoiding overlap across partitions.
Paraphrases and near duplicates are not removed.

Saves actual splits, vocabulary, and preparation statistics in
`result/stage_4_result/generation/prepared_data.json`. Future RNN/LSTM/GRU
experiments should use these same splits. The command verifies all three
partitions with the existing `JokeDataset` and `collate_jokes` interfaces.
This step prepares data only; it does not train a generator or evaluate quality.

## Train a generator

```bash
.venv/bin/python -m script.stage_4_script.run_generation --output result/stage_4_result/generation/rnn_baseline --epochs 10
```

Uses the prepared data above. Each run requires a new output directory.
`--cell rnn|lstm|gru`, `--learning-rate`, `--batch-size`, and `--dropout`
allow controlled comparisons using the same data. Default embeddings are
randomly initialized and trained; this generator does not load GloVe.

Each run saves `config.json`, `history.json` (training/validation losses),
`best.pt` (selected by validation loss, including vocabulary and model settings),
`metrics.json` (validation loss/perplexity), and `sample.json`.
The three-word prompt defaults to `what did the`; change it with `--prompt`.
Sampling suppresses PAD, UNK, and BOS and ends at EOS or the token limit.
Exact training-match detection is a limited copying check, not a quality score.
Read samples to assess coherence, grammar, repetition, and joke structure.

Only add `--evaluate-test` for final evaluation after selecting settings on
validation data. Test loss/perplexity uses the training vocabulary, including
UNK for unseen words. Keep tokenization and vocabulary fixed across comparisons.

Generate another sample from a saved checkpoint without training:

```bash
.venv/bin/python -m script.stage_4_script.generate_text result/stage_4_result/generation/rnn_baseline/best.pt --prompt "what did the"
```

Run all Stage 4 checks:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_stage4*.py'
```

## Classification tuning

Start with a validation baseline (uses the existing classifier architecture):

```bash
.venv/bin/python -m script.stage_4_script.tune_classification --output result/stage_4_result/classification_tuning/baseline_01 --plan baseline
```

Run seven experiments varying one of learning rate, batch size, or dropout:

```bash
.venv/bin/python -m script.stage_4_script.tune_classification --output result/stage_4_result/classification_tuning/sweep_01
```

This can take substantial time on CPU. `--device cuda` uses CUDA when available.
`--plan grid` runs all 27 combinations instead. Use a fresh output directory.
The 25,000 provided training reviews are split into 20,000 training and 5,000
validation reviews with stratification and seed 42. Vocabulary is built only
from those 20,000 reviews. GloVe is loaded once; each run receives independent
copies of the initial weights. Original baseline files are left intact.

Runs save training/validation losses and accuracies, full validation metrics,
settings, and best-validation-loss checkpoints with vocabulary. Early stopping
uses three epochs without validation-loss improvement. `summary.csv` compares
runs and `best_run.json` identifies the smallest validation loss. Use these
outputs for the hyperparameter table and Doga's convergence plots.
This validation baseline uses less training data than the original model, so
its scores are not directly comparable to the original test accuracy.

Only after selecting settings on validation data, evaluate the chosen model:

```bash
.venv/bin/python -m script.stage_4_script.evaluate_tuned_classification result/stage_4_result/classification_tuning/sweep_01/run_01/best.pt
```

Replace `run_01` with the selected run in `best_run.json`. Results are saved as
`test_metrics.json` beside that checkpoint. Do not use test results to select
further hyperparameters. Improvement over the original 80.28% is not guaranteed.
