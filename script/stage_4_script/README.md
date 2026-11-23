# Stage 4 team workflow

The shared classifier and training command now support RNN, LSTM, and GRU.
Evaluation reuses those implementations; no separate classifier is introduced.
The original training runs use the provided train/test split. Hiral's tuning
runs use a held-out validation split and must be reported separately.

## Evaluate the team's classifiers

Run these from the repository root; evaluation loads saved weights, not GloVe:

```bash
.venv/bin/python -m script.stage_4_script.run_evaluation --cell rnn
.venv/bin/python -m script.stage_4_script.run_evaluation --cell lstm
.venv/bin/python -m script.stage_4_script.run_evaluation --cell gru
```

Each writes its own `<cell>_classification_metrics.json` and predictions.
Use `--output-dir /tmp/stage4-evaluation` to verify results without replacing
existing result files. The legacy evaluation module entry points still work.
Classifier checkpoints contain weights only, so evaluation reconstructs the
15,000-token vocabulary from the unchanged full training set, exactly as the
team training script does. Tuning checkpoints instead save their own vocabulary.

## Collect report tables and generation examples

```bash
.venv/bin/python -m script.stage_4_script.summarize_results
```

Writes `summary/classification_test.csv` and `summary/generation_validation.csv`.
These collect saved results without retraining. Generated examples still need
manual review for coherence, grammar, repetition, and copying. A lower loss
alone does not satisfy the instructor's readable-generation requirement.
The saved LSTM classifier reports 85.40% test accuracy, above the 85% minimum.

Doga's existing generation plot command is:

```bash
.venv/bin/python -m local_code.stage_4_code.plot_generation_curve
```

It reads `generation/rnn_review_baseline/history.json`. Other generator and
tuning histories are saved beside their checkpoints for additional plots.

## Classification evaluation details


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
remain supported. All three cell types use the same metrics function and checkpoint evaluation command.

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


## Existing classification tuning results

- `classification_tuning/baseline_agent_01/`: completed validation baseline.
- `classification_tuning/sweep_agent_01/summary.csv`: baseline and two completed
  learning-rate comparisons. Best completed validation accuracy is 80.56%.
- `classification_tuning/sweep_agent_01/status.json`: records the interrupted
  sweep. Batch-size/dropout comparisons are not complete. These validation
  results must not be described as test scores or a finished seven-run sweep.

To reuse the completed baseline in a new sweep:

```bash
.venv/bin/python -m script.stage_4_script.tune_classification --output result/stage_4_result/classification_tuning/sweep_new --baseline-from result/stage_4_result/classification_tuning/baseline_agent_01
```

Reuse checks split indices, vocabulary, and training settings. It reuses only
the baseline; it does not resume partially completed non-baseline runs.
Older incomplete local folders are not part of the shared results.

## Readable-generation demonstration

The original baseline is retained. The separate `rnn_glove_readability_01`
experiment uses GloVe initialization, 256 hidden units, a 64-token context,
dropout 0.2, and 40 epochs. `best.pt` still means minimum validation loss;
`last.pt` is the final model, kept separately for qualitative comparison.
The final model produces readable examples but memorizes training text and
still fails on some prompts. Do not describe its validation overfitting as fixed.

Run an actual terminal demonstration suitable for capturing a screenshot:

```bash
.venv/bin/python -m script.stage_4_script.generate_text result/stage_4_result/generation/rnn_glove_readability_01/last.pt --prompt "why did the" --temperature 0.4 --top-k 3 --seed 42 --plain --compare-training
```

This prints the checkpoint epoch and sampling settings, generated text, and
training-text overlap. The saved example is an exact training match; the
screenshot should retain that fact. No response text is hardcoded or retrieved
from the dataset during sampling; the training comparison runs afterward.

Review a fixed panel of three prompts and three seeds:

```bash
.venv/bin/python -m script.stage_4_script.review_generation result/stage_4_result/generation/rnn_glove_readability_01/last.pt --temperature 0.4 --top-k 3 --output /tmp/stage4-generation-review.json
```

The review command records all outputs, exact matches, and longest contiguous
training-token matches. Its manual-review fields start empty for a reviewer to
fill. Saved `review_last.json` includes qualitative observations and failures;
`review_best.json` and `review_additional.json` preserve other sampled outputs.
Perplexity alone and absence of an exact match are not evidence of coherent,
original writing. Training loss histories remain available for Doga's plots.

## Regularized generation experiments

`--sequence-training` predicts all next tokens in a complete example together.
It uses the same causal RNN; a test verifies that its evaluation loss matches
rolling-prefix evaluation when the complete examples fit the context window.
This reduces repeated prefix processing, so epoch counts and optimizer-step
counts should not be treated as identical to earlier rolling-window runs.

Additional options are `--min-freq`, `--embedding-dropout`, `--weight-decay`,
`--patience`, `--tie-embeddings`, and `--freeze-embeddings`. All default to the
original behavior. Filtering singleton training words with `--min-freq 2`
reduces the vocabulary from 4,157 to 1,643 tokens and supplies actual UNK targets
in training. Validation loss/perplexity must not be directly compared across
these two vocabularies. All experiments keep the original data partitions.

Completed same-vocabulary RNN comparisons:
- `rnn_tied_regularized_01`: tied embeddings, 256 hidden units; best epoch 17.
- `rnn_regularized_01`: untied embeddings, 256 hidden units; best epoch 17.
- `rnn_regularized_02`: untied embeddings, 128 hidden units; best epoch 35.

The smaller untied RNN has the lowest validation loss of these three RNNs.
The regularized models still have inconsistent output. The frozen, unfiltered
experiment `rnn_tied_frozen_01` was not an improvement. The existing LSTM
*generation* option was also tested in `lstm_regularized_01`; no classifier
architecture was added or changed.

For a deterministic readability demonstration, beam decoding keeps five
candidate continuations and blocks repeated trigrams:

```bash
.venv/bin/python -m script.stage_4_script.generate_text result/stage_4_result/generation/rnn_glove_readability_01/last.pt --prompt "what did the" --beam-size 5 --plain --compare-training
```

`review_beam.json` preserves all three default prompt outputs. For this model,
all three are readable training reproductions. This improves the demonstration's
consistency, not evidence of novelty or held-out generalization. The original
sampling reviews are retained alongside it. Summary tables identify vocabulary
size to prevent misleading perplexity comparisons across experiments.
