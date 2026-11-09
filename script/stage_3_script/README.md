# Stage 3 evaluation and tuning

Run commands from the repository root. The models in this runner reproduce the existing MNIST, ORL, and CIFAR baseline architectures. The existing architecture ablations (`half_filters` and `kernel5`) remain available. Hiral's study keeps architecture fixed and compares learning rate and batch size.

## Setup

Use Python 3.14 and a virtual environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stage3.txt
```

The datasets are tracked with Git LFS. Install Git LFS if needed, then:

```bash
git lfs install --local
git lfs pull --include="data/stage_3_data/*" origin
```

Alternatively, extract the instructor's files as `data/stage_3_data/MNIST`, `ORL`, and `CIFAR`. They must be the actual pickle files, not small text pointers. Only load dataset/prediction files from a trusted source.

## Checks

```bash
python -m unittest discover -s tests -v
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available())"
```

A short real-data check (not a report result):

```bash
python script/stage_3_script/script_cnn_ablation.py orl --quick --variants baseline --validation-fraction 0.2 --output-dir /tmp/orl-smoke-check
```

## Hyperparameter protocol

For each dataset, use the same seed-1 stratified 80/20 split of the instructor's training set. Compare these four configurations, keeping architecture, optimizer, loss, weight decay, and epoch budget fixed:

| Configuration | Learning rate | Batch size |
|---|---:|---:|
| baseline | 0.001 | 16 ORL; 64 MNIST/CIFAR |
| low_lr | 0.0001 | Default |
| high_lr | 0.01 | Default |
| small_batch | 0.001 | 8 ORL; 32 MNIST/CIFAR |

Select the highest validation weighted F1 (exact ties favor baseline order), retrain a fresh model on the complete training set, and evaluate on the supplied test set. There is no early stopping or test-based selection. MNIST trains for 15 epochs; ORL and CIFAR for 30. Adam uses weight decay 0.0001 only for CIFAR. All runs use cross-entropy.

The suite resumes completed runs with matching settings and writes separate experiment folders:

```bash
python script/stage_3_script/script_cnn_tuning.py --device mps
```

Use `--device cuda` on supported NVIDIA hardware or `--device cpu` otherwise. ORL always runs on CPU to match the existing ORL comparison. MPS is the optional GPU component for MNIST/CIFAR. Do not change the device midway through a dataset's study; the suite checks recorded devices. Exact GPU reproducibility is not guaranteed.

To run one dataset:

```bash
python script/stage_3_script/script_cnn_tuning.py --datasets mnist --device mps
```

To reproduce one setting in a fresh folder:

```bash
python script/stage_3_script/script_cnn_ablation.py mnist --variants baseline --learning-rate 0.0001 --batch-size 64 --validation-fraction 0.2 --device mps --output-dir /tmp/mnist-low-lr-reproduction
```

Without `--validation-fraction`, the runner trains on the full training set and evaluates on the test set. Use that mode only after choosing the settings. The runner rejects an output directory that already contains completed results. Quick runs go in a separate `quick_test` directory and are marked in the CSV.

## Saved outputs

Results are under `result/stage_3_result/hiral_tuning/`. Each run includes configuration/metric CSV, a loss curve, and (for new runs) full-precision loss CSV and class predictions. The first four ORL validation runs predate numerical-loss/prediction export and retain their original curves and metrics. Their ignored local logs are not required to reproduce them.

`selection.json` identifies the validation-selected configuration. `final_test_results.csv` contains the three final held-out evaluations. `command.json` records the exact local invocation; use the portable commands above on another machine.

After all datasets finish, verify predictions and rebuild the summary, combined curves, per-class reports, and confusion matrices:

```bash
python script/stage_3_script/script_summarize_cnn_tuning.py
```

Precision, recall, and F1 are support-weighted with `zero_division=0`. Scores in CSV files are fractions; the summary expresses them as percentages. Do not confuse validation scores with final test scores or use the quick-test outputs in a report.
