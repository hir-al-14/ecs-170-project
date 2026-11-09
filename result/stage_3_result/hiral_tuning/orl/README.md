# ORL hyperparameter comparison

All runs use the baseline ORL architecture, Adam, cross-entropy, 30 epochs, and seed 1. The stratified split uses 288 training images and 72 validation images from the supplied training set. The 40 test images were not evaluated. Scores below are percentages; precision, recall, and F1 are support-weighted.

| Experiment | Learning rate | Batch size | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.001 | 16 | 94.44 | 93.98 | 94.44 | 93.33 |
| low_lr | 0.0001 | 16 | 88.89 | 86.11 | 88.89 | 86.02 |
| high_lr | 0.01 | 16 | 1.39 | 0.02 | 1.39 | 0.04 |
| small_batch | 0.001 | 8 | 94.44 | 94.91 | 94.44 | 93.61 |

Selection criterion: highest validation weighted F1. Batch size 8 with learning rate 0.001 ranks first among these four settings. Its accuracy matches the baseline (68/72 correct), so the small F1 improvement is not evidence of a robust advantage across seeds or splits. The lower learning rate is worse under the fixed 30-epoch budget; the higher learning rate finishes with loss near 3.69 and poor validation performance.

These are validation results, not final test results. Retrain the selected configuration on all 360 training images before final test evaluation.

Example reproduction command (use a fresh output directory for each experiment):

```bash
python script/stage_3_script/script_cnn_ablation.py orl --variants baseline --validation-fraction 0.2 --learning-rate 0.001 --batch-size 8 --output-dir /tmp/orl-small-batch-reproduction
```
