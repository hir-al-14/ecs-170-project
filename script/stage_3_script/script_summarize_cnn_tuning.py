"""Verify saved final predictions and rebuild tables and figures without training."""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_3_code.Evaluate_Classification import evaluate_predictions

OUTPUT = ROOT / 'result' / 'stage_3_result' / 'hiral_tuning'
DATASETS = ['orl', 'mnist', 'cifar']


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def main():
    lines = ['# Stage 3 evaluation and hyperparameter tuning', '',
             'All metrics below are percentages. Precision, recall, and F1 use support-weighted averaging.', '',
             '## Experiment protocol', '',
             'Each dataset uses its existing baseline CNN architecture, Adam, cross-entropy, seed 1, '
             'and a fixed stratified 80/20 split of the supplied training set. '
             'The four configurations vary learning rate or batch size one at a time. '
             'Training lasts 15 epochs for MNIST and 30 for ORL/CIFAR. '
             'The configuration with the highest validation weighted F1 is selected; exact ties favor '
             'the earlier configuration (baseline first). A fresh model is then trained on the entire '
             'supplied training set and evaluated on the supplied test set. '
             'ORL uses CPU; MNIST and CIFAR use MPS.', '',
             '## Validation comparison', '',
             '| Dataset | Configuration | LR | Batch | Accuracy | Precision | Recall | F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    fig_scores, axes_scores = plt.subplots(1, 3, figsize=(14, 4))
    fig_losses, axes_losses = plt.subplots(1, 3, figsize=(14, 4))
    finals = []
    for index, dataset in enumerate(DATASETS):
        folder = OUTPUT / dataset
        rows = read_rows(folder / 'validation_comparison.csv')
        if len(rows) != 4 or any(row['evaluation_split'] != 'validation' or row['quick'] != 'False'
                                 for row in rows):
            raise ValueError(f'Incomplete validation results: {dataset}')
        for row in rows:
            metrics = [float(row[key]) for key in ['accuracy', 'precision', 'recall', 'f1']]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics):
                raise ValueError(f'Invalid metrics: {row}')
            lines.append('| ' + ' | '.join([dataset.upper(), row['experiment'], row['learning_rate'],
                                           row['batch_size']] + [f'{value * 100:.2f}' for value in metrics]) + ' |')
        best = max(rows, key=lambda row: float(row['f1']))
        selected = json.loads((folder / 'selection.json').read_text())
        assert selected['selected_experiment'] == best['experiment']
        ax = axes_scores[index]
        ax.bar([row['experiment'].replace('_', '\n') for row in rows],
               [float(row['f1']) * 100 for row in rows], color='#3976a8')
        ax.set(title=dataset.upper(), ylabel='Validation weighted F1 (%)', ylim=(0, 100))

        final = read_rows(folder / 'final' / f'{dataset}_ablation_results.csv')[0]
        assert final['evaluation_split'] == 'test' and final['quick'] == 'False'
        assert float(final['learning_rate']) == selected['learning_rate']
        assert int(final['batch_size']) == selected['batch_size']
        expected_train, expected_test = {'orl': (360, 40), 'mnist': (60000, 10000),
                                        'cifar': (50000, 10000)}[dataset]
        assert int(final['train_samples']) == expected_train
        assert int(final['evaluation_samples']) == expected_test
        predictions = json.loads((folder / 'final' / f'{dataset}_baseline_predictions.json').read_text())
        true, pred = predictions['true_y'], predictions['pred_y']
        assert predictions['evaluation_split'] == 'test'
        assert len(true) == len(pred) == expected_test
        scores = evaluate_predictions(true, pred)
        for metric, value in scores.items():
            assert math.isclose(float(final[metric]), value, rel_tol=1e-12, abs_tol=1e-12)
        final['selected_experiment'] = selected['selected_experiment']
        finals.append(final)

        history = read_rows(folder / 'final' / f'{dataset}_baseline_loss_history.csv')
        assert [int(row['epoch']) for row in history] == list(range(1, int(final['epochs']) + 1))
        losses = [float(row['training_loss']) for row in history]
        assert all(math.isfinite(loss) and loss >= 0 for loss in losses)
        axes_losses[index].plot(range(1, len(losses) + 1), losses)
        axes_losses[index].set(title=dataset.upper(), xlabel='Epoch', ylabel='Training loss')
        report = classification_report(true, pred, output_dict=True, zero_division=0)
        (folder / 'final' / 'classification_report.json').write_text(json.dumps(report, indent=2) + '\n')
        matrix = confusion_matrix(true, pred, labels=list(range(40 if dataset == 'orl' else 10)))
        figure, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap='Blues')
        ax.set(title=f'{dataset.upper()} final test confusion matrix',
               xlabel='Predicted class index', ylabel='True class index')
        figure.colorbar(im, ax=ax, label='Samples')
        figure.tight_layout()
        figure.savefig(folder / 'final' / 'confusion_matrix.png', dpi=160)
        plt.close(figure)

    lines += ['', '## Final held-out test results', '',
              '| Dataset | Selected setting | Accuracy | Precision | Recall | F1 |',
              '|---|---|---:|---:|---:|---:|']
    for row in finals:
        lines.append('| ' + ' | '.join([row['dataset'].upper(), row['selected_experiment']] +
                     [f"{float(row[key]) * 100:.2f}" for key in ['accuracy', 'precision', 'recall', 'f1']]) + ' |')
    lines += ['', '## Interpretation and limitations', '',
              'Selection uses validation F1, not final test performance. These are single-seed results '
              'on one validation split; small differences do not establish a statistically reliable advantage. '
              'ORL has only 72 validation images and 40 test images, making its percentages particularly sensitive '
              'to individual errors. The epoch budget is fixed, so a lower learning rate may need more epochs '
              'than this study allows. GPU arithmetic can cause small differences on reruns. '
              'Weighted averages can hide weak classes; the saved per-class reports and confusion matrices '
              'provide additional detail.', '',
              'The architecture label `baseline` in each run describes the fixed CNN architecture, not necessarily '
              'the default learning rate or batch size. See the `experiment` names and recorded settings.', '',
              '## Figures', '',
              '![Validation F1 comparison](validation_f1_comparison.png)', '',
              '![Final training convergence](final_learning_curves.png)', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines))
    for figure, filename in [(fig_scores, 'validation_f1_comparison.png'),
                             (fig_losses, 'final_learning_curves.png')]:
        figure.tight_layout()
        figure.savefig(OUTPUT / filename, dpi=160)
        plt.close(figure)
    print('Verified all three final predictions, metrics, selected settings, and loss histories.')
    print('\n'.join(lines[lines.index('## Final held-out test results'):lines.index('## Interpretation and limitations')]))


if __name__ == '__main__':
    main()
