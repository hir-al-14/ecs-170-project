"""Run the fixed hyperparameter study, then test the validation-selected model."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'script' / 'stage_3_script' / 'script_cnn_ablation.py'
OUTPUT = ROOT / 'result' / 'stage_3_result' / 'hiral_tuning'
CONFIGS = [('baseline', 0.001), ('low_lr', 0.0001),
           ('high_lr', 0.01), ('small_batch', 0.001)]


def run_experiment(dataset, name, learning_rate, batch_size, validation, device):
    folder = OUTPUT / dataset / name
    results_folder = folder / 'validation' if validation else folder
    results_file = results_folder / f'{dataset}_ablation_results.csv'
    command = [sys.executable, str(RUNNER), dataset, '--variants', 'baseline',
               '--learning-rate', str(learning_rate), '--batch-size', str(batch_size),
               '--output-dir', str(folder), '--device', device]
    if validation:
        command += ['--validation-fraction', '0.2']
    if not results_file.exists():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        print(f'Starting {dataset} {name}: LR={learning_rate}, batch={batch_size}', flush=True)
        with (folder / 'training.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    with results_file.open() as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f'Expected one baseline-architecture result: {results_file}')
    row = rows[0]
    expected_split = 'validation' if validation else 'test'
    if (row['dataset'] != dataset or row['variant'] != 'baseline'
            or row['evaluation_split'] != expected_split or row['quick'] != 'False'
            or float(row['learning_rate']) != learning_rate
            or int(row['batch_size']) != batch_size or int(row['seed']) != 1
            or int(row['epochs']) != (15 if dataset == 'mnist' else 30)
            or float(row['validation_fraction']) != (0.2 if validation else 0.0)
            or row.get('device', 'cpu') != device):
        raise ValueError(f'Existing result has different settings: {results_file}')
    row = {'experiment': name, **row}
    print(f"Finished {dataset} {name}: {expected_split} F1={float(row['f1']):.6f}", flush=True)
    return row


def save_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['orl', 'mnist', 'cifar'],
                        default=['orl', 'mnist', 'cifar'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                        help='Device for MNIST/CIFAR; ORL stays on CPU to match its existing study')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        batch = 16 if dataset == 'orl' else 64
        device = 'cpu' if dataset == 'orl' else args.device
        rows = [run_experiment(dataset, name, lr,
                               batch // 2 if name == 'small_batch' else batch,
                               True, device) for name, lr in CONFIGS]
        save_table(OUTPUT / dataset / 'validation_comparison.csv', rows)
        # Fixed ordering breaks exact ties in favor of the baseline.
        best = max(rows, key=lambda row: float(row['f1']))
        selection = {'criterion': 'validation weighted F1',
                     'selected_experiment': best['experiment'],
                     'learning_rate': float(best['learning_rate']),
                     'batch_size': int(best['batch_size']),
                     'validation_f1': float(best['f1'])}
        (OUTPUT / dataset / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        run_experiment(dataset, 'final', selection['learning_rate'],
                       selection['batch_size'], False, device)
    finals = []
    for dataset in ['orl', 'mnist', 'cifar']:
        path = OUTPUT / dataset / 'final' / f'{dataset}_ablation_results.csv'
        if path.exists():
            with path.open() as file:
                finals.extend(csv.DictReader(file))
    save_table(OUTPUT / 'final_test_results.csv', finals)
    print('Completed. Results:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
