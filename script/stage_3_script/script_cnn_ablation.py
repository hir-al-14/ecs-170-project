import argparse
import csv
import math
import json
import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base
from local_code.stage_3_code.Evaluate_Classification import evaluate_predictions
from local_code.stage_3_code.Method_Dataset_Loader import (
    Dataset_MNIST, Dataset_ORL, Dataset_CIFAR
)

DATASETS = {
    'mnist': Dataset_MNIST,
    'orl': Dataset_ORL,
    'cifar': Dataset_CIFAR
}

SETTINGS = {
    'mnist': (15, 64, 0.0),
    'orl': (30, 16, 0.0),
    'cifar': (30, 64, 1e-4)
}

VARIANTS = ['baseline', 'half_filters', 'kernel5']


class CNN(Method_CNN_Base):
    def __init__(self, dataset, variant, shape):
        super().__init__()

        if dataset == 'mnist':
            widths, repeats, hidden, classes = [25, 50], 1, 64, 10
        elif dataset == 'orl':
            widths, repeats, hidden, classes = [16, 32, 64], 1, 128, 40
        else:
            widths, repeats, hidden, classes = [32, 64, 128], 2, 256, 10

        if variant == 'half_filters':
            widths = [max(1, w // 2) for w in widths]

        kernel = 5 if variant == 'kernel5' else 3

        layers = []
        in_channels = shape[0]

        for out_channels in widths:
            for _ in range(repeats):
                layers.append(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel, padding=kernel // 2
                    )
                )

                if dataset == 'cifar':
                    layers.append(nn.BatchNorm2d(out_channels))

                layers.append(nn.ReLU())
                in_channels = out_channels

            layers.append(nn.MaxPool2d(2))

            if dataset == 'cifar':
                layers.append(nn.Dropout(0.25))

        self.conv = nn.Sequential(*layers)

        self.conv.eval()
        with torch.no_grad():
            flat = self.conv(
                torch.zeros(1, *shape)
            ).flatten(1).shape[1]
        self.conv.train()

        dense = [
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU()
        ]

        if dataset == 'mnist':
            dense.append(nn.Dropout(0.2))
        elif dataset == 'cifar':
            dense.append(nn.Dropout(0.5))

        dense.append(nn.Linear(hidden, classes))
        self.fc = nn.Sequential(*dense)

    def forward(self, x):
        return self.fc(self.conv(x))


def split_training_data(dataset, validation_fraction):
    """Keep class proportions and use the same split for every tuning run."""
    train_indices, validation_indices = train_test_split(
        list(range(len(dataset))),
        test_size=validation_fraction,
        random_state=1,
        stratify=dataset.labels,
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset',
        choices=['mnist', 'orl', 'cifar']
    )

    parser.add_argument(
        '--data-dir',
        help='Folder containing real MNIST, ORL, and CIFAR files'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Small code test only, NOT report results'
    )

    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Adam learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int,
                        help='Override the dataset-specific training batch size')
    parser.add_argument('--output-dir', type=Path,
                        help='Separate folder for this experiment')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=VARIANTS,
                        help='Architectures to run (default: all three)')

    parser.add_argument('--validation-fraction', type=float, default=0.0,
                        help='Fraction of training data used for validation; '
                             '0 evaluates on the supplied test set (default)')

    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'],
                        default='auto', help='Training device (default: auto)')

    args = parser.parse_args()
    if not math.isfinite(args.validation_fraction) or not 0 <= args.validation_fraction < 1:
        parser.error('--validation-fraction must be in [0, 1)')
    if args.validation_fraction > 0 and args.output_dir is None:
        parser.error('Use --output-dir to keep validation results separate')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be a finite positive number')
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')
    if (args.learning_rate != 0.001 or args.batch_size is not None) and args.output_dir is None:
        parser.error('Use --output-dir for tuning runs to preserve existing results')

    dataset_class = DATASETS[args.dataset]

    if args.data_dir:
        dataset_class.dataset_source_folder_path = str(
            Path(args.data_dir).resolve()
        )

    source = (
        Path(dataset_class.dataset_source_folder_path)
        / dataset_class.file_name
    )

    if source.exists():
        with source.open('rb') as file:
            if file.read(16).startswith(b'version https://'):
                parser.error(
                    'Dataset is a Git LFS pointer. '
                    'Use real files from stage_3_data.zip.'
                )

    train = dataset_class('train')

    shape = tuple(train.data.shape[1:])
    epochs, batch_size, weight_decay = SETTINGS[args.dataset]

    if args.validation_fraction > 0:
        try:
            train, evaluation_set = split_training_data(train, args.validation_fraction)
        except ValueError as error:
            parser.error(f'Cannot create stratified validation split: {error}')
        evaluation_split = 'validation'
    else:
        evaluation_set = dataset_class('test')
        evaluation_split = 'test'

    print(f'Training samples: {len(train)}, '
          f'{evaluation_split} samples: {len(evaluation_set)}')

    if args.batch_size is not None:
        batch_size = args.batch_size

    if args.quick:
        train = Subset(train, range(min(96, len(train))))
        evaluation_set = Subset(evaluation_set, range(min(48, len(evaluation_set))))
        epochs = 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif (
        hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
    ):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.device != 'auto':
        if args.device == 'mps' and not torch.backends.mps.is_available():
            parser.error('MPS is not available in this process')
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA is not available in this process')
        device = torch.device(args.device)

    print('Device:', device)

    folder_name = (
        'chaemin_quick_test'
        if args.quick else 'chaemin_ablation'
    )

    out = (
        ROOT / 'result' / 'stage_3_result' / folder_name
    )
    if args.output_dir is not None:
        out = args.output_dir.resolve()
        if args.quick:
            out = out / 'quick_test'
    if evaluation_split == 'validation':
        out = out / 'validation'
    out.mkdir(parents=True, exist_ok=True)

    result_path = out / f'{args.dataset}_ablation_results.csv'
    if result_path.exists():
        parser.error(f'Results already exist at {result_path}; use a fresh output directory')

    rows = []
    curves = {}

    for variant in args.variants:
        torch.manual_seed(1)

        model = CNN(
            args.dataset, variant, shape
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=weight_decay
        )

        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(1)
        )

        loss_history = []
        start_time = time.perf_counter()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                loss = nn.functional.cross_entropy(
                    model(x), y
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)

            average_loss = total_loss / len(train)
            loss_history.append(average_loss)
            with (out / f'{args.dataset}_{variant}_loss_history.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'training_loss'])
                writer.writerows(enumerate(loss_history, start=1))

            print(
                f'{args.dataset} {variant}: '
                f'epoch {epoch + 1}/{epochs}, '
                f'loss={average_loss:.4f}',
                flush=True
            )

        model.eval()

        actual = []
        predicted = []

        with torch.no_grad():
            evaluation_loader = DataLoader(
                evaluation_set,
                batch_size=batch_size * 2
            )

            for x, y in evaluation_loader:
                predictions = model(
                    x.to(device)
                ).argmax(1).cpu().tolist()

                predicted.extend(predictions)
                actual.extend(y.tolist())

        scores = evaluate_predictions(actual, predicted)
        with (out / f'{args.dataset}_{variant}_predictions.json').open('w') as file:
            json.dump({'evaluation_split': evaluation_split,
                       'true_y': actual, 'pred_y': predicted}, file)


        rows.append({
            'dataset': args.dataset,
            'variant': variant,
            'learning_rate': args.learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'seed': 1,
            'device': str(device),
            'torch_version': torch.__version__,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
            'duration_seconds': time.perf_counter() - start_time,
            'quick': args.quick,
            'evaluation_split': evaluation_split,
            'validation_fraction': args.validation_fraction,
            'train_samples': len(train),
            'evaluation_samples': len(evaluation_set),
            'parameters': sum(
                p.numel() for p in model.parameters()
            ),
            **scores
        })

        curves[variant] = loss_history

        print(
            f'{evaluation_split.upper()} ACCURACY:',
            f"{scores['accuracy']:.4f}",
            flush=True
        )

        csv_path = out / (
            f'{args.dataset}_ablation_results.csv'
        )

        with csv_path.open('w', newline='') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    for name, losses in curves.items():
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            label=name
        )

    plt.title(
        f'{args.dataset.upper()} Ablation Learning Curve'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out / f'{args.dataset}_ablation_curve.png',
        dpi=160
    )

    plt.close()

    print('Saved results in:', out)


if __name__ == '__main__':
    main()
