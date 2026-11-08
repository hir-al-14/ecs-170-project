import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


class CNN(nn.Module):
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

    args = parser.parse_args()

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
    test = dataset_class('test')

    shape = tuple(train.data.shape[1:])
    epochs, batch_size, weight_decay = SETTINGS[args.dataset]

    if args.quick:
        train = Subset(train, range(min(96, len(train))))
        test = Subset(test, range(min(48, len(test))))
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

    print('Device:', device)

    folder_name = (
        'chaemin_quick_test'
        if args.quick else 'chaemin_ablation'
    )

    out = (
        ROOT / 'result' / 'stage_3_result' / folder_name
    )
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}

    for variant in VARIANTS:
        torch.manual_seed(1)

        model = CNN(
            args.dataset, variant, shape
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=weight_decay
        )

        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(1)
        )

        loss_history = []

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
            test_loader = DataLoader(
                test,
                batch_size=batch_size * 2
            )

            for x, y in test_loader:
                predictions = model(
                    x.to(device)
                ).argmax(1).cpu().tolist()

                predicted.extend(predictions)
                actual.extend(y.tolist())

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                actual,
                predicted,
                average='weighted',
                zero_division=0
            )
        )

        accuracy = accuracy_score(
            actual, predicted
        )

        rows.append({
            'dataset': args.dataset,
            'variant': variant,
            'parameters': sum(
                p.numel() for p in model.parameters()
            ),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        curves[variant] = loss_history

        print(
            'TEST ACCURACY:',
            f'{accuracy:.4f}',
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