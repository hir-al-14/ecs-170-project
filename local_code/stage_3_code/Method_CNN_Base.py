"""
Method_CNN_Base.py

Shared training / evaluation / plotting logic for the Stage 3 CNN models.
Subclasses only define their architecture in _build().

If your course template has a `method` base classgi (as used by Method_MLP.py),
you can change the base to `class Method_CNN_Base(method, nn.Module)`.
"""

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN_Base(nn.Module):
    # default hyperparameters (override per subclass or via __init__ kwargs)
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 64
    weight_decay = 0.0

    def __init__(self, input_shape, num_classes, name='CNN', **hparams):
        super().__init__()
        self.name = name
        self.input_shape = tuple(input_shape)      # (C, H, W)
        self.num_classes = int(num_classes)

        # allow overriding any class-level hyperparameter, e.g. max_epoch=30
        for k, v in hparams.items():
            setattr(self, k, v)

        self.device = get_device()
        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        self.features, self.classifier = self._build()   # subclass defines these
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(self.device)

    # architecture hook
    def _build(self):
        raise NotImplementedError

    def _flat_dim(self, features):
        """Size of the flattened feature map, found with a dummy forward pass."""
        with torch.no_grad():
            was_training = features.training
            features.eval()
            out = features(torch.zeros(1, *self.input_shape))
            features.train(was_training)
        return int(np.prod(out.shape[1:]))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    # evaluation
    @torch.no_grad()
    def evaluate(self, dataset, batch_size=256):
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss, preds, labels = 0.0, [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self(x)
            total_loss += self.loss_fn(logits, y).item() * len(y)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        return total_loss / len(dataset), preds, labels

    @staticmethod
    def metrics(preds, labels, num_classes):
        """Accuracy plus weighted precision / recall / F1 (no sklearn needed)."""
        acc = float((preds == labels).mean())
        p_sum = r_sum = f_sum = 0.0
        for c in range(num_classes):
            tp = float(((preds == c) & (labels == c)).sum())
            fp = float(((preds == c) & (labels != c)).sum())
            fn = float(((preds != c) & (labels == c)).sum())
            support = tp + fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            p_sum += prec * support
            r_sum += rec * support
            f_sum += f1 * support
        n = float(len(labels))
        return {'accuracy': acc, 'precision': p_sum / n, 'recall': r_sum / n, 'f1': f_sum / n}

    # training
    def fit(self, train_set, test_set=None, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        start = time.time()
        for epoch in range(1, self.max_epoch + 1):
            self.train()
            run_loss, correct = 0.0, 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()

            self.history['train_loss'].append(run_loss / len(train_set))
            self.history['train_acc'].append(correct / len(train_set))

            msg = (f"[{self.name}] epoch {epoch:3d}/{self.max_epoch} "
                   f"train_loss={self.history['train_loss'][-1]:.4f} "
                   f"train_acc={self.history['train_acc'][-1]:.4f}")
            if test_set is not None:
                t_loss, p, l = self.evaluate(test_set)
                self.history['test_loss'].append(t_loss)
                self.history['test_acc'].append(float((p == l).mean()))
                msg += f" test_loss={t_loss:.4f} test_acc={self.history['test_acc'][-1]:.4f}"
            if verbose:
                print(msg, flush=True)

        self.train_time = time.time() - start

    def run(self, train_set, test_set, verbose=True):
        """Train, then evaluate on the test set. Returns a results dict."""
        self.fit(train_set, test_set, verbose=verbose)
        _, preds, labels = self.evaluate(test_set)
        res = self.metrics(preds, labels, self.num_classes)
        res['train_time_s'] = self.train_time
        res['n_params'] = sum(p.numel() for p in self.parameters())
        return res

    # plotting
    def plot_learning_curves(self, path=None, show=False):
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        ax[0].plot(epochs, self.history['train_loss'], label='train')
        if self.history['test_loss']:
            ax[0].plot(epochs, self.history['test_loss'], label='test')
        ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Cross-entropy loss')
        ax[0].set_title(f'{self.name}: loss'); ax[0].legend(); ax[0].grid(alpha=.3)

        ax[1].plot(epochs, self.history['train_acc'], label='train')
        if self.history['test_acc']:
            ax[1].plot(epochs, self.history['test_acc'], label='test')
        ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy')
        ax[1].set_title(f'{self.name}: accuracy'); ax[1].legend(); ax[1].grid(alpha=.3)

        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
"""
Method_CNN_Base.py

Shared training / evaluation / plotting logic for the Stage 3 CNN models.
Subclasses only define their architecture in _build().

If your course template has a `method` base class (as used by Method_MLP.py),
you can change the base to `class Method_CNN_Base(method, nn.Module)`.
"""

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN_Base(nn.Module):
    # default hyperparameters (override per subclass or via __init__ kwargs)
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 64
    weight_decay = 0.0

    def __init__(self, input_shape, num_classes, name='CNN', **hparams):
        super().__init__()
        self.name = name
        self.input_shape = tuple(input_shape)      # (C, H, W)
        self.num_classes = int(num_classes)

        # allow overriding any class-level hyperparameter, e.g. max_epoch=30
        for k, v in hparams.items():
            setattr(self, k, v)

        self.device = get_device()
        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        self.features, self.classifier = self._build()   # subclass defines these
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(self.device)

    # architecture hook
    def _build(self):
        raise NotImplementedError

    def _flat_dim(self, features):
        """Size of the flattened feature map, found with a dummy forward pass."""
        with torch.no_grad():
            was_training = features.training
            features.eval()
            out = features(torch.zeros(1, *self.input_shape))
            features.train(was_training)
        return int(np.prod(out.shape[1:]))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    # evaluation
    @torch.no_grad()
    def evaluate(self, dataset, batch_size=256):
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss, preds, labels = 0.0, [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self(x)
            total_loss += self.loss_fn(logits, y).item() * len(y)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        return total_loss / len(dataset), preds, labels

    @staticmethod
    def metrics(preds, labels, num_classes):
        """Accuracy plus weighted precision / recall / F1 (no sklearn needed)."""
        acc = float((preds == labels).mean())
        p_sum = r_sum = f_sum = 0.0
        for c in range(num_classes):
            tp = float(((preds == c) & (labels == c)).sum())
            fp = float(((preds == c) & (labels != c)).sum())
            fn = float(((preds != c) & (labels == c)).sum())
            support = tp + fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            p_sum += prec * support
            r_sum += rec * support
            f_sum += f1 * support
        n = float(len(labels))
        return {'accuracy': acc, 'precision': p_sum / n, 'recall': r_sum / n, 'f1': f_sum / n}

    # training
    def fit(self, train_set, test_set=None, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        start = time.time()
        for epoch in range(1, self.max_epoch + 1):
            self.train()
            run_loss, correct = 0.0, 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()

            self.history['train_loss'].append(run_loss / len(train_set))
            self.history['train_acc'].append(correct / len(train_set))

            msg = (f"[{self.name}] epoch {epoch:3d}/{self.max_epoch} "
                   f"train_loss={self.history['train_loss'][-1]:.4f} "
                   f"train_acc={self.history['train_acc'][-1]:.4f}")
            if test_set is not None:
                t_loss, p, l = self.evaluate(test_set)
                self.history['test_loss'].append(t_loss)
                self.history['test_acc'].append(float((p == l).mean()))
                msg += f" test_loss={t_loss:.4f} test_acc={self.history['test_acc'][-1]:.4f}"
            if verbose:
                print(msg, flush=True)

        self.train_time = time.time() - start

    def run(self, train_set, test_set, verbose=True):
        """Train, then evaluate on the test set. Returns a results dict."""
        self.fit(train_set, test_set, verbose=verbose)
        _, preds, labels = self.evaluate(test_set)
        res = self.metrics(preds, labels, self.num_classes)
        res['train_time_s'] = self.train_time
        res['n_params'] = sum(p.numel() for p in self.parameters())
        return res

    # plotting
    def plot_learning_curves(self, path=None, show=False):
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        ax[0].plot(epochs, self.history['train_loss'], label='train')
        if self.history['test_loss']:
            ax[0].plot(epochs, self.history['test_loss'], label='test')
        ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Cross-entropy loss')
        ax[0].set_title(f'{self.name}: loss'); ax[0].legend(); ax[0].grid(alpha=.3)

        ax[1].plot(epochs, self.history['train_acc'], label='train')
        if self.history['test_acc']:
            ax[1].plot(epochs, self.history['test_acc'], label='test')
        ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy')
        ax[1].set_title(f'{self.name}: accuracy'); ax[1].legend(); ax[1].grid(alpha=.3)

        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)