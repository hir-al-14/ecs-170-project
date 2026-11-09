"""
Method_CNN_Base.py

Minimal shared base class for the Stage 3 CNN models.

The interface deliberately mirrors the existing Method_CNN_ORL.py so a model can
be moved onto this base without changing how it is called:

    model = SomeCNN()
    model.train_model(train_set)        # fills model.loss_history / model.acc_history
    pred_y = model.test_model(test_set) # numpy array of predicted labels
    scores = model.evaluate(test_set)   # accuracy / precision / recall / f1
    model.plot_learning_curve()         # saves the loss curve

It only relies on the torch Dataset protocol (len + (image, label) items), so it
works with the original Method_Dataset_Loader and the updated one alike.

Subclasses define two attributes in __init__:
    self.conv_layers   nn.Sequential, (N, C, H, W) -> feature maps
    self.fc_layers     nn.Sequential, flattened features -> class logits
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__:
    from .Evaluate_Classification import evaluate_predictions
else:
    from Evaluate_Classification import evaluate_predictions


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN_Base(nn.Module):
    # defaults, overridable per subclass (same names as Method_CNN_ORL.py)
    learning_rate = 0.001
    batch_size = 64
    max_epoch = 30
    weight_decay = 0.0

    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.loss_history = []   # mean training loss per epoch
        self.acc_history = []    # training accuracy per epoch

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

    @staticmethod
    def flat_dim(conv_layers, input_shape):
        """Number of features after conv_layers for an input of (C, H, W)."""
        with torch.no_grad():
            return int(np.prod(conv_layers(torch.zeros(1, *input_shape)).shape[1:]))

    # training
    def train_model(self, train_set):
        self.to(self.device)
        self.train()
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_function = nn.CrossEntropyLoss()
        self.loss_history, self.acc_history = [], []

        for epoch in range(self.max_epoch):
            loss_sum, correct = 0.0, 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * len(y_batch)
                correct += (y_pred.argmax(dim=1) == y_batch).sum().item()

            avg_loss = loss_sum / len(train_set)
            self.loss_history.append(avg_loss)
            self.acc_history.append(correct / len(train_set))
            print('Epoch:', epoch, 'Loss:', avg_loss, 'Acc:', self.acc_history[-1])

    # testing
    def _predict(self, dataset):
        self.to(self.device)
        self.eval()
        loader = DataLoader(dataset, batch_size=500, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                preds.append(self.forward(X_batch.to(self.device)).argmax(dim=1).cpu().numpy())
                labels.append(y_batch.numpy())
        return np.concatenate(preds), np.concatenate(labels)

    def test_model(self, test_set):
        return self._predict(test_set)[0]

    def evaluate(self, test_set):
        pred_y, true_y = self._predict(test_set)
        return evaluate_predictions(true_y, pred_y)

    # plotting
    @staticmethod
    def result_dir():
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'result', 'stage_3_result')
        os.makedirs(d, exist_ok=True)
        return d

    def plot_learning_curve(self, file_name, title='CNN Training Loss'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.loss_history) + 1)
        plt.figure()
        plt.plot(epochs, self.loss_history)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(title)
        path = os.path.join(self.result_dir(), file_name)
        plt.savefig(path)
        plt.close()
        return path