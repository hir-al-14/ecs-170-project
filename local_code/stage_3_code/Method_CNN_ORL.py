import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if __package__:
    from .Evaluate_Classification import evaluate_predictions
else:
    from Evaluate_Classification import evaluate_predictions
if __package__:
    from .Method_CNN_Base import Method_CNN_Base
    from .Method_Dataset_Loader import Dataset_ORL
else:
    from Method_CNN_Base import Method_CNN_Base
    from Method_Dataset_Loader import Dataset_ORL

# reuse my plot function from stage 2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage_2_code'))
from plot_utils import plot_learning_curve


class Method_CNN_ORL(Method_CNN_Base):
    learning_rate = 0.001
    batch_size = 16
    max_epoch = 30

    def __init__(self):
        super().__init__()
        # 3 conv blocks, each one halves the image size with maxpool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x92 -> 56x46
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 28x23
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 14x11
        )
        # flatten then fc layers like the stage 2 mlp
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 40),  # 40 people in orl
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def train_model(self, train_set):
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        self.loss_history = []

        for epoch in range(self.max_epoch):
            loss_sum = 0.0
            for X_batch, y_batch in loader:
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * len(y_batch)

            # average loss over the whole epoch
            avg_loss = loss_sum / len(train_set)
            self.loss_history.append(avg_loss)
            print('Epoch:', epoch, 'Loss:', avg_loss)

    def test_model(self, test_set):
        X_test = torch.from_numpy(test_set.data)
        with torch.no_grad():  # no training here so no gradients
            y_pred = self.forward(X_test).argmax(dim=1)
        return y_pred.numpy()


if __name__ == '__main__':
    train_set = Dataset_ORL('train')
    test_set = Dataset_ORL('test')

    model = Method_CNN_ORL()
    model.train_model(train_set)

    pred_y = model.test_model(test_set)
    true_y = test_set.labels

    scores = evaluate_predictions(true_y, pred_y)
    print('Accuracy:', scores['accuracy'])
    print('Precision:', scores['precision'])
    print('Recall:', scores['recall'])
    print('F1:', scores['f1'])

    # save the convergence plot
    result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'result', 'stage_3_result')
    os.makedirs(result_dir, exist_ok=True)
    plot_learning_curve(model.loss_history, os.path.join(result_dir, 'ORL_learning_curve.png'), 'CNN Training Loss (ORL)')
    print('saved ORL_learning_curve.png')