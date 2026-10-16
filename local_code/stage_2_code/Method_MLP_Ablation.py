'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP_Ablation(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 64

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 128)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(128, 64)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(64, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        y_pred = self.fc_layer_3(h2)
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        n_samples = X_tensor.shape[0]
        self.loss_history = []


        for epoch in range(self.max_epoch):
            perm = torch.randperm(n_samples)
            X_shuffled = X_tensor[perm]
            y_shuffled = y_tensor[perm]

            epoch_losses = []
            epoch_correct = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_losses.append(train_loss.item())
                epoch_correct += (y_pred.max(1)[1] == y_batch).sum().item()

            avg_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / n_samples
            self.loss_history.append(avg_loss)

            if epoch % 100 == 0:
                print('Epoch:', epoch, 'Loss:', avg_loss, 'Accuracy:', train_acc)

        print('Final Loss:', avg_loss)

    def test(self, X):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}