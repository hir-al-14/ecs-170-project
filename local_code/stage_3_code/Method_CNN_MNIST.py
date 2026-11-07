# import packages
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# import respective data loader
from Method_Dataset_Loader import Dataset_MNIST 

def load_mnist():
    train, test = Dataset_MNIST('train'), Dataset_MNIST('test') # from dataset loader class
    return (torch.from_numpy(train.data), torch.from_numpy(train.labels), torch.from_numpy(test.data), torch.from_numpy(test.labels))

if __name__ == '__main__':
    d_train, l_train, d_test, l_test = load_mnist() # get label and data tensor