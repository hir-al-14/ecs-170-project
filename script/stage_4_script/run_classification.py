# import package
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import modules
from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.Text_Preprocessor import (build_vocab, ReviewDataset, collate_reviews, load_glove_matrix)
from local_code.stage_4_code.Method_RNN_Classification import RNNClassifier
from local_code.stage_4_code.Config import (EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE,MAX_EPOCHS, MAX_SEQUENCE_LENGTH, get_glove_path)
# setting
MAX_LEN = MAX_SEQUENCE_LENGTH
EPOCHS = MAX_EPOCHS
HIDDEN_SIZE = HIDDEN_DIM
MAX_VOCAB = 15000
DROPOUT = 0.4

# use cuda if possible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# data loading 
data = Stage4DatasetLoader().load_classification()
train_texts = data["train"]["X"]
train_labels = data["train"]["y"]
test_texts = data["test"]["X"]
test_labels = data["test"]["y"]
# vocab from training
vocab = build_vocab(train_texts, max_vocab_size=MAX_VOCAB)
print("vocab sz:", len(vocab))
train_data = ReviewDataset(train_texts, train_labels, vocab, max_length=MAX_LEN)
test_data = ReviewDataset(test_texts, test_labels, vocab, max_length=MAX_LEN)
# get loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_reviews)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_reviews)

# glove setting
glove_weights, matched = load_glove_matrix(get_glove_path(EMBEDDING_DIM), vocab, embedding_dim=EMBEDDING_DIM)


