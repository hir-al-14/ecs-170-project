# import package
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import modules
from local_code.stage_4_code.evaluate_classification import evaluate_classification
from local_code.stage_4_code.Dataset_Loader import Stage4DatasetLoader
from local_code.stage_4_code.Text_Preprocessor import (build_vocab, ReviewDataset, collate_reviews, load_glove_matrix)
from local_code.stage_4_code.Method_RNN_Classification import RNNClassifier
from local_code.stage_4_code.Config import (EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, MAX_SEQUENCE_LENGTH, get_glove_path)
# setting
MAX_LEN = MAX_SEQUENCE_LENGTH
EPOCHS = MAX_EPOCHS
HIDDEN_SIZE = HIDDEN_DIM
MAX_VOCAB = 15000
DROPOUT = 0.4
LEARNING_RATE = 0.0005
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
print("train reviews:", len(train_texts), "test reviews:", len(test_texts))
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

# model
model = RNNClassifier(glove_weights, hidden_SZ=HIDDEN_SIZE, dropout=DROPOUT).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x, lengths)
            loss = loss_fn(output, y)
            total_loss += loss.item() * len(y)
            correct += (output.argmax(dim=1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total

# training
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, lengths, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x, lengths)
        loss = loss_fn(output, y)
        loss.backward()
        # keep gradient stable
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (output.argmax(dim=1) == y).sum().item()
        total += len(y)

    train_loss = total_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(test_loader)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # display
    print("epoch %d : train loss %.4f, accuracy %.4f ; test loss %.4f, accuracy %.4f" % (epoch + 1, train_loss, train_acc, test_loss, test_acc))

# save res
from pathlib import Path
result_dir = Path(__file__).resolve().parents[2] / "result" / "stage_4_result"
result_dir.mkdir(parents=True, exist_ok=True)
his = {"train_loss": train_loss_list,"train_acc": train_acc_list,"test_loss": test_loss_list,"test_acc": test_acc_list}
with open(result_dir / "classification_training_log.json", "w") as f:
    json.dump(his, f)
# save model
torch.save(model.state_dict(), result_dir / "rnn_classification.pt")

# Use the shared evaluator for the final trained classifier.
y_true, y_pred = [], []
model.eval()
with torch.inference_mode():
    for x, lengths, y in test_loader:
        predictions = model(x.to(device), lengths).argmax(dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(predictions.cpu().tolist())
metrics = evaluate_classification(y_true, y_pred)
with open(result_dir / "rnn_classification_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Final classification metrics:", metrics)
