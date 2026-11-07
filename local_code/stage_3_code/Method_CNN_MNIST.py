# import packages
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import respective data loader
from Method_Dataset_Loader import Dataset_MNIST 

# SETUP

# use GPU if avail
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def load_mnist():
    train, test = Dataset_MNIST('train'), Dataset_MNIST('test') # from dataset loader class
    return (torch.from_numpy(train.data), torch.from_numpy(train.labels), torch.from_numpy(test.data), torch.from_numpy(test.labels))

d_train, l_train, d_test, l_test = load_mnist() # get data and label
train_loader = DataLoader(TensorDataset(d_train, l_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(d_test, l_test), batch_size=1000)

class CNN_MNIST(nn.Module):

    def __init__(self):
        super().__init__()

        # conv layers
        self.conv1 = nn.Conv2d(1, 25, 3, padding=1)
        self.conv2 = nn.Conv2d(25, 50, 3, padding=1)

        # others
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)       # half height and width
        self.dropout = nn.Dropout(0.2)

        # fully connected layers
        self.fc1 = nn.Linear(50 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)  

    def forward(self, x):
        # batch, 1, 28, 28
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  #14

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) 

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN_MNIST().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()


# TRAINING
loss_arr = []
epochs = 15
for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() * images.size(0)

    epoch_loss = total_loss / len(d_train)
    loss_arr.append(epoch_loss)
    print('epoch', epoch + 1, 'with loss', epoch_loss)

# generate learning curve
plt.plot(range(1, epochs + 1), loss_arr)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Learning curve')
plt.savefig('MNIST_learning_curve.png')


# TEST

model.eval()
prediction_arr = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        prediction_arr.append(predicted.cpu())

# convert to numpy arr
predictions = torch.cat(prediction_arr).numpy()
true_labels = l_test.numpy()

accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, s = precision_recall_fscore_support(true_labels, predictions, average='weighted')
print('')
print('accuracy:', accuracy)
print('precision:', precision)
print('recall:', recall)
print('f1:', f1)

