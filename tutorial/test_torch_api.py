# third party
import sys
import pytest
from demo_data import load_adult_data
import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# data_iq absolute
from data_iq.dataiq_class import DataIQ_Torch


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class Example_NN(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=2):
        super(Example_NN, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = F.relu(self.dense3(X))
        X = F.softmax(self.output(X))
        return X


X_train, X_test, y_train, y_test, X, y = load_adult_data(split_size=0.3)

# Pre-processing & normalization
X_train = X_train.to_numpy().astype(np.float32)
y_train = y_train.values.astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_test = y_test.values.astype(np.float32)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

LEARNING_RATE = 0.01
EPOCHS = 10


@pytest.mark.skipif(sys.platform == "darwin", reason="libomp crash on OSX")
def test_torch_example() -> None:

    train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init model
    net = Example_NN(input_size=X_train.shape[1], nlabels=len(np.unique(y_train)))
    net.to(device)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    sf = nn.LogSoftmax()
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    dataiq = DataIQ_Torch(X=X_train, y=y_train, sparse_labels=True)

    for e in range(1, EPOCHS + 1):
        net.train()
        epoch_loss = 0
        epoch_acc = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = net(X_batch)
            _, predicted = torch.max(y_pred.data, 1)

            y_batch = y_batch.to(torch.int64)

            loss = criterion(sf(y_pred), y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (predicted == y_batch).sum().item() / len(y_batch)

        print(
            f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}",
        )

        dataiq.on_epoch_end(net, device=device)

    aleatoric_uncertainty = dataiq.aleatoric
    confidence = dataiq.confidence

    assert len(aleatoric_uncertainty) == len(X_train)
    assert len(confidence) == len(X_train)
