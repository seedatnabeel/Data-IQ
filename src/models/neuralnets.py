# third party
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net1(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=2):
        super(Net1, self).__init__()

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


class Net2(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=2):
        super(Net2, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 16)
        self.dense2 = nn.Linear(16, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = F.softmax(self.output(X))
        return X


class Net3(nn.Module):  # old module4
    def __init__(self, input_size=12, num_units=256, nonlin=F.relu, nlabels=2):
        super(Net3, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 16)
        self.dense5 = nn.Linear(16, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = F.relu(self.dense3(X))
        X = F.relu(self.dense4(X))
        X = F.relu(self.dense5(X))
        X = F.softmax(self.output(X))
        return X


class Net4(nn.Module):  # old module3
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=2):
        super(Net4, self).__init__()

        self.dense0 = nn.Linear(input_size, 16)
        self.dense1 = nn.Linear(16, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


class Net5(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=2):
        super(Net5, self).__init__()

        self.dense0 = nn.Linear(input_size, 16)
        self.nonlin = nonlin
        self.output = nn.Linear(16, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.softmax(self.output(X))
        return X
