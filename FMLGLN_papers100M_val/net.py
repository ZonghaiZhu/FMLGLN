# coding:utf-8
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, d_in=512, classes=121, single_class=False):
        super(Net, self).__init__()
        self.single_class = single_class
        self.fc = nn.Sequential(
            nn.Linear(d_in, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(1024, classes)

    def forward(self, inputs):
        feats = self.fc(inputs)
        outputs = self.fc2(feats)
        if self.single_class:
            return outputs
        else:
            return torch.sigmoid(outputs)


class NetBN(nn.Module):
    def __init__(self, d_in=512, classes=121, single_class=False):
        super(NetBN, self).__init__()
        self.single_class = single_class
        self.fc = nn.Sequential(
            nn.Linear(d_in, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(1024, classes)

    def forward(self, inputs):
        feats = self.fc(inputs)
        outputs = self.fc2(feats)
        if self.single_class:
            return outputs
        else:
            return torch.sigmoid(outputs)