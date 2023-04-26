# coding:utf-8
import torch
import torchvision
import torch.utils.data as data
import numpy as np
import os


class TransData(data.Dataset):
    def __init__(self, args, feats, targets, dim):
        self.args = args
        self.feats = feats
        self.dim = dim
        self.targets = targets

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, item):
        temp_data = torch.tensor(self.feats[item], dtype=torch.float32).view(-1, self.dim)
        temp_data = temp_data.view(-1, self.dim)
        return temp_data, self.targets[item]


class GraphData(data.Dataset):
    def __init__(self, args, feat1, feat2, feat3, targets):
        self.args = args
        self.feats = torch.cat((torch.tensor(feat1), torch.tensor(feat2), torch.tensor(feat3)),dim=1)
        self.targets = targets

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, item):
        return self.feats[item], self.targets[item]


class GraphData4(data.Dataset):
    def __init__(self, args, feat1, feat2, feat3, feat4, targets):
        self.args = args
        self.feats = torch.cat((torch.tensor(feat1), torch.tensor(feat2), torch.tensor(feat3),  torch.tensor(feat4)),dim=1)
        self.targets = targets

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, item):
        return self.feats[item], self.targets[item]