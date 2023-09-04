import torch 
import torch.nn as nn
import torch.nn.functional as F


class Adaptive_module(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.emb = nn.Sequential(
                        nn.Linear(in_planes,1024),
                        nn.ReLU(),
                        nn.Linear(1024,512),
                        nn.ReLU(),
                        nn.Linear(512,128),
                        nn.ReLU()
                        )
        self.clf = nn.Linear(128, 2)

    def forward(self, X):
        X = self.emb(X)
        X = self.clf(X)
        output = torch.sigmoid(X)
        return output



class StrucEmb(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.scaler = nn.Sequential(
                        nn.Linear(5, 512),
                        nn.ReLU(),
                        nn.Linear(512, out_planes),
                        nn.ReLU()
        )
    def forward(self, x_struct):
        out = self.scaler(x_struct)
        return out


class ClfMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.clf = nn.Sequential(
                        nn.Linear(256,512),
                        nn.ReLU(),
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,2)
                        )

    def forward(self, X):
        X = self.clf(X)
        output = F.sigmoid(X)
        return output
