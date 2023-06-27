import torch 
import torch.nn as nn


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