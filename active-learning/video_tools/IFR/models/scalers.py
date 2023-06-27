import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

class Struc_emb(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.scaler = nn.Sequential(
                        nn.Linear(5, 512),
                        nn.ReLU(),
                        nn.Linear(512,256),
                        nn.ReLU(),
                        nn.Linear(256, out_planes),
                        nn.ReLU()
        )

    def forward(self, x_struct):
        out = self.scaler(x_struct)
        return out

class clf_metric(nn.Module):
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