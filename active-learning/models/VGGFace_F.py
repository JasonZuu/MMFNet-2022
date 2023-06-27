import torch
import torch.nn as nn
from models.baseline_fusion import Struc_emb

class VGGFace_F(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.input_scaler = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(2622, int(out_planes/2)),
                        nn.ReLU(),
        )
        self.struc_emb = Struc_emb(int(out_planes/2))
    
    def forward(self, X0, X_struc):
        X0 = self.input_scaler(X0)
        X1 = self.struc_emb(X_struc)
        X = torch.cat((X0, X1), axis=1)
        return X