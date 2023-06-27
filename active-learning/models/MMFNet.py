import torch
import torch.nn as nn
from models.Multi_attention import Multi_Attention

class MMFNet(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.Multi_attention = Multi_Attention(out_planes=512)
    
    def forward(self, X0, X_struc):
        X0 = X0.unsqueeze(1)
        X_struc = X_struc.unsqueeze(-1)
        # 处理批量数据
        for i in range(X0.shape[0]):
            tmp_X1 = torch.mm(X_struc[i],X0[i])
            tmp_X1 = tmp_X1.unsqueeze(0)
            if i ==0:
                X1 = tmp_X1
            else:
                X1 = torch.cat((X1, tmp_X1), 0)

        X = torch.cat((X0, X1), 1)
        X = self.Multi_attention(X)
        return X