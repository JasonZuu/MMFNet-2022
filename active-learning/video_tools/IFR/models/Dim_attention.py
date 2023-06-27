import torch 
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.attention = nn.Linear(128, 3*128)
        self.ln_1 = nn.LayerNorm(256)
        self.forward_propagate = nn.Linear(256, 256)
        self.ln_2 = nn.LayerNorm(512)
        self.scaler = nn.Sequential(
                        nn.Linear(6*512, out_planes),
                        nn.ReLU()
        )

    def forward(self, x):
        assert (x.shape[1],x.shape[2]) == (6, 128)
        QKV = self.attention(x) # 这样每次只能处理一个数据
        Q = QKV[:, :, 0:128]
        K = QKV[:, :, 128:2*128]
        V = QKV[:, :, 2*128:3*128]
        for i in range(Q.shape[0]):
            tmp_attn = torch.mm(Q[i], K[i].t())
            tmp_result = torch.mm(tmp_attn, V[i])
            tmp_result = tmp_result.unsqueeze(0)
            if i ==0:
                attn_result = tmp_result
            else:
                attn_result = torch.cat((attn_result, tmp_result), 0)

        m = torch.cat((x, attn_result), 2)
        n = self.ln_1(m)
        n = self.forward_propagate(n)
        f = torch.cat((n, m), 2)
        f = self.ln_2(f)
        f = f.reshape(-1, 6*512)
        out = self.scaler(f)
        return out
