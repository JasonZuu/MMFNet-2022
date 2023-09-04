import torch
import torch.nn as nn


class MultiAttention(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.attention0 = nn.Linear(128, 3*32)
        self.attention1 = nn.Linear(128, 3*32)
        self.attention2 = nn.Linear(128, 3*32)
        self.attention3 = nn.Linear(128, 3*32)
        self.ln_1 = nn.LayerNorm(256)
        self.forward_propagate = nn.Linear(256, 256)
        self.ln_2 = nn.LayerNorm(512)
        self.scaler = nn.Sequential(
                        nn.Linear(6*512, out_planes),
                        nn.ReLU()
        )

    def forward(self, x):
        assert (x.shape[1],x.shape[2]) == (6, 128)
        attn0 = self._multi_attn_fusion(x, self.attention0, 32)
        attn1 = self._multi_attn_fusion(x, self.attention1, 32)
        attn2 = self._multi_attn_fusion(x, self.attention2, 32)
        attn3 = self._multi_attn_fusion(x, self.attention3, 32)

        attn = torch.cat((attn0, attn1, attn2, attn3), 2)
        m = torch.cat((x, attn), 2)
        n = self.ln_1(m)
        n = self.forward_propagate(n)
        f = torch.cat((n, m), 2)
        f = self.ln_2(f)
        f = f.reshape(-1, 6*512)
        out = self.scaler(f)
        return out

    def _multi_attn_fusion(self, x, attn_layer, matrix_size:int):
        """multi_attn_fusion
        Args:
           x(int):
           attn_layer(torch.nn.Module)
           matrix_size(int):注意力方阵的尺寸

        Return:
           attn_result(pytorch.tensor):shape = (batch_size, 6, matrix_size)

        @Author  :   JasonZuu
        @Time    :   2021/09/18 00:02:00
        """
        QKV = attn_layer(x)
        Q = QKV[:, :, 0:matrix_size]
        K = QKV[:, :, matrix_size:2 * matrix_size]
        V = QKV[:, :, 2 * matrix_size:3 * matrix_size]
        for i in range(Q.shape[0]):
            tmp_attn = torch.mm(Q[i], K[i].t())  # 注意力矩阵
            tmp_result = torch.mm(tmp_attn, V[i])  # 注意力矩阵和原始输入相乘
            tmp_result = tmp_result.unsqueeze(0)
            if i == 0:
                attn_result = tmp_result
            else:
                attn_result = torch.cat((attn_result, tmp_result), 0)
        return attn_result


class MMFNet(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.Multi_attention = MultiAttention(out_planes=512)
    
    def forward(self, X0, X_struc):
        X0 = X0.unsqueeze(1)
        X_struc = X_struc.unsqueeze(-1)
        # batch processing
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
    