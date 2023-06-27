import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.output_layer = nn.Linear(6*512, out_planes)


    def forward(self, x):
        assert (x.shape[1], x.shape[2]) == (6, 128)
        attn0 = multi_attn_fusion(x, self.attention0, 32)
        attn1 = multi_attn_fusion(x, self.attention1, 32)
        attn2 = multi_attn_fusion(x, self.attention2, 32)
        attn3 = multi_attn_fusion(x, self.attention3, 32)
        attn_result = attn0
        attn_result = torch.cat((attn_result, attn1), 2)
        attn_result = torch.cat((attn_result, attn2), 2)
        attn_result = torch.cat((attn_result, attn3), 2)

        m = torch.cat((x, attn_result), 2)
        n = self.ln_1(m)
        n = self.forward_propagate(n)
        f = torch.cat((n, m), 2)
        f = self.ln_2(f)
        f = f.view(-1, 6*512)
        out = self.output_layer(f)
        return out


def multi_attn_fusion(x, attn_layer, matrix_size=32):
    """multi_attn_fusion

    用于使用注意力层融合特征

    Args: 
       x(int):
       attn_layer(pytorch.nn)
       matrix_size(int):注意力方阵的尺寸 

    Return: 
       attn_result(pytorch.tensor):shape = (batch_size, 6, matrix_size) 

    @Author  :   JasonZuu
    @Time    :   2021/09/18 00:02:00
    """
    QKV = attn_layer(x)
    Q = QKV[:, :, 0:matrix_size]
    K = QKV[:, :, matrix_size:2*matrix_size]
    V = QKV[:, :, 2*matrix_size:3*matrix_size]
    for i in range(Q.shape[0]):
        tmp_attn = torch.mm(Q[i], K[i].t())
        tmp_result = torch.mm(tmp_attn, V[i])
        tmp_result = tmp_result.unsqueeze(0)
        if i == 0:
            attn_result = tmp_result
        else:
            attn_result = torch.cat((attn_result, tmp_result), 0)
    return attn_result
