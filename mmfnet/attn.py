import torch
import torch.nn as nn


class MultiAttentionBlock(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.attention0 = nn.Linear(512, 3*128)
        self.attention1 = nn.Linear(512, 3*128)
        self.attention2 = nn.Linear(512, 3*128)
        self.attention3 = nn.Linear(512, 3*128)
        self.ln_1 = nn.LayerNorm(1024)
        self.forward_propagate = nn.Linear(1024, 256)
        self.ln_2 = nn.LayerNorm(256)
        self.scaler = nn.Sequential(
                        nn.Linear(6*256, out_planes),
                        nn.ReLU()
        )

    def forward(self, x):
        assert (x.shape[1], x.shape[2]) == (6, 512), \
            f"input shape must be (batch_size, 6, 512), but got {x.shape}"
        attn0 = self._multi_attn_fusion(x, self.attention0, 128)
        attn1 = self._multi_attn_fusion(x, self.attention1, 128)
        attn2 = self._multi_attn_fusion(x, self.attention2, 128)
        attn3 = self._multi_attn_fusion(x, self.attention3, 128)

        attn = torch.cat((attn0, attn1, attn2, attn3), 2)
        m = torch.cat((x, attn), 2)
        n = self.ln_1(m)
        n = self.forward_propagate(n)
        f = self.ln_2(n)
        f = f.reshape(-1, 6*256)
        out = self.scaler(f)
        return out

    def _multi_attn_fusion(self, x, attn_layer, matrix_size: int):
        """multi_attn_fusion

        @Author  :   JasonZuu
        @Time    :   2021/09/18 00:02:00
        """
        QKV = attn_layer(x)
        Q = QKV[:, :, 0:matrix_size]
        K = QKV[:, :, matrix_size:2 * matrix_size]
        V = QKV[:, :, 2 * matrix_size:3 * matrix_size]
        for i in range(Q.shape[0]):
            tmp_attn = torch.mm(Q[i], K[i].t())  # attn map
            tmp_result = torch.mm(tmp_attn, V[i])  # pay attention
            tmp_result = tmp_result.unsqueeze(0)
            if i == 0:
                attn_result = tmp_result
            else:
                attn_result = torch.cat((attn_result, tmp_result), 0)
        return attn_result


class MultiAttnFusionModule(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.attn = MultiAttentionBlock(out_planes=out_planes)

    def forward(self, image_embedding, X_struc):
        image_embedding = image_embedding.unsqueeze(1)
        X_struc = X_struc.unsqueeze(-1)
        # batch processing
        for i in range(image_embedding.shape[0]):
            tmp_X1 = torch.mm(X_struc[i], image_embedding[i])
            tmp_X1 = tmp_X1.unsqueeze(0)
            if i == 0:
                X1 = tmp_X1
            else:
                X1 = torch.cat((X1, tmp_X1), 0)

        X = torch.cat((image_embedding, X1), 1)
        Z = self.attn(X)
        return Z
