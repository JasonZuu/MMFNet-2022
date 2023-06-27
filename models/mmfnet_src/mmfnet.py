import torch
import torch.nn as nn

from models.mmfnet_src.inception_resnet import InceptionResnet
from models.mmfnet_src.multi_attention import MultiAttention


class MMFNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = InceptionResnet(128)
        self.attn = MultiAttention(out_planes=512)
        self.output_block = nn.Sequential(
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, 128),
            nn.ELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, x_struc):
        x = self.backbone(img)
        x = x.unsqueeze(1)
        x_struc = x_struc.unsqueeze(-1)
        # 处理批量数据
        for i in range(x.shape[0]):
            tmp_x1 = torch.mm(x_struc[i], x[i])
            tmp_x1 = tmp_x1.unsqueeze(0)
            if i == 0:
                x1 = tmp_x1
            else:
                x1 = torch.cat((x1, tmp_x1), 0)
        # skip fusion
        x = torch.cat((x, x1), 1)
        # attn
        x = self.attn(x)
        x = self.output_block(x)
        return x


if __name__ == "__main__":
    img = torch.ones(2, 3, 224, 224)
    x_struc = torch.ones(2, 5)
    model = MMFNet(num_classes=2)
    out = model(img, x_struc)
    print(out)