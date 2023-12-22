import torch
import torch.nn as nn

from mmfnet.inception_resnet import InceptionResnetV1
from mmfnet.attn import MultiAttnFusionModule


class MMFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = InceptionResnetV1(pretrained='vggface2')
        self.fusion_layers = MultiAttnFusionModule(256)
        self.cls_head = nn.Sequential(nn.Linear(256, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 1))
        self._freeze_img_encoder()

    def forward(self, imgs, X_struc, output_logits=True):
        img_embeddings = self.img_encoder(imgs)
        features = self.fusion_layers(img_embeddings, X_struc)
        y_logits = self.cls_head(features)
        if not output_logits:
            y_logits = torch.sigmoid(y_logits)
        return y_logits

    def _freeze_img_encoder(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = False

