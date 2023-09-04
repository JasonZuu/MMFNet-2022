import torch 
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveModule(nn.Module):
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



class StrucEmb(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.scaler = nn.Sequential(
                        nn.Linear(5, 512),
                        nn.ReLU(),
                        nn.Linear(512, out_planes),
                        nn.ReLU()
        )
    def forward(self, x_struct):
        out = self.scaler(x_struct)
        return out


class FusionLayers(nn.Module):
    _default_img_planes = {
        "vggface_f": 2622,
        "resnet_f": 1000,
        "facenet_f": 512,
        "arcface_f": 512,
    }
    def __init__(self, out_planes: int, img_encoder_name: str):
        assert img_encoder_name in self._default_img_planes.keys(), \
            f"img_encoder_name must be one of {self._default_img_planes.keys()}"
        img_planes = self._default_img_planes[img_encoder_name]
        super().__init__()
        self.input_scaler = nn.Sequential(
            nn.ReLU(),
            nn.Linear(img_planes, out_planes//2),
            nn.ReLU(),
        )
        self.struc_emb = StrucEmb(out_planes//2)

    def forward(self, X0, X_struc):
        X0 = self.input_scaler(X0)
        X1 = self.struc_emb(X_struc)
        X = torch.cat((X0, X1), axis=1)
        return X


class ClassifyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
                        nn.Linear(256,512),
                        nn.ReLU(),
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,2),
                        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_sigmoid=True):
        x = self.mlp(x)
        if use_sigmoid:
            x = self.sigmoid(x)
        return x