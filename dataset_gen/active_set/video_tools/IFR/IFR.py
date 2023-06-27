import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
import pandas as pd
from .models.Dim_attention import Self_Attention
from .models.focal_loss import FocalLoss
from .inception_resnet_v1 import InceptionResnetV1


BATCH_SIZE = 32
EPOCHS = 25
DEBUG_INTERVAL = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


class IFR(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Sequential(
                        nn.Linear(512,1024),
                        nn.ReLU(),
                        nn.Linear(1024,512),
                        nn.ReLU(),
                        nn.Linear(512,128),
                        nn.ReLU()
                        )
        self.Self_attention = Self_Attention(out_planes=512)
    
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
        X = self.Self_attention(X)
        X = self.emb(X)
        return X


class metric_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.clf = nn.Linear(128,2)

    def forward(self, X):
        X = self.clf(X)
        output = F.sigmoid(X)
        return output

class IFR_Model_Calling_System():
    def __init__(self, params_path="./IFR/IFR_params/IFR.params", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.name = "IFR"
        self.device = device
        self.extractor = InceptionResnetV1(pretrained='casia-webface',classify=True, num_classes=128)
        self.extractor.to(device)
        self.frame = IFR()
        self.frame.to(device)
        self.clf = metric_fc()
        self.clf.to(device)
        self._load_params(params_path)
        self.extractor.eval()
        self.frame.eval()
        self.clf.eval()


    def _load_params(self, params_path):
        """
        不需要指定每个模块的名字，只需要 {root}/{modelname}_{epoch}.pth即可
        """
        params_name, exp_name = os.path.splitext(params_path)
        self.extractor.load_state_dict(torch.load(f"{params_name}_extractor{exp_name}"))
        self.frame.load_state_dict(torch.load(f"{params_name}_frame{exp_name}"))
        self.clf.load_state_dict(torch.load(f"{params_name}_clf{exp_name}"))


    def call(self, img, X_struc):
        X0 = self.extractor(img.to(self.device))
        features = self.frame(X0, X_struc.to(self.device))
        outputs = self.clf(features)
        pred = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
        pred = pred.reshape(-1)
        return pred


if __name__ == "__main__":
    pass
