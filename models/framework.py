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

from loss_fn.focal_loss import FocalLoss
from utils.DL_load_data import ArcFace_dataset, dl_load_data, split_test_verification


import pandas as pd


class IFRFramework(nn.Module):
    def __init__(self,
                 img_encoder: nn.Module,
                 fusion_layers: nn.Module,
                 classify_layers: nn.Module):
        super().__init__()
        self.img_encoder = img_encoder
        self.fusion_layers = fusion_layers
        self.classify_layers = classify_layers

    def forward(self, imgs, X_struc):
        X0 = self.img_encoder(imgs)
        features = self.fusion_layers(X0, X_struc)
        return self.classify_layers(features)

    def freeze_img_encoder(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = False

    def unfreeze_img_encoder(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = True


def test_IFRFramework():
    pass


if __name__ == "__main__":
    test_IFRFramework()
