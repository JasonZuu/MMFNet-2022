import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class X_Dataset(Dataset):
    def __init__(self, X):
        self.img_paths = X['img_path'].values
        self.X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        self.X_struc = StandardScaler().fit_transform(self.X_struc)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 160*160*3:# 需要放大的情况
            img = cv2.resize(img, (160, 160), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (160, 160), cv2.INTER_AREA)
        img = img.swapaxes(0,2).swapaxes(1,2)
        return torch.tensor(img, dtype=torch.float), torch.tensor(self.X_struc[index], dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]

class Xy_Dataset(Dataset):
    def __init__(self, X, y):
        self.img_paths = X['img_path'].values
        self.X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        self.X_struc = StandardScaler().fit_transform(self.X_struc)
        self.labels = y.values

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 160*160*3:# 需要放大的情况
            img = cv2.resize(img, (160, 160), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (160, 160), cv2.INTER_AREA)
        img = img.swapaxes(0,2).swapaxes(1,2)
        y = [1, 0] if self.labels[index]==0 else [0, 1]
        return torch.tensor(img, dtype=torch.float), torch.tensor(self.X_struc[index], dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]

class X_Dataset_ArcFace(Dataset):
    def __init__(self, X):
        self.img_paths = X['img_path'].values
        self.X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        self.X_struc = StandardScaler().fit_transform(self.X_struc)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 112*112*3:# 需要放大的情况
            img = cv2.resize(img, (112, 112), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (112, 112), cv2.INTER_AREA)
        img = img.swapaxes(0,2).swapaxes(1,2)
        return torch.tensor(img, dtype=torch.float), torch.tensor(self.X_struc[index], dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]

class Xy_Dataset_ArcFace(Dataset):
    def __init__(self, X, y):
        self.img_paths = X['img_path'].values
        self.X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        self.X_struc = StandardScaler().fit_transform(self.X_struc)
        self.labels = y.values

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 112*112*3:# 需要放大的情况
            img = cv2.resize(img, (112, 112), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (112, 112), cv2.INTER_AREA)
        img = img.swapaxes(0,2).swapaxes(1,2)
        y = [1, 0] if self.labels[index]==0 else [0, 1]
        return torch.tensor(img, dtype=torch.float), torch.tensor(self.X_struc[index], dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]


class X_Dataset_VGGFace(Dataset):
    def __init__(self, X):
        self.img_paths = X['img_path'].values
        self.X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        self.X_struc = StandardScaler().fit_transform(self.X_struc)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 224*224*3:# 需要放大的情况
            img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
        img = img.swapaxes(0,2).swapaxes(1,2)
        return torch.tensor(img, dtype=torch.float), torch.tensor(self.X_struc[index], dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]

class Xy_Dataset_VGGFace(Dataset):
    def __init__(self, X, y):
        self.img_paths = X['img_path'].values
        self.X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        self.X_struc = StandardScaler().fit_transform(self.X_struc)
        self.labels = y.values

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 224*224*3:# 需要放大的情况
            img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
        img = img.swapaxes(0,2).swapaxes(1,2)
        y = [1, 0] if self.labels[index]==0 else [0, 1]
        return torch.tensor(img, dtype=torch.float), torch.tensor(self.X_struc[index], dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]

if __name__ =="__main__":
    pass