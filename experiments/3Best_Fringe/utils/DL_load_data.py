import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dl_load_data(csv_path, random_seed, test_size=0.2):
    """load_data
    
    从指定路径中加载数据，并获取其训练集、测试集标签
    深度学习DL专用
    
    Args: 
       csv_path(str): csv数据集路径
       test_size(float): 测试集占整个数据集占比 
    
    Return: 
       X_train, X_test, y_train, y_test 
    
    @Author  :   JasonZuu
    @Time    :   2021/05/05 19:36:04
    """
    datas = pd.read_csv(csv_path, sep=',')
    # 获取并处理X
    X = datas[['img_path', 'frequency', 'size', 'relative_size', 'acu', 'relative_acu']]
    y = datas.labels
    X_train, X_very, y_train, y_very = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    return X_train, X_very, y_train, y_very

def irregular_samples_split(X_test, y_test):
    """irregular_samples_split
    
    将测试集划分为数据部规则的2个集合
    
    Args: 
       X_test, y_test
    
    Return: 
       X_Tsample(dataframe): 正样本占90%集合的数据
       X_Fsample(dataframe): 负样本占90%集合的数据
       y_Tsample(dataframe): 
       y_Fsample(dataframe): 
    
    @Author  :   JasonZuu
    @Time    :   2021/10/14 00:48:36
    """
    T_index = y_test.loc[y_test==1].index
    F_index = y_test.loc[y_test==0].index
    X_T1 = X_test.loc[T_index[:9000]]
    X_T2 = X_test.loc[F_index[:1000]]
    y_T1 = y_test.loc[T_index[:9000]]
    y_T2 = y_test.loc[F_index[:1000]]
    X_Tsample = X_T1.append(X_T2, ignore_index=True)
    y_Tsample = y_T1.append(y_T2, ignore_index=True)
    X_F1 = X_test.loc[T_index[9000:]]
    X_F2 = X_test.loc[F_index[1000:]]
    y_F1 = y_test.loc[T_index[9000:]]
    y_F2 = y_test.loc[F_index[1000:]]
    X_Fsample = X_F1.append(X_F2, ignore_index=True)
    y_Fsample = y_F1.append(y_F2, ignore_index=True)
    return X_Tsample, X_Fsample, y_Tsample, y_Fsample

    
class ArcFace_dataset(Dataset):
    def __init__(self, X, y):
        self.img_paths = X['img_path'].values
        X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        Sscaler = StandardScaler()
        self.X_struc = Sscaler.fit_transform(X_struc)
        self.labels = y.values

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img.size < 112*112*3:# 需要放大的情况
            img = cv2.resize(img, (112, 112), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(img, (112, 112), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# 色彩空间转换
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
        y = [1, 0] if self.labels[index]==0 else [0, 1]
        return img, torch.tensor(self.X_struc[index], dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]


class FaceNet_clf_dataset(Dataset):
    def __init__(self, X, y):
        self.img_paths = X['img_path'].values
        X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        Sscaler = StandardScaler()
        self.X_struc = Sscaler.fit_transform(X_struc)
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

class VGG_dataset(Dataset):
    def __init__(self, X, y):
        self.img_paths = X['img_path'].values
        X_struc = X[['frequency', 'size', 'relative_size', 'acu', 'relative_acu']].values
        Sscaler = StandardScaler()
        self.X_struc = Sscaler.fit_transform(X_struc)
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
    X_train, X_very, X_test, y_train, y_very, y_test = dl_load_data("./dataset/structure.csv",1, test_size=0.2)
    irregular_samples_split(X_test, y_test)   