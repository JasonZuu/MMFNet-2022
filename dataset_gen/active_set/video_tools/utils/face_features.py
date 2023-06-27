import os
import cv2
import numpy as np
from copy import deepcopy
import torch

def get_faces_features(model, faces, device, batch_size=32):
   """get_faces_features
   
   获取输入人脸的特征，用于聚类
   调用insightface.extract_feature()
   分批次提取全部人脸的特征，并返回存储其特征的np.arr,(Num, features)
   
   Args: 
      model(nn.Module): 提取特征的模型
      faces(list): 提取出来的人脸列表
      batch_size(int): 指定模型进行一次处理数据的大小，当小于该值时直接进行输入，否则分次输入
   
   Return: 
      features(np.array): 包含图片提取特征的矩阵，(N, features)
   
   @Author  :   JasonZuu
   @Time    :   2021/04/17 11:51:26
   """
   # 处理一下输入长度的问题
   inp_index = []
   left = len(faces)
   if left <= batch_size: # 输入人脸小于等于batch_size的情况，一次全部输入
      index = list(range(left))
      inp_index.append(deepcopy(index))
   else: # 输入人脸大于batch_size的情况，分成多次输入
      j = 0
      while left > batch_size:
         index = list(range(j*batch_size, (j+1)*batch_size))
         inp_index.append(deepcopy(index))
         j += 1
         left -= batch_size
      index = list(range(j*batch_size, len(faces)))
      inp_index.append(deepcopy(index))

   first = True
   _first = True # 指定是否为第一次循环
   features = np.array([])
   # 开始输入
   for index in inp_index:
      first = True
      for i in index:
         if faces[i].size < 112*112*3:# 需要放大的情况
            img = cv2.resize(faces[i], (112, 112), cv2.INTER_CUBIC)
         else: # 缩小的情况
            img = cv2.resize(faces[i], (112, 112), cv2.INTER_AREA)
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# 色彩空间转换
         img = np.transpose(img, (2, 0, 1))
         img = torch.from_numpy(img).unsqueeze(0).float()
         img.div_(255).sub_(0.5).div_(0.5)
         if first:
            inp = img
            first = False
         else:
            inp = torch.cat((inp, img), axis=0)
         if i==index[-1]:
            fs = model(inp.to(device))
            emb = fs.cpu().detach().numpy()
            norm = np.sqrt(np.sum(emb*emb)+0.00001)
            emb /= norm
            if _first:
               features = emb.copy()
               _first = False
            else:
               features = np.concatenate((features, emb), axis=0)
   return features

if __name__ == "__main__":
   pass
   