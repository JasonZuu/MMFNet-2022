import os
import cv2
import numpy as np
from copy import deepcopy
# from insightface_slim.face_model import get_insightface


def get_faces_features(model, faces, batch_size=32):
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

   _first = True # 指定是否为第一次循环
   features = np.array([])
   # 开始输入
   for index in inp_index:
      inp = []
      for i in index:
         face = model.get_input(faces[i])
         inp.append(deepcopy(face))
         if i==index[-1]:
            inp = np.array(inp)
            fs = model.get_feature(inp)
            if _first:
               features = fs.copy()
               _first = False
            else:
               features = np.concatenate((features, fs), axis=0)
   return features

if __name__ == "__main__":
   model = get_insightface(0)
   # f, _ = get_imgs_features(model, "./datas/1")
   # print(f.shape)
   