# python3.8.5
# -*- coding: utf-8 -*-
"""
@File    :   matrix_change.py
@Time    :   2021/04/17 17:21:13
@Author  :   JasonZuu
@Contact :   zhumingcheng@stu.scu.edu.cn
"""
import numpy as np

def normalization(data):
    """normalization
    
    返回矩阵的归一化结果
    
    Args: 
       data(np.arr): 一个一维矩阵 
    
    Return: 
       nor_arr(np.arr): 归一化的矩阵
    
    @Author  :   JasonZuu
    @Time    :   2021/04/17 17:21:48
    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def unitize(data):
    """unitize
    
    返回输入向量的单位向量
    
    Args: 
       data(np.arr): 一个一维矩阵 
    
    Return: 
       unit(np.arr): 输入向量的单位向量
    
    @Author  :   JasonZuu
    @Time    :   2021/04/17 17:29:16
    """
    length = np.linalg.norm(data, axis=0)
    unit = data / length
    # length1 = np.linalg.norm(unit, axis=0)
    # print(length1)
    return unit.copy()

def cosin_distance(face_to_compare, features):
   """cosin_distance
   
   计算输入向量和特征向量集的余弦距离，并输出一张(1,N)的矩阵
   
   Args: 
      face_to_compare(np.arr): 用来进行对比的脸部特征向量， shape=(1, 512)
      features(list): 存储特征的列表，shape=(N, 512) 
   
   Return: 
      cosin_arr(np.arr): 对比后的余弦距离,shape=(N,)  其中元素和features对应
   
   @Author  :   JasonZuu
   @Time    :   2021/04/17 18:06:55
   """
   cosin_list = []
   for i in range(len(features)):
      dot = np.dot(features[i], face_to_compare) 
      cos = dot/(np.linalg.norm(features[i])*np.linalg.norm(face_to_compare))
      sim = 0.5  + 0.5*cos # 归一化
      cosin_list.append(sim)
   cosin_arr = np.array(cosin_list)
   return cosin_arr.copy()



    
