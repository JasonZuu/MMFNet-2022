import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from DBface.DBFace import DBFace
from DBface import common
from copy import deepcopy
import os 
from DBface.common import detect
from copy import deepcopy

HAS_CUDA = True

def get_DBface(params_path):
    """
    @Author  :   JasonZuu
    @Time    :   2021/03/31 09:40:06
    :params: params_path: 模型参数的路径
    :return: model： 加载完参数的模型
    """
    dbface = DBFace()
    dbface.eval()
    if HAS_CUDA:
        dbface.cuda()
    dbface.load(params_path)
    return dbface

def pull_face_return(image, bboxs, size_threshold=1):
    """
    对图片中的人脸进行截取，并返回人脸图像列表
    @Author  :   JasonZuu
    @Time    :   2021/03/31 17:48:28
    :params: image: cv2读取的图片
    :params: bboxs: detect获取的图像中人脸检测结果
    :return: faces: 包含当前帧人脸的列表
    """
    faces = []
    for bbox in bboxs:
        x, y, r, b = common.intv(bbox.box)
        cropped = image[y:b, x:r]
        if cropped.size > size_threshold*3:
            faces.append(deepcopy(cropped))
    return faces

def get_face_landmarks(model, img):
    """
    获取一张图片中所有人脸的5个特征点
    return： 
    face_landmarks: 包含全部人脸的坐标点的列表，其中元素为face_landmark，为一张人脸的坐标点。坐标点共有5个，格式为[x,y]，分别对应左眼右眼鼻子左嘴角右嘴角
    """
    face_landmarks = []
    bboxs = detect(model, img, threshold=0.3)
    for bbox in bboxs:
       xxxxxyyyyy = bbox.xxxxxyyyyy_cat_landmark
       face_landmark = []
       for i in range(5):
           point = []
           point.append(xxxxxyyyyy[i])
           point.append(xxxxxyyyyy[i+5])
           face_landmark.append(deepcopy(point))
       face_landmarks.append(deepcopy(face_landmark))
    return face_landmarks
    

if __name__ == "__main__":
    pass
