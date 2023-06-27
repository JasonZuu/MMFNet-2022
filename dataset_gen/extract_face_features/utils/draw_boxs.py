# python3.8.5
# -*- coding: utf-8 -*-
"""
@File    :   draw_boxs.py
@Time    :   2021/04/21 18:51:52
@Author  :   JasonZuu
@Contact :   zhumingcheng@stu.scu.edu.cn
"""
from copy import deepcopy
import cv2
import numpy as np
from DBface import common

def drawbox(image, bboxs, groups, colors, textcolor=(255, 255, 255), thickness=2):
    """drawbox
    
    将输入的图片使用bboxs进行画图，并写字，会对输入的image产生更改。
    只进行画框，不进行文字标注
    
    Args: 
       image(np.arr): 需要修改的图像
       bboxs(list): 存放人像信息的数据结构
       groups(list): #当前帧人脸对应的类
       N_colors(list): FR_system.N_colors,其中存放类别数量-对应颜色列表，如[3, (0,255,255)]
       textcolor(tuple): 输出文字颜色 默认值(0,0,0)
       thickness(int): 框线条粗细
    
    Return: 
    
    @Author  :   JasonZuu
    @Time    :   2021/04/17 19:52:55
    """
    for i in range(len(bboxs)):
        if groups[i] == -1: # -1类为认为应该抛弃的人脸图像
            continue
        # 框出人脸
        x, y, r, b = common.intv(bboxs[i].box)
        if r>image.shape[1]:
            r = image.shape[1]
        if b>image.shape[0]:
            b = image.shape[0]
        color = colors[groups[i]]
        cv2.rectangle(image, (x, y), (r, b), color, thickness, 4)
        # 输出文字
        w = r - x + 1
        h = b - y + 1
        border = thickness / 2
        pos = (x + 3, y - 5)
        cv2.rectangle(image, common.intv(x - border, y - 21, w + thickness, 21), color, -1, 4)
        cv2.putText(image, f"person:{groups[i]}", pos, 0, 0.5, textcolor, 1, 16)


def draw_IrreOrNorm(image, bboxs, groups, irre_groups, textcolor=(255, 255, 255), thickness=2):
    """drawbox
    
    将输入的图片使用bboxs进行画图, 区分无关有关人脸，会对输入的image产生更改。
    
    Args: 
       image(np.arr): 需要修改的图像
       bboxs(list): 存放人像信息的数据结构
       groups(list): #当前帧人脸对应的类
       irre_groups(list): 存放无关人组数的列表
       textcolor(tuple): 输出文字颜色 默认值(255, 255, 255)
       thickness(int): 框线条粗细
    
    Return: 
    
    @Author  :   JasonZuu
    @Time    :   2021/04/17 19:52:55
    """
    irre_color = (0, 0, 255)
    norm_color = (0, 255, 0)
    for i in range(len(bboxs)):
        if groups[i] == -1: # -1类为认为应该抛弃的人脸图像
            continue
        if groups[i] in irre_groups:
            # 框出人脸
            x, y, r, b = common.intv(bboxs[i].box)
            if r>image.shape[1]:
                r = image.shape[1]
            if b>image.shape[0]:
                b = image.shape[0]
            cv2.rectangle(image, (x, y), (r, b), irre_color, thickness, 4)
            # 输出文字
            w = r - x + 1
            h = b - y + 1
            border = thickness / 2
            pos = (x + 3, y - 5)
            cv2.rectangle(image, common.intv(x - border, y - 21, w + thickness, 21), irre_color, -1, 4)
            cv2.putText(image, f"irrevelent", pos, 0, 0.5, textcolor, 1, 16)
        else:
            # 框出人脸
            x, y, r, b = common.intv(bboxs[i].box)
            cv2.rectangle(image, (x, y), (r, b), norm_color, thickness, 4)
            # 输出文字
            w = r - x + 1
            h = b - y + 1
            border = thickness / 2
            pos = (x + 3, y - 5)
            cv2.rectangle(image, common.intv(x - border, y - 21, w + thickness, 21), norm_color, -1, 4)
            cv2.putText(image, f"normal", pos, 0, 0.5, textcolor, 1, 16)


def mask(pic,neighbor,h,w,x,y):
    '''
    @description: 每帧照片打码
    @param {pic:照片；neighbor:模糊度调节；h:高度；w:宽度；x:左上角横坐标；y:左上角纵坐标}
    '''
    for i in range(0, h , neighbor):  
        for j in range(0, w , neighbor):
            rect = [j + x, i + y]
            color = pic[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            x2=rect[0] + neighbor - 1   # 关键点2 减去一个像素
            y2=rect[1] + neighbor - 1
            if x2>x+w:
                x2=x+w
            if y2>y+h:
                y2=y+h
            right_down = (x2,y2)  
            cv2.rectangle(pic, left_up, right_down, color, -1) 


def protect_irre(image, bboxs, groups, irre_groups, textcolor=(255, 255, 255), thickness=2):
    """drawbox
    
    将输入的图片使用bboxs和无关人信息进行无关人打码，会对输入的image产生更改。
    
    Args: 
       image(np.arr): 需要修改的图像
       bboxs(list): 存放人像信息的数据结构
       groups(list): #当前帧人脸对应的类
       irre_groups(list): 存放无关人组数的列表
       textcolor(tuple): 输出文字颜色 默认值(255, 255, 255)
       thickness(int): 框线条粗细
    
    Return: 
    
    @Author  :   JasonZuu
    @Time    :   2021/04/17 19:52:55
    """
    irre_groups.append(-1)
    for i in range(len(bboxs)):
        if groups[i] in irre_groups:
            # 检测是否超出页面范围
            x, y, r, b = common.intv(bboxs[i].box)
            if r>image.shape[1]:
                r = image.shape[1]
            if b>image.shape[0]:
                b = image.shape[0]
            w = r - x + 1
            h = b - y + 1
            mask(image, 10, h,w,x,y)