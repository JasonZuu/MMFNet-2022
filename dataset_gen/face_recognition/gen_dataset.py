# python3.9.1
# -*- coding: utf-8 -*-
"""
@File    :   gen_dataset.py
@Time    :   2021/04/22 16:06:59
@Author  :   JasonZuu
@Contact :   zhumingcheng@stu.scu.edu.cn
"""
import os
import cv2
from system import FR_system_cluster
import time

def get_video_paths(video_dir):
    """get_video_paths
    
    获取文件夹下全部视频路径
    
    Args: 
       video_dir(str): 存放视频的dir 
    
    Return: 
       video_paths(list): video_dir下全部视频路径存储的列表 
    
    @Author  :   JasonZuu
    @Time    :   2021/04/22 16:11:04
    """
    video_paths = []    
    for path, file_dir, files in os.walk(video_dir):
        for file in files:
            video_path = os.path.join(path, file)
            video_paths.append(video_path)
    return video_paths

def call_system(video_paths, save_dir, cluster_method='cdp_multiple', detect_fps=10, recog_threshold=0.45, log_path='./logs.txt'):
    """call_system
    
    功能说明
    
    Args: 
       arg1(int): 
    
    Return:  
    
    @Author  :   JasonZuu
    @Time    :   2021/04/22 16:19:18
    """
    f = open(log_path,'a')
    f.write(f"\n\n{time.ctime()}: NEW Project:\n")
    system = FR_system_cluster(save_dir, cluster_method)
    for path in video_paths:
        suc, reason = system.app(path, detect_fps, recog_threshold)
        basename = os.path.basename(path)
        if suc:
            log = f"{time.ctime()}:\t{basename} processed successfully\n"
            print(log)
            f.writelines(log)
        else:
            log = f"{time.ctime()}:\t{basename} fail to proc\tReason: {reason}\n"
            print(log)
            f.writelines(log)
    f.close()


if __name__ == "__main__":
    video_paths = get_video_paths('./raw_data/tiktok') 
    call_system(video_paths, './tiktok', 'cdp_multiple', detect_fps=15, recog_threshold=0.45)

