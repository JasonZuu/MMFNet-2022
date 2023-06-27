"""
@File    :   system.py
@Time    :   2021/04/19 17:35:45
@Author  :   JasonZuu
@Contact :   zhumingcheng@stu.scu.edu.cn
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from copy import deepcopy
import os 
from random import randint
from moviepy.editor import *
from video_tools.DBface import common
from video_tools.arcface.backbones import get_model
from video_tools.utils.pull_faces import get_DBface, pull_face_return
from video_tools.utils.face_features import get_faces_features
from video_tools.utils.draw_boxs import drawbox, draw_IrreOrNorm, protect_irre
from video_tools.cluster.chinese_whisper import cw_cluster
from video_tools.cluster.cdp.call_cdp import S_cdp_cluster, M_cdp_cluster
from video_tools.audio import add_audio                                                            

class FR_system():
    def __init__(self, save_dir, device=torch.device("cuda:7" if torch.cuda.is_available() else "cpu"), cluster_method='cdp_multiple', DBface_params_path="./video_tools/DBface/dbface.pth", insightface_path="./video_tools/arcface/backbone.pth"):
        """FR_system.__init__
        
        Args: 
           save_dir(str): 存储生成内容(聚类图像、视频)的文件夹 
           cluster_method(str): 目前只能选择chinese_whisper, cdp_multiple, cdp_single, 如果不在这三个里面，默认会变成cdp_multiple
           DBface_params_path(str): 存储DBface参数的文件夹
           insightface_path(int): insightface使用的gpu
           
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/04/19 20:43:21
        """
        if not os.path.exists(save_dir):
            cwd = os.getcwd()
            os.mkdir(cwd+'/'+save_dir)
        self.save_dir = save_dir
        self.dbface = get_DBface(DBface_params_path, device=device)
        self.insightface = get_model("r100", fp16=True)
        self.insightface.load_state_dict(torch.load(insightface_path, map_location="cpu"))
        self.insightface.eval()
        self.insightface.to(device)
        self.device = device
        self.clear()
        # 聚类方法判断。输入为feat_matrix:(N, feats), 输出为faces_group(list):(N,)其中每个值为face的组
        if cluster_method == "cdp_multiple":
            self.cdp = M_cdp_cluster()
            self.cluster = self.cdp.cluster
        elif cluster_method == "cdp_single":
            self.cdp = S_cdp_cluster()
            self.cluster = self.cdp.cluster
        elif cluster_method == "chinese_whisper":
            self.cluster = cw_cluster
        else:
            self.cdp = M_cdp_cluster()
            self.cluster = self.cdp.cluster
        
    @torch.no_grad()
    def _prepare(self, video_path, detect_fps=15, recog_threshold=0.4):
        """prepare
        
        FR_system聚类前的准备工作,识别人脸并提取特征，放入self.feature_matrix
        同时生成存储视频相关内容的文件
        调用pull_face_return()
        读取路径中的视频，按detect_fps的频率进行检测，按recog_threshold进行人脸识别
        
        Args: 
           video_path(str): 检测视频路径
           detect_fps(int): 检测频率(x帧/s)
           recog_threshold(float): DBface识别人脸时所使用的置信度阈值
        
        Return: 
            success(bool): 准备是否成功
            reason(str): 失败的原因，成功返回空字符串
            
        
        @Author  :   JasonZuu
        @Time    :   2021/04/17 20:00:41
        """
        if not self.CLEAR:
            self.clear()
        
        # 视频是否已经处理过检测
        basename = os.path.basename(video_path)
        video_name, _ = os.path.splitext(basename)
        self.save_video_dir = f"{self.save_dir}/{video_name}"
        self.video_path = video_path
        if not os.path.exists(self.save_video_dir):
            cwd = os.getcwd()
            os.mkdir(cwd+'/'+self.save_video_dir)

        Cap = cv2.VideoCapture(self.video_path)
        F_num = int(Cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps=Cap.get(cv2.CAP_PROP_FPS)
        seq_num = int(raw_fps / detect_fps) #进行检测的帧
        if seq_num < 1:
            seq_num = 1
        # video_time = F_num/raw_fps
        # if video_time<10 or video_time>60:
        #     return False, "Video's duration is not between 10s and 60s."
        success, img = Cap.read() # 第一张图片
        i = 0
        i_frame = -1
        while success:
            i_frame += 1
            if i%seq_num == 0:
                objs = common.detect(self.dbface, img, device=self.device, threshold=recog_threshold)
                faces = pull_face_return(img, objs)
                if len(objs) != len(faces): # 不相等时，说明dbface人脸识别出现问题
                    success, img = Cap.read()
                    continue
                self.bboxs_list.append(objs)
                self.faces_frame.extend([i_frame for i_tmp in range(len(faces))])
                self.faces.extend(faces)
                self.proc_frames.append(deepcopy(i_frame))
            success, img = Cap.read()
            i += 1
        self.feature_matrix = get_faces_features(self.insightface, self.faces, self.device)
        self.CLEAR = False
        return True, ''

    def _cluster(self):
        """cluster
        
        调用self.cluster，进行聚类，同时根据聚类类别数，生成不同类颜色
        
        Args: 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/04/21 09:55:48
        """
        self.faces_group = self.cluster(self.feature_matrix)
        groups = set(self.faces_group)
        for i_group in groups:
            color = (randint(0,255),randint(0,255),randint(0,255))
            self.colors.append(color)
                
    
    def _write_img(self):
        """write_img
        self.colors
        将聚类后的系统内容写出到文件夹
        
        Args: 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/04/21 10:06:58
        """
        # 处理文件夹
        dir_list = list(set(self.faces_group))
        for group in dir_list:
            if group == -1:
                continue
            group_dir = f"{self.save_video_dir}/{group}"
            if not os.path.exists(group_dir):
                os.mkdir(group_dir)

        # 处理人脸图片
        for i_face in range(len(self.faces)):
            save_path = f"{self.save_video_dir}/{self.faces_group[i_face]}/{i_face}_{self.faces_frame[i_face]}.jpg"
            cv2.imwrite(save_path, self.faces[i_face])

        
    def face_recognize(self, video_path, detect_fps=15, recog_threshold=0.44):
        """face_recognize
        
        调用self.prepare(),self.cluster(), self.write_img(),self.write_video(), self.clear()
        实现系统完整的工作任务
        
        Args: 
           video_path(str): 检测视频路径
           detect_fps(int): 检测频率(x帧/s)
           recog_threshold(float): DBface识别人脸时所使用的置信度阈值
        
        Return: 
            success(bool): 表示该视频是否处理成功
            reason(str): 视频处理失败的原因

        @Author  :   JasonZuu
        @Time    :   2021/04/22 14:54:02
        """
        suc_prepare, suc_reason = self._prepare(video_path,  detect_fps, recog_threshold)
        if not suc_prepare:
            return False, suc_reason
        elif self.feature_matrix.size <= (10*512): #说明视频中没有人脸或人脸太少，无法构成有效数据集
            return False, 'There is no face or not enough face to construct an effective dataset in the videos'
        print(self.feature_matrix.shape)
        self._cluster()
        self._write_img()
        return True, ''

    def mark_irre_faces(self, irre_groups):
        """irre_recognize
        
        结合无关人识别的结果，将无关人与有关人分别标出
        
        Args: 
           irre_groups(list): 无关人所属组
        
        Return: 
            success(bool): 表示该视频是否处理成功
            reason(str): 视频处理失败的原因

        @Author  :   JasonZuu
        @Time    :   2021/04/22 14:54:02
        """
        Cap = cv2.VideoCapture(self.video_path)
        raw_fps=Cap.get(cv2.CAP_PROP_FPS)
        raw_width = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        basename = os.path.basename(self.video_path)
        name, _ = os.path.splitext(basename)
        tmp_save_path = f"{self.save_video_dir}/{name}_tmp.mp4"
        save_path = f"{self.save_video_dir}/{name}_IR.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Videowriter = cv2.VideoWriter(tmp_save_path, fourcc, raw_fps, (raw_width,raw_height))
        
        i_frame = 0
        i_bbox = 0
        i_face = 0
        success, img = Cap.read()
        bboxs = self.bboxs_list[0] # 防止第一帧没有被检测
        groups = self.faces_group[0:len(bboxs)]
        while success:
            if i_frame in self.proc_frames:
                bboxs = self.bboxs_list[i_bbox]
                groups = self.faces_group[i_face:i_face+len(bboxs)]
                i_face += len(bboxs)
                i_bbox += 1
            draw_IrreOrNorm(img, bboxs, groups, irre_groups)
            Videowriter.write(img)
            success, img = Cap.read()
            i_frame +=1
        Videowriter.release()
        add_audio(raw_video_path=self.video_path, tmp_video_path=tmp_save_path, save_path=save_path)

    def clear(self):
        """clear
        
        用于清空在系统运行过程中，产生的一些参数。在处理不同视频时需要运行一次
        
        Args: 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/04/19 18:12:15
        """
        self.bboxs_list = [] # 存放识别帧bboxs的列表
        self.faces_frame = [] # 存放每张图像出现的帧
        self.faces = [] # 存放识别帧人脸图像的列表， 在聚类后清空
        self.feature_matrix = np.array([]) # 和faces对应人脸的特征
        self.colors =[] # 其中存放类别对应颜色列表，如[(0,255,255)]
        self.faces_group = [] # 存放人脸对应的类 [1, 0, ...]其中下标为与self.faces对应，值为具体类
        self.proc_frames = [] # 存放处理帧序号的列表
        self.save_video_path = '' # 处理视频的路径
        self.CLEAR = True


if __name__ == "__main__":
    system = FR_system(save_dir="./datasets/active_data")
    system.face_recognize('./datas/Snaptik_7000672911061241089_tiktok.mp4', 10, 0.44)
    system.clear()
    