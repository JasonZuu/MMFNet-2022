import os
import re
from copy import deepcopy
import pandas as pd
from utils.get_image_features import get_image_size, get_image_acu
import numpy as np
import cv2


class EF_system():
    def __init__(self):
        self.IS_CLEAR = False
        self.clear()

    def prepare(self, video_dir):
        assert self.IS_CLEAR == True
        self._set_video_dir(video_dir)
        dirs = os.listdir(video_dir)
        # 输出所有文件和文件夹
        for file in dirs:
            if file[-4:] == ".mp4":
                raw_video_path=f"{video_dir}/{file}"
        self._get_raw_video_info(raw_video_path)
        self._feature_extract()
        self._get_labels()
        self.IS_CLEAR = False
    
    def output(self):
        assert len(self.img_labels)==len(self.img_paths)
        output_dict = deepcopy(self.value_dict)
        output_dict["labels"] = self.img_labels
        output_df = pd.DataFrame(output_dict, index=None)
        return output_df


    def clear(self):
        if not self.IS_CLEAR:
            self.img_paths = [] # 每张图片路径
            self.img_groups = [] # 每张图片所属的人组
            self.img_labels = [] # 每张照片对应的标签，0为有关人，1为无关人
            self.img_group_freq = [] # 存放每个图片所属类在其所属视频中出现的频率
            self.img_acus = [] # 每张图像的锐度，越大越清晰，越小越模糊
            self.img_acus_norm = [] # 每张图像的锐度正则化
            self.img_sizes = [] # 每张照片的尺寸
            self.img_sizes_norm = [] # 每个视频中出现的人脸图像归一化，不是全部的图像归一化
            self.X_direction = [] # 每张图片面部朝向的X方向向量
            self.video_img_maxsize = []  # 存放每个视频中图片尺寸最大值 和self.video_dirs对应
            self.video_img_minsize = [] # 存放每个视频中图片尺寸最小值 和self.video_dirs对应
            self.video_img_maxacus = [] # 图像最大锐度
            self.video_img_minacus = [] # 图像最小锐度
            self.fps = 0 # 视频的fps
            self.width = 0 # 视频输出帧的宽度
            self.height = 0 # 视频输出帧的长度
            self.frame_size = 0 # 视频输出帧的尺寸
            self.N_frames = 0 # 视频帧总数
            self.N_groups = [] # 每个组所拥有的照片个数
            self.irre_N_groups = [] # 每个组所拥有的无关人照片个数
            self.first_time_groups = [] # 每个组第一张照片出现的时间
            self.last_time_groups = [] # 每组最后一张照片出现时间
            self.rele_groups = [] # 有关人对应的组
            self.value_dict = {} # 存放特征的dict
            self.IS_CLEAR = True # 用于标志模型处理视频时是否经过清理
        

    def _set_video_dir(self, video_dir):
        """set_video_dir
        
        可以重新设定需要处理的视频文件夹,同时会调用_clear函数进行模型清空
        
        Args: 
           arg1(int): 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/05/14 20:55:28
        """
        self.video_dir = video_dir
        
    def _get_raw_video_info(self, raw_video_path):
        """get_raw_video_info
        
        获取原始视频信息，如fps、frame_size等
        
        Args: 
           raw_video_path(str): 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/05/14 21:49:08
        """
        Cap = cv2.VideoCapture(raw_video_path)
        self.fps = Cap.get(cv2.CAP_PROP_FPS)
        self.width = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = self.width*self.height
        self.N_frames = Cap.get(cv2.CAP_PROP_FRAME_COUNT)
  

    def _get_labels(self):
        """_get_labels
        
        读取文件夹获取labels
        
        Args:         
        Return: 

        @Author  :   JasonZuu
        @Time    :   2021/05/14 21:29:37
        """
        dirs = os.listdir(self.video_dir)
        for file in dirs:
            if file[-4:] == ".txt":
                logs_path = f"{self.video_dir}/{file}"
        with open(logs_path) as f:
            lines = f.readlines()
        
        # 获取全部的有关人组
        for i_line in range(1, len(lines)-1):
            line = lines[i_line]
            rele_group = int(line[:-1])
            self.rele_groups.append(deepcopy(rele_group))
        # 根据有关人组信息，给予每个图片标签
        for img_group in self.img_groups:
            if img_group in self.rele_groups:
                self.img_labels.append(0)
            else:
                self.img_labels.append(1)            
       

    def _feature_extract(self):
        """Video_proc_system._feature_extract()
        
        调用了特征提取的全部模块,以获取进行分类所需要的全部信息
        
        Args: 
         
        
        @Author  :   JasonZuu
        @Time    :   2021/05/14 21:06:12
        """
        self._get_paths_groups_frames()
        self._extract_features()
        self.value_dict = { 'img_path':self.img_paths,
                            'frequency':self.img_group_freq,
                            'size':self.img_sizes,
                            'relative_size':self.img_sizes_norm,
                            'acu':self.img_acus,
                            'relative_acu':self.img_acus_norm}
        


    def _get_paths_groups_frames(self): 
        """_get_paths_groups_frames
        
        获取人脸识别后人脸图片路径,所属组,所在帧数的函数
        
        Args: 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/05/14 20:47:58
        """ 
        FIRST = True 
        for root, dirs, files in os.walk(self.video_dir):
            if FIRST:
                group_dir = deepcopy(dirs)
                for i_dir in range(len(group_dir)):
                    group_dir[i_dir] = os.path.join(root, dirs[i_dir])
                FIRST = False
            else:
                for file in files:
                    # 处理文件路径
                    file_path = os.path.join(root, file)
                    self.img_paths.append(file_path)
                    # 处理groups
                    pattern0 = r"/\d{1,3}"
                    ret = re.search(pattern0, root)
                    group = int(os.path.basename(ret.group()))
                    self.img_groups.append(group)


    def _proc_group_freq(self, video_dir):
        """self._proc_group_freq
        
        处理人脸出现频率，获取该目录下全部视频，随后返回每类中人脸占比
        
        Args: 
            video_dir(str): 处理视频文件夹
        
        Return: 
           freq_list(list): 该目录中，每个视频所属类的出现频率 
        
        @Author  :   JasonZuu
        @Time    :   2021/04/26 11:34:37
        """
        FIRST = True
        freq_list = []
        N_imgs_each_group = []
        N_sum = 0
        for root, dirs, files in os.walk(video_dir):
            if FIRST: # 第一层组遍历不统计信息
                # all_groups = dirs
                FIRST = False
            else: #其余层统计人脸个数
                N_imgs = len(files)
                N_sum += N_imgs
                N_imgs_each_group.append(deepcopy(N_imgs))
        freq_each_group = [N_imgs/N_sum for N_imgs in N_imgs_each_group]
        for i_group in range(len(freq_each_group)):
            freq_group = [freq_each_group[i_group] for i in range(N_imgs_each_group[i_group])]
            freq_list += freq_group
        return freq_list

    def _get_max_min_features(self):
        """_get_max_min_sizes_fuzzy
        
        获取每个视频中图片尺寸的最大值和最小值
        用于归一化图片尺寸
        获取每个视频中图片锐度的最大值和最小值
        用于归一化图片锐度
        
        Args: 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/05/03 16:31:34
        """
        assert self.img_sizes != [] #要求已经提取了图像大小
        assert self.img_acus != [] #要求已经提取了图像锐度
        FIRST = True
        for i_img in range(len(self.img_sizes)):
            img_size = self.img_sizes[i_img]
            img_acu = self.img_acus[i_img]
            if FIRST:
                max_size = img_size 
                min_size = img_size 
                max_acu = img_acu
                min_acu = img_acu
                FIRST = False
            else:
                max_size = max(max_size, img_size)
                min_size = min(min_size, img_size)
                max_acu = max(max_acu, img_acu)
                min_acu = min(min_acu, img_acu)
        self.video_img_maxsize.append(deepcopy(max_size))
        self.video_img_minsize.append(deepcopy(min_size))
        self.video_img_maxacus.append(deepcopy(max_acu))
        self.video_img_minacus.append(deepcopy(min_acu))

    def _norm_img_features(self):
        """_norm_img_sizes_fuzzy
        
        用于正则化图片特征
        
        Args: 
        
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/05/03 16:31:34
        """
        assert self.video_img_maxacus != [] # 要求提取了图像模糊度最大最小值
        assert self.video_img_maxsize != [] # 要求提取了图像尺寸最大最小值
        i_video = 0
        max_size = self.video_img_maxsize[i_video]
        min_size = self.video_img_minsize[i_video]
        max_acu = self.video_img_maxacus[i_video]
        min_acu = self.video_img_minacus[i_video]
        for i_img in range(len(self.img_sizes)):
            img_size = self.img_sizes[i_img]
            img_acu = self.img_acus[i_img]
            norm_img_size = (img_size-min_size)/(max_size-min_size)
            norm_img_acu = (img_acu-min_acu)/(max_acu-min_acu)
            self.img_sizes_norm.append(deepcopy(norm_img_size))
            self.img_acus_norm.append(deepcopy(norm_img_acu))


    def _extract_features(self):
        """_extract_structural_features

        用于从原始图像中提取结构化数据
        视频中人脸出现频率 用该人人脸数/人脸总数,
        帧中人脸图像面积占比 ,
        人脸图像锐度,
        面部X方向
        
        Args: 
        Return: 
        
        @Author  :   JasonZuu
        @Time    :   2021/04/26 10:27:59
        """
        # 处理人脸频率 self.img_group_freq
        freq_list = self._proc_group_freq(self.video_dir)
        self.img_group_freq += freq_list
        # 处理每个图像的人脸锐度、尺寸
        for img_path in self.img_paths:
            img_size = get_image_size(img_path)
            img_acu = get_image_acu(img_path)
            self.img_acus.append(deepcopy(img_acu))
            self.img_sizes.append(deepcopy(img_size))
        # 处理正则化图像大小和模糊程度
        self._get_max_min_features()
        self._norm_img_features()

def batch_process(VID, dataset_dir, save_path):
    FIRST_LAYER = True
    FIRST_DF = True
    for root, dirs, _ in os.walk(dataset_dir, topdown=True):
        if FIRST_LAYER:
            FIRST_LAYER = False
            for dir_name in dirs:
                video_dir = os.path.join(root, dir_name)
                VID.prepare(video_dir)
                df = VID.output()
                VID.clear()
                if FIRST_DF:
                    FIRST_DF = False
                    final_df = deepcopy(df) # 最终输出的DataFrame
                else:
                    final_df = pd.concat([final_df,deepcopy(df)], axis=0)              
        else:
            break
    final_df.to_csv(save_path, index=0, sep=',')



if __name__ == "__main__":
    VID = EF_system()
    batch_process(VID, "./dataset/IF-Dataset", "./dataset/IF_Features.csv")