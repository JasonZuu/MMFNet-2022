from moviepy.editor import *

def add_audio(raw_video_path, tmp_video_path, save_path):
    """add_audio
    
    给生成视频添加音频
    
    Args: 
       raw_video_path(str): 原始视频文件
       tmp_video_path(str): 生成的无声音音频
       save_path(str): 最终视频保存路径
    
    Return: 
        
    @Author  :   JasonZuu
    @Time    :   2021/07/29 12:54:31
    """
    raw_video = VideoFileClip(raw_video_path)
    audio = raw_video.audio
    tmp_video = VideoFileClip(tmp_video_path)
    final_video = tmp_video.set_audio(audio)
    final_video.write_videofile(save_path)