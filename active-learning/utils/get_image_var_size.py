import cv2
import numpy as np

def get_image_var_size(img_path):
    """get_image_var_size
    
    获取图片的模糊度度量和图片大小，并返回
    
    Args: 
       img_path(str):图片存储位置 
    
    Return: 
       img_var(float): 图片模糊度度量（具体为拉普拉斯算子处理后的方差）
       img_size(int): 图片大小
    
    @Author  :   JasonZuu
    @Time    :   2021/05/08 18:13:37
    """
    image = cv2.imread(img_path, 0)
    # 处理大小
    img_size = image.shape[0] * image.shape[1]
    # 处理图片模糊程度
    img_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return img_var, img_size

if __name__ == "__main__":
    var, size = get_image_var_size("./show1.jpg")
    print(var)
    print(size)