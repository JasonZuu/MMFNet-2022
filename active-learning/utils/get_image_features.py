import cv2

def img_resize(img, target_size):
   """target_size:[height, length, layers]"""
   if img.size < target_size[0]*target_size[1]*target_size[2]:# 需要放大的情况
      img = cv2.resize(img, (target_size[0], target_size[1]), cv2.INTER_CUBIC)
   else: # 缩小的情况
      img = cv2.resize(img, (target_size[0], target_size[1]), cv2.INTER_AREA)
   return img


def get_image_size(img_path):
   """get_image_var_size
   
   获取图片的模糊度度量和图片大小，并返回
   
   Args: 
      img_path(str):图片存储位置 
   
   Return: 
      img_acu(float): 图片模糊度度量（具体为拉普拉斯算子处理后的方差）
      img_size(int): 图片大小
   
   @Author  :   JasonZuu
   @Time    :   2021/05/08 18:13:37
   """
   image = cv2.imread(img_path, 0)
   # 处理大小
   img_size = image.shape[0] * image.shape[1]
   
   return img_size

def get_image_acu(img_path):
   image = cv2.imread(img_path, 0)
   image = img_resize(image, [160, 160, 1])
   # 处理图片锐度
   x = cv2.Sobel(image,cv2.CV_16S,1,0)
   y = cv2.Sobel(image,cv2.CV_16S,0,1)

   absX = cv2.convertScaleAbs(x)   # 转回uint8
   absY = cv2.convertScaleAbs(y)
   img_acu = cv2.addWeighted(absX,0.5,absY,0.5,0).var()
   
   return img_acu


if __name__ == "__main__":
    size = get_image_size("./0.jpg")
    acu = get_image_acu("./0.jpg")
    print(acu)
    print(size)