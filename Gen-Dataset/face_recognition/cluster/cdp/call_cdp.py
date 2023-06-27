import os
import numpy as np
from cluster.cdp.single_api import CDP as S_cdp
from cluster.cdp.multi_api import CDP as M_cdp

class S_cdp_cluster():
   def __init__(self, K=20, threshold=0.35, metric='angulardist'):
      """S_cdp_cluster.__init__()
      
      Args: 
      
      Return: 
      
      @Author  :   JasonZuu
      @Time    :   2021/04/22 19:00:05
      """
      self.cdp = S_cdp(K, threshold, metric)
      
   def cluster(self, X):
    """cdp_single_cluster
    
    调用singel_cdp模型，来进行聚类
    
    Args: 
       X(np.array): 提取好的人脸特征，shape=(N_imgs, N_features)
    
    Return: 
       pred(list): 人脸聚类结果，其中每个元素下标对应faces，值为faces类别  
    
    @Author  :   JasonZuu
    @Time    :   2021/04/22 11:04:43
    """
    self.cdp.fit(X)
    return list(self.cdp.labels_)


class M_cdp_cluster():
   def __init__(self, K=20, threshold=0.35, metric='angulardist'):
      """S_cdp_cluster.__init__()
      
      Args: 
      
      Return: 
      
      @Author  :   JasonZuu
      @Time    :   2021/04/22 19:00:05
      """
      self.cdp = M_cdp(K, threshold, metric)
      
   def cluster(self, X):
    """cdp_single_cluster
    
    调用singel_cdp模型，来进行聚类
    
    Args: 
       X(np.array): 提取好的人脸特征，shape=(N_imgs, N_features)
    
    Return: 
       pred(list): 人脸聚类结果，其中每个元素下标对应faces，值为faces类别  
    
    @Author  :   JasonZuu
    @Time    :   2021/04/22 11:04:43
    """
   #  print(X.shape)
    self.cdp.fit([X])
    return list(self.cdp.labels_)


if __name__ == "__main__":
    X = np.rand(100, 256)
    cdp_cluster = M_cdp_cluster()
    pred = cdp_cluster.cluster(X)
    print(type(pred))
    print(pred)