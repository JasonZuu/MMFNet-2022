# -*-coding:utf-8 -*-
import numpy as np
from random import shuffle
import networkx as nx
import numpy as np
import re
def face_distance(face_encodings, face_to_compare):
    """
    计算一组特征值与带比较特征值之间的距离，默认采用欧氏距离
    参数配置
    face_encodings:一组特征值，包含多个
    face_to_compare:待比较特征值，只有一个
    return:返回不同特征向量之间距离的数组矩阵
    """
    
    if len(face_encodings) == 0:
        return np.empty((0))

    dist=[]
 

    # 1-sigmoid(欧氏距离)
    # for i in range(0,len(face_encodings)):
    #     # sim = 1.0/(1.0+np.linalg.norm(face_encodings[i]-face_to_compare))
    #     tmp = np.linalg.norm(face_encodings[i]-face_to_compare)
    #     sim = 1 - (1.0 / (1.0 + np.exp(-tmp)))
    #     dist.append(sim)

    # 余弦相似度
    for i in range(0, len(face_encodings)):
        num=np.dot(face_encodings[i],face_to_compare)
        cos=num/(np.linalg.norm(face_encodings[i])*np.linalg.norm(face_to_compare))
        sim=0.5+0.5*cos # 归一化
        dist.append(sim)
    return dist
 
def find_all_index(arr,item):
    '''获取list中相同元素的索引
    输入：
        arr：待求取list
        item：待获取元素
    输出：
        相同元素索引，格式为list'''
    return [i for i, a in enumerate(arr) if a==item]
 
def chinese_whispers(feature_matrix, threshold=0.75, iterations=10, debug_mode=False):
    """chinese_whispers

    人脸聚类算法概要
        1.初始化每个节点为一个类
        2.选取任意节点开始迭代
            选择该节点邻居中边权重最大者，将两则归为一类；若邻居中有2者以上属于同一类，将这些类权重相加进行比较
    
    Args: 
       feature_matrix(np.array): 包含人脸图像提取特征的矩阵，(N, features)
       threshold(float): 是否建立边的阈值
       iterations(int): 聚类迭代次数
    
    Return: 
       group_cluster(list): 存放类别和id的列表 [[img_id]] 
    
    @Author  :   JasonZuu
    @Time    :   2021/04/11 18:22:12
    """ 
    # Create graph
    nodes = []
    edges = []
 
    #encoding_list格式为
    #[(path1,encode1),(path2,encode2),(path3,encode3)]
    #image_paths, encodings = zip(*encoding_list)
    encodings=[]
    #image_paths=[]
    for i in range(feature_matrix.shape[0]):
        encodings.append(feature_matrix[i,:])
        #image_paths.append(r'F:\outCluster\%d\\' %i)
 
    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []
 
    ''' 
    节点初始化：
        1.将每个特征向量设为一个类
        2.计算每个特征向量之间的距离，并根据门限判定是否构成邻接边
    '''
    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx
        # print(idx)
 
        # 节点属性包括
        # node_id:节点id,(0,n-1)
        # label:节点类别，初始化每个节点一个类别
        # path：节点导出路径，用于图片分类导出
        node = (node_id, {'label':idx})
        #node = (node_id, {'label': idx, 'path': image_paths[idx]})
        nodes.append(node)
 
        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break
 
        #构造比较向量组
        #若当前向量为i,则比较向量组为[i+1:n]
        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            # 若人脸特征匹配，则在这两个节点间添加关联边
            # print(distance)
            if distance >= threshold:
                #edge_id：与node_id相连接的节点的node_id
                edge_id = idx+i+1
                encoding_edges.append((node_id, edge_id, {'weight': distance}))
 
        edges = edges + encoding_edges
 
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
 
    '''
    迭代过程
    '''
    for i in range(iterations):
        cluster_nodes = list(G.nodes()) #返回节点id
        shuffle(cluster_nodes)# 随机选取一个开始节点
        for node in cluster_nodes:
            # 当前节点的所有邻接边，如节点4邻接边为(4,5,weight=8)(4,8,weight=10)
            # 则G[4]返回值为AtlasView({5:{'weight':8}, 8:{'weight':10}})
            neighbors = G[node]
            # cluster形式
            # {'cluster_path':weight}   其中cluster_path=node属性的cluster值
            labels = {}
 
            for ne in neighbors: # ne即为当前节点邻接的节点id
                if isinstance(ne, int):
                    '''
                    判断该邻居的类别是否在其他邻居中存在
                        若存在，则将相同类别的权重相加。
                    '''
                    if G.nodes[ne]['label'] in labels:#G.node[ne]['label']即为id=ne节点的label属性
                        labels[G.nodes[ne]['label']] += G[node][ne]['weight']#将这条邻接边(node,ne)的weight属性赋值给cluster[节点ne的cluster]
                    else:
                        labels[G.nodes[ne]['label']] = G[node][ne]['weight']
 
            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #将邻居节点的权重最大值对应的文件路径给到当前节点
            #这里cluster即为path
            for id in labels:
                if labels[id] > edge_weight_sum:
                    edge_weight_sum = labels[id]
                    max_cluster = id
 
            # set the class of target node to the winning local class
            #print('node %s was clustered in %s' %(node, max_cluster))
            G.nodes[node]['label'] = max_cluster
    list_label_out = []
    for i in range(len(encodings)):
        list_label_out.append(G.nodes[i]['label'])
    #print(list_label_out)
 
    ''' 
    统计分类错误数量=新类别中不属于原类别的数量      eg： list_label_out=[1,3,4,2,2,4,3,1]
    # group_all 返回最终类别标签                     group_all=[1,2,3,4]
    # group_num 最终分类数量                        group_num=4
    # group_cluster: list,返回相同标签的节点id       group_cluster=[[0,7],[3,4],[1,6],[2,5]]
    '''
    group_all = set(list_label_out)
    group_num = len(group_all)
    group_cluster = []
 
    for item in group_all:
        group_cluster.append(find_all_index(list_label_out,item))
 
    if debug_mode:
        print('最终分类数量：%s' %group_num)
        for i in range(0,group_num):
            print('第%d类：%s'%(i,group_cluster[i]))

    return group_cluster
 
def cw_cluster(features, threshold=0.75, iterations=10, debug_mode=False):
    group_cluster = chinese_whispers(features,threshold, iterations, debug_mode)
    faces_group =  list(range(features.shape[0]))
    for i_group in range(len(group_cluster)):
        for term in group_cluster[i_group]:
            faces_group[term] = i_group
    return faces_group

if __name__ == '__main__':
    pass