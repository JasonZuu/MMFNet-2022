from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import insightface
from insightface.utils import face_align


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, prefix, epoch, layer):
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, ctx_id, model_prefix, model_epoch, image_size=(112,112), use_large_detector=False):
        if use_large_detector:
            self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        else:
            self.detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
        self.detector.prepare(ctx_id=ctx_id)
        if ctx_id>=0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        self.model = get_model(ctx, image_size, model_prefix, model_epoch, 'fc1')
        self.image_size = image_size

    def get_input(self, face_img):
        if face_img.size < 112*112*3:# 需要放大的情况
            img = cv2.resize(face_img, (112, 112), cv2.INTER_CUBIC)
        else: # 缩小的情况
            img = cv2.resize(face_img, (112, 112), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# 色彩空间转换
        img = np.transpose(img, (2, 0, 1))
        return img

    def get_feature(self, input_blob):
        # input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        norm = np.sqrt(np.sum(emb*emb)+0.00001)
        emb /= norm
        return emb

def get_insightface(gpu_id=-1, img_size=(112,112), model_prefix="insightface_slim/model-r100-ii/model", model_epoch=0):
    """get_insightface
    
    调用mxnet的model，参数都是与mxnet加载模型相关的，建议除了gpu_id外不做更改
    
    Args: 
       img_size(array): 输入模型的img大小，模型会根据这个值进行图片分辨率重构
       gpu_id(int): 使用gpu的id，-1表示不用，单个gpu时，选择使用则置为0
       model_prefix(str): 模型位置，最后名称为model即可
       model_epoch(int): 模型运行轮数
    
    Return: 
       model(mxnet.model): 加载了参数的模型 
    
    @Author  :   JasonZuu
    @Time    :   2021/04/13 17:13:34
    """
    model_prefix = os.getcwd() + '/' + model_prefix # 获取当前工作路径下的文件坐标
    model = FaceModel(gpu_id, model_prefix, model_epoch)
    return model

