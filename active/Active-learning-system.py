import torch
from torch._C import device
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
from ResNet_F_first import ResNet_F_first_PS
from ResNet_F_active import ResNet_F_active_PS
# from ArcFace_F_first import ArcFace_F_first_PS
# from ArcFace_F_active import ArcFace_F_active_PS
# from MMFNet_first import MMFNet_first_PS
# from MMFNet_active import MMFNet_active_PS
# from FaceNet_F_first import FaceNet_F_first_PS
# from FaceNet_F_active import FaceNet_F_active_PS
# from VGGFace_F_first import VGGFace_F_first_PS
# from VGGFace_F_active import VGGFace_F_active_PS
from copy import deepcopy

class Active_learning_system():
    """
    基于tri-learning的思路，设计的半监督自学习系统
    包括进行原始3模型训练（3折训练法）和后续自主epochs轮训练
    还包括逐论测试
    """
    def __init__(self, 
                raw_train_data_path, 
                active_train_data_path, 
                test_data_path, 
                model_param_dir="./model_params", 
                train_times_per_epoch=10,
                model_name="MMFNet",
                device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu")):
        self.raw_train_data_path = raw_train_data_path
        self.active_train_data_path = active_train_data_path
        self.test_data_path = test_data_path
        self.model_param_dir = model_param_dir
        self.train_times_per_epoch = train_times_per_epoch
        self.model_name = model_name
        self.device = device
        if not os.path.exists(self.model_param_dir): # 生成模型参数文件夹
            cwd = os.getcwd()
            os.mkdir(cwd+'/'+self.model_param_dir)
        test_datas = pd.read_csv(self.test_data_path, sep=',')
        active_data = pd.read_csv(self.active_train_data_path, sep=',')
        self.X_active = active_data[['img_path', 'frequency', 'size', 'relative_size', 'acu', 'relative_acu']]
        self.X_test = test_datas[['img_path', 'frequency', 'size', 'relative_size', 'acu', 'relative_acu']]
        self.y_test = test_datas.labels

        self.best_f1s = [0.7490, 0.7528, 0.7164] # first_result_f1s, 不要求从大到小排序
        self.best_model_params_path = [f"./model_params/0/{self.model_name}_0.params", f"./model_params/0/{self.model_name}_1.params", f"./model_params/0/{self.model_name}_2.params"] # related model params paths
    
    def first_train(self):
        df = pd.read_csv(self.raw_train_data_path, sep=',', index_col=None)
        X = df[['img_path', 'frequency', 'size', 'relative_size', 'acu', 'relative_acu']]
        y = df.labels
        kf = KFold(3)
        i_model = 0
        params_dir =  f"./model_params/0/"
        if not os.path.exists(params_dir):
            cwd = os.getcwd()
            os.mkdir(cwd+'/'+params_dir)
        for train_index, test_index in kf.split(X):
            X_used = X.iloc[train_index]
            y_used = y.iloc[train_index]
            ProcSystem = ResNet_F_first_PS(name=f"{self.model_name}_{i_model}", device=self.device)
            params_path = f"{params_dir}{ProcSystem.name}.params"
            ProcSystem.train(X_used, y_used, params_path)
            ProcSystem.load_params(params_path)
            ProcSystem.test(self.X_test, self.y_test)
            i_model += 1

    def active_learn(self, n_epoch=3):
        """
        Actively learn from datas for specified epochs

        n_epoch(int): train epochs
        """
        assert n_epoch>0

        for i_epoch in range(1, n_epoch+1):
            last_best_f1 = deepcopy(self.best_f1s) # 收敛检验机制
            self.self_train(i_epoch)
            if last_best_f1 == self.best_f1s:
                print(f"Converge at {i_epoch} epoch")
                break
        print("================================BEST RESULT=================================")
        print(f"Best F1: {max(self.best_f1s)}")
        print(f"Best model params path: {self.best_model_params_path[self.best_f1s.index(max(self.best_f1s))]}")

    
    def self_train(self, i_epoch):
        """
        finish the training task of specified epoch

        i_epoch(int): the specified epoch
        """
        assert i_epoch>0
        result_dir_column = f"./results/column/{i_epoch}/"
        if not os.path.exists(result_dir_column):
            cwd = os.getcwd()
            os.mkdir(os.path.join(cwd, result_dir_column))
        result_dir_roc = f"./results/roc/{i_epoch}/"
        if not os.path.exists(result_dir_roc):
            cwd = os.getcwd()
            os.mkdir(os.path.join(cwd, result_dir_roc))
        params_dir =  f"./model_params/{i_epoch}/"
        if not os.path.exists(params_dir):
            cwd = os.getcwd()
            os.mkdir(os.path.join(cwd, params_dir))

        model_indexs = [0, 1, 2]
        for model_index in model_indexs:
            ProcSystem = ResNet_F_active_PS(name=f"{self.model_name}_{model_index}", device=self.device)
            last_params_path = self.best_model_params_path[model_index]
            ProcSystem.load_params(last_params_path)
            params_path = f"{params_dir}{ProcSystem.name}.params"
            data_index, y_active = self._get_active_labels(model_index, i_epoch)
            X_active = self.X_active.iloc[data_index]
            ProcSystem.train(X_active, y_active, params_path)
            ProcSystem.load_params(params_path)
            f1 = ProcSystem.test(self.X_test, self.y_test, i_epoch)
            if f1 > min(self.best_f1s):
                min_best_index = self.best_f1s.index(min(self.best_f1s))
                self.best_f1s[min_best_index] = f1
                self.best_model_params_path[min_best_index] = params_path

    def _get_active_labels(self, for_i_model, i_epoch):
        """
        Inner_function
        Getting semi-training labels for certain model at certain epoch.
        """
        assert i_epoch > 0
        model_params_dir = f"./model_params/{i_epoch-1}/"
        model_indexs = [0, 1, 2]
        model_indexs.remove(for_i_model)
        preds = []
        for model_index in model_indexs:
            model_params_path = f"{model_params_dir}{self.model_name}_{model_index}.params"
            ProcSystem = ResNet_F_active_PS(name=self.model_name, device=self.device)
            ProcSystem.load_params(model_params_path)
            pred = ProcSystem.output(self.X_active)
            preds.append(pred)
        # 对比2个pred，生成data_index，其中存放需要使用的数据的index
        # 同时，保留相同的预测值，作为preds
        assert len(preds) == 2
        preds = np.array(preds)
        data_index = []
        output = []
        for i_labels in range(preds.shape[1]):
            if preds[0][i_labels]==preds[1][i_labels]: # 说明2个模型的预测结果相同
                data_index.append(i_labels)
                output.append(preds[0][i_labels])
        
        assert len(data_index) == len(output)
        return data_index, pd.DataFrame({"labels":output}, index=None)

if __name__ == "__main__":
    ALS = Active_learning_system(raw_train_data_path="./dataset/IF-Dataset/IF_Features.csv",
                                active_train_data_path="./dataset/active_data/Features.csv",
                                test_data_path="./dataset/tiktok/tiktok_Features.csv",
                                model_param_dir="./model_params",
                                train_times_per_epoch=5,
                                model_name="ResNet_F",
                                device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu"))
    # ALS.first_train()
    ALS.active_learn(n_epoch=20)