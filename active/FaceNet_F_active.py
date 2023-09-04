from models.inception_resnet_v1 import InceptionResnetV1
from models.facenet_f import FaceNet_F
from models.modules import Adaptive_module
from models.Fringe_loss import Fringe_Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.DL_load_data import X_Dataset, Xy_Dataset
from utils.DL_base_ProcSystem import Base_ProcSystem
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split


class FaceNet_F_active_PS(Base_ProcSystem):
    def __init__(self,
                name, 
                eps=0.2,
                device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu"), 
                init_lr=1e-5, 
                epochs=5,
                batch_size=32):
        super().__init__(self)
        self.name = name
        self.count_no_best = 0
        self.best_f1 = 0
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.extractor = InceptionResnetV1(pretrained='casia-webface',classify=True, num_classes=512)
        self.extractor.to(device)
        self.frame = FaceNet_F(out_planes=512)
        self.frame.to(device)
        self.adaptive_module = Adaptive_module(in_planes=512)
        self.adaptive_module.to(device)
        self.optimizer = optim.Adam([
            {'params': self.extractor.parameters(), "lr":init_lr*0.01},
            {'params': self.frame.parameters(), "lr":init_lr*0.01},
            {'params': self.adaptive_module.parameters()}],
            lr=init_lr)
        self.criterion = Fringe_Loss(eps=eps, device=self.device)
        

    def train(self, X, y, save_path, debug_interval=100):
        X_train, X_very, y_train, y_very = train_test_split(X, y, random_state=27, test_size=0.2)
        Train_dataset = Xy_Dataset(X_train, y_train)
        Train_loader = DataLoader(dataset=Train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True) 
        Very_dataset = Xy_Dataset(X_very, y_very)
        Very_loader = DataLoader(dataset=Very_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True) 
        for epoch in range(self.epochs):
            self.extractor.train()
            self.frame.train()
            self.adaptive_module.train()
            for i, (imgs, X_struc, labels) in enumerate(Train_loader):
                labels = labels.to(self.device)
                X0 = self.extractor(imgs.to(self.device))
                features = self.frame(X0, X_struc.to(self.device))
                outputs = self.adaptive_module(features)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i%debug_interval) == 0:
                    pred = outputs.data.max(1, keepdim=True)[1]
                    target = labels.data.max(1, keepdim=True)[1]
                    correct = pred.eq(target.data.view_as(pred)).sum()
                    accuracy = correct/ self.batch_size
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t accuracy:{:.2f}'.format(
                    epoch, i * len(imgs), len(Train_loader.dataset),
                    100. * i / len(Train_loader), loss.item(),
                    accuracy))
            STOP = self.verify(Very_loader, save_path, epoch)
            if STOP:
                return 0
        
    @torch.no_grad()
    def verify(self, Very_loader, save_path, epoch):
        save_name, exp_name = os.path.splitext(save_path)
        save_dir, _ = os.path.split(save_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.extractor.eval()
        self.frame.eval()
        self.adaptive_module.eval()
        TP, TN, FP, FN = 0, 0, 0, 0
        for i, (imgs, X_struc, labels) in enumerate(Very_loader):
            labels = labels.to(self.device)
            X0 = self.extractor(imgs.to(self.device))
            features = self.frame(X0, X_struc.to(self.device))
            outputs = self.adaptive_module(features)
            pred = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
            target = labels.data.max(1, keepdim=True)[1].cpu().numpy()
            for i in range(len(target)):
                if pred[i] == 1 and target[i]==1:
                    TP += 1
                elif pred[i] == 0 and target[i]==0:
                    TN += 1
                elif pred[i] == 1 and target[i]==0:
                    FP += 1
                elif pred[i] == 0 and target[i]==1:
                    FN += 1
        accu = (TP+TN)/(TP+TN+FP+FN)
        if TP+FP == 0:
            prec = 0
        else:
            prec = TP/(TP+FP)
        if TP+FN == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)
        if prec+recall == 0:
            f1 = 0
        else:
            f1 = 2*prec*recall/(prec+recall)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.save_params(f"{save_name}{exp_name}") 
            self.count_no_best = 0
        else:
            self.count_no_best += 1
        print("-------------------------------------VERIFY----------------------------------------------------")
        print(f"current epoch:{epoch+1}\tleft epoch:{self.epochs-1-epoch}\tno_best_account:{self.count_no_best}")
        print(f"Accuracy: {accu:3.2%}\tPrecision: {prec:3.2%}\tRecall: {recall:3.2%}\tF1 score: {f1:3.2%}")
        print(f"best f1:{self.best_f1}")
        print("-----------------------------------END VERIFY----------------------------------------------------")
        if self.count_no_best > 4:
            self.count_no_best = 0
            self.best_f1 = 0
            return True
        else:
            self.count_no_best = 0
            self.best_f1 = 0
            return False

    @torch.no_grad()
    def test(self, X_test, y_test, i_epoch):
        self.extractor.eval()
        self.frame.eval()
        self.adaptive_module.eval()
        Test_dataset = Xy_Dataset(X_test, y_test)
        Test_loader = DataLoader(dataset=Test_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False)
        TP, TN, FP, FN = 0, 0, 0, 0
        scores = None
        for i, (imgs, X_struc, labels) in enumerate(Test_loader):
            labels = labels.to(self.device)
            X0 = self.extractor(imgs.to(self.device))
            features = self.frame(X0, X_struc.to(self.device))
            outputs = self.adaptive_module(features)
            score = outputs.detach().cpu().numpy()[:,1]
            score.reshape(-1)
            if scores is None:
                scores = score
            else:
                scores = np.concatenate((scores, score), axis=0)
            pred = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
            target = labels.data.max(1, keepdim=True)[1].cpu().numpy()
            for i in range(len(target)):
                if pred[i] == 1 and target[i]==1:
                    TP += 1
                elif pred[i] == 0 and target[i]==0:
                    TN += 1
                elif pred[i] == 1 and target[i]==0:
                    FP += 1
                elif pred[i] == 0 and target[i]==1:
                    FN += 1
        accu = (TP+TN)/(TP+TN+FP+FN)
        if TP+FP == 0:
            prec = 0
        else:
            prec = TP/(TP+FP)
        if TP+FN == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)
        if prec+recall == 0:
            f1 = 0
        else:
            f1 = 2*prec*recall/(prec+recall)
        fpr, tpr, thres = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        print("-----------------------------------TEST----------------------------------------------------")
        print(f"Accuracy: {accu:3.2%}\tPrecision: {prec:3.2%}\tRecall: {recall:3.2%}\tF1 score: {f1:3.2%}\tAUC:{roc_auc:3.2%}\n") 
        column_datas = {"Accuracy":accu,"Precision":prec,"Recall":recall,"F1 score":f1, "AUC":roc_auc}
        df = pd.DataFrame(column_datas, index=[0])
        df.to_csv(f"./results/column/{i_epoch}/{self.name}.csv", sep=",", index=None)

              
        # 存储运行数据，用于绘制ROC曲线
        roc_datas = {"FPR":fpr,"TPR":tpr,"THRESHOLD":thres}
        df = pd.DataFrame(roc_datas, index=None)
        df.to_csv(f"./results/roc/{i_epoch}/{self.name}.csv", sep=",", index=None)
        return f1

    @torch.no_grad()
    def output(self, X):
        self.extractor.eval()
        self.frame.eval()
        self.adaptive_module.eval()
        Output_dataset = X_Dataset(X)
        Output_loader = DataLoader(dataset=Output_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False)
        for i, (imgs, X_struc) in enumerate(Output_loader):
            X0 = self.extractor(imgs.to(self.device))
            features = self.frame(X0, X_struc.to(self.device))
            outputs = self.adaptive_module(features)
            pred = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
            if i == 0:
                preds = pred
            else:
                preds = np.concatenate((preds, pred))
        
        preds = preds.reshape(-1)
        return preds


if __name__ == "__main__":
    ProcSystem = MMFNet_active_PS(name="MMFNet")
    i_epoch = 8
    params_path = f"./model_params/{i_epoch}/{ProcSystem.name}_1.params"
    datas = pd.read_csv("./dataset/tiktok/tiktok_Features.csv", sep=',')
    X_tiktok = datas[['img_path', 'frequency', 'size', 'relative_size', 'acu', 'relative_acu']]
    y_tiktok = datas.labels
    # ProcSystem.train(X_train, y_train, X_very, y_very, params_path)
    ProcSystem.load_params(params_path)
    ProcSystem.test(X_tiktok, y_tiktok, i_epoch)