from models.inception_resnet import InceptionResnetV1
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils.DL_load_data import FaceNet_clf_dataset, dl_load_data
from utils.DL_base_ProcSystem import NO_Fusion_ProcSystem
import os
import numpy as np
from models.Fringe_loss import Fringe_Loss
from models.baseline_fusion import clf_metric
from sklearn.metrics import roc_curve, auc
import pandas as pd


BATCH_SIZE = 32
EPOCHS = 10
DEBUG_INTERVAL = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


class Inception_ProcSystem(NO_Fusion_ProcSystem):
    def __init__(self, name, device=DEVICE, eps=0.2, init_lr=1e-4, decay_steps=5, decay_gamma=0.1):
        super().__init__()
        self.name = name
        self.count_no_best = 0
        self.best_accu = 0
        self.device = device
        self.extractor = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=128)
        self.extractor.to(device)
        self.clf = clf_metric(in_planes=128)
        self.clf.to(device)
        self.optimizer = optim.Adam([
            {'params': self.extractor.parameters(), "lr":init_lr*0.01},
            {'params': self.clf.parameters()}],
            lr=init_lr)
        self.scheduler = StepLR(self.optimizer, step_size=decay_steps, gamma=decay_gamma)
        self.criterion = Fringe_Loss(eps=eps, device=self.device)
        
    def train(self, X_train, y_train, X_very, y_very, save_path, epochs=EPOCHS, debug_interval=DEBUG_INTERVAL, batch_size=BATCH_SIZE):
        
        Train_dataset = FaceNet_clf_dataset(X_train, y_train)
        Train_loader = DataLoader(dataset=Train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True) 
        Very_dataset = FaceNet_clf_dataset(X_very, y_very)
        Very_loader = DataLoader(dataset=Very_dataset,
                                    batch_size=batch_size,
                                    shuffle=True) 
        for epoch in range(epochs):
            self.scheduler.step()
            self.extractor.train()
            self.clf.train()
            for i, (imgs, X_struc, labels) in enumerate(Train_loader):
                labels = labels.to(self.device)
                X0 = self.extractor(imgs.to(self.device))
                outputs = self.clf(X0)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i%debug_interval) == 0:
                    pred = outputs.data.max(1, keepdim=True)[1]
                    target = labels.data.max(1, keepdim=True)[1]
                    correct = pred.eq(target.data.view_as(pred)).sum()
                    accuracy = correct/ batch_size
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
        self.clf.eval()
        TP, TN, FP, FN = 0, 0, 0, 0
        for i, (imgs, X_struc, labels) in enumerate(Very_loader):
            labels = labels.to(self.device)
            X0 = self.extractor(imgs.to(self.device))
            outputs = self.clf(X0)
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
        if accu > self.best_accu:
            self.best_accu = accu
            self.save_params(f"{save_name}{exp_name}") 
            self.count_no_best = 0
        else:
            self.count_no_best += 1
        print("-------------------------------------VERIFY----------------------------------------------------")
        print(f"current epoch:{epoch+1}\tleft epoch:{EPOCHS-1-epoch}\tno_best_account:{self.count_no_best}\tlearning_rate:{self.scheduler.get_last_lr()}")
        print(f"Accuracy: {accu:3.2%}\tPrecision: {prec:3.2%}\tRecall: {recall:3.2%}\tF1 score: {f1:3.2%}")
        print(f"best accuracy:{self.best_accu}")
        print("-----------------------------------END VERIFY----------------------------------------------------")
        if self.count_no_best > 4:
            self.count_no_best = 0
            return True
        else:
            return False

    @torch.no_grad()
    def test(self, X_test, y_test, random_seed, batch_size=BATCH_SIZE, tiktok=False):
        self.extractor.eval()

        self.clf.eval()
        Test_dataset = FaceNet_clf_dataset(X_test, y_test)
        Test_loader = DataLoader(dataset=Test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
        TP, TN, FP, FN = 0, 0, 0, 0
        scores = None
        for i, (imgs, X_struc, labels) in enumerate(Test_loader):
            labels = labels.to(self.device)
            X0 = self.extractor(imgs.to(self.device))
            outputs = self.clf(X0)
            score = outputs.detach().cpu().numpy()[:,1]
            # score = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
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
        print(f"Accuracy: {accu:3.2%}\tPrecision: {prec:3.2%}\tRecall: {recall:3.2%}\tF1 score: {f1:3.2%}\tAUC: {roc_auc}\n") 
        column_datas = {"Accuracy":accu,"Precision":prec,"Recall":recall,"F1 score":f1, "AUC":roc_auc}
        if tiktok:
            df = pd.DataFrame(column_datas, index=[0])
            df.to_csv(f"./results/column/{self.name}_tiktok_{random_seed}.csv", sep=",", index=None)
        else:
            df = pd.DataFrame(column_datas, index=[0])
            df.to_csv(f"./results/column/{self.name}_{random_seed}.csv", sep=",", index=None)


              
        # 存储运行数据，用于绘制ROC曲线
        roc_datas = {"FPR":fpr,"TPR":tpr,"THRESHOLD":thres}
        if tiktok:
            df = pd.DataFrame(roc_datas, index=None)
            df.to_csv(f"./results/roc/{self.name}_tiktok_{random_seed}.csv", sep=",", index=None)
        else:
            df = pd.DataFrame(roc_datas, index=None)
            df.to_csv(f"./results/roc/{self.name}_{random_seed}.csv", sep=",", index=None)
        

if __name__ == "__main__":
    for random_seed in range(5):
        print(f"current_seed:{random_seed}")
        ProcSystem = Inception_ProcSystem(eps=0.2, name="FaceNet")
        params_path = f"./model_params/{ProcSystem.name}/{ProcSystem.name}_{random_seed}.params"
        X_train, X_very,y_train, y_very,= dl_load_data("./dataset/IF_train.csv",random_seed, test_size=0.2) # 这一步已经打乱数据了
        
        test_datas = pd.read_csv("./dataset/IF_test.csv", sep=',')
        X_test = test_datas[['img_path', 'frequency', 'size', 'relative_size', 'acu', 'relative_acu']]
        y_test = test_datas.labels
    
        ProcSystem.train(X_train, y_train, X_very, y_very, params_path)
        ProcSystem.load_params(params_path)
        ProcSystem.test(X_test=X_test, y_test=y_test, random_seed=random_seed, tiktok=False)