import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader

from configs.train_configs import MMFNetConfig
from mmfnet import MMFNet


@ torch.no_grad()
def test_fn(model: MMFNet, test_config: MMFNetConfig, test_dataset):
    print("Start Testing")
    model = model.to(test_config.device)

    test_loader = DataLoader(test_dataset, batch_size=test_config.batch_size,
                             shuffle=False, num_workers=4)

    model.eval()

    all_labels = []
    all_y_pred = []
    all_y_score = []

    pbar = tqdm(total=len(test_loader), desc="Testing")
    for images, structs, labels in test_loader:
        images, structs, labels = images.to(test_config.device), structs.to(test_config.device), labels.to(test_config.device)

        y_logits = model(images, structs)
        y_score = torch.sigmoid(y_logits).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)

        all_labels.extend(labels.cpu().numpy().astype(np.int8))
        all_y_pred.extend(y_pred)
        all_y_score.extend(y_score)

        pbar.update()
    pbar.close()

    roauc = roc_auc_score(y_true=all_labels, y_score=np.vstack(all_y_score))
    f1 = f1_score(y_true=all_labels, y_pred=all_y_pred)
    acc = accuracy_score(y_true=all_labels, y_pred=all_y_pred)

    test_result = {"roauc": roauc, "f1": f1, "acc": acc}
    return test_result
