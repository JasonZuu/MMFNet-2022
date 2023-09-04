import torch
import torch.nn as nn
import torch.nn.functional as F


class FringeLoss(nn.Module):
    def __init__(self, eps, device):
        super(Fringe_Loss, self).__init__()
        self.eps = eps
        self.device = device

    def forward(self, input, target):
        u_k = torch.normal(mean=0.5, std=0.15, size=target.shape)
        u_k = u_k.to(self.device)
        H_pq = F.binary_cross_entropy(input, target)
        H_pn = F.binary_cross_entropy(input, u_k)
        p = torch.exp(-H_pq)
        loss = p*((1-self.eps)*H_pq + self.eps*H_pn)
        return loss.mean()