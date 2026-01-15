import torch
import torch.nn as nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        logpt = -F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1-pt)**self.gamma) * logpt
        return loss.mean()
