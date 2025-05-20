import torch
import torch.nn as nn

class CorrelationLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(CorrelationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = torch.flatten(pred)
        target = torch.flatten(target)

        N = len(target)

        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        pred_std = torch.std(pred) + self.epsilon
        target_std = torch.std(target) + self.epsilon

        pred_hat = (pred-pred_mean)/pred_std
        target_hat = (target-target_mean)/target_std

        corr = torch.sum(pred_hat * target_hat)/(N-1)

        nom = 2 * corr * pred_std * target_std
        den = torch.square(pred_std) + torch.square(target_std) + torch.square(pred_mean - target_mean) + self.epsilon

        ccc = nom/den

        loss = 1-ccc

        return loss