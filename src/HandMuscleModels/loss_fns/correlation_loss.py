import torch
import torch.nn as nn

class CorrelationLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8, reduction: str = 'mean'):  # CHANGED: added reduction parameter
        super(CorrelationLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction  # CHANGED: store reduction method

    def forward(self, pred, target):
        # CHANGED: per-sample flatten to preserve batch dimension
        pred_flat = pred.reshape(pred.size(0), -1)  # shape (B, N)
        target_flat = target.reshape(target.size(0), -1)  # shape (B, N)

        N = pred_flat.size(1)  # number of elements per sample

        # compute per-sample means
        pred_mean = pred_flat.mean(dim=1, keepdim=True)  # shape (B,1)
        target_mean = target_flat.mean(dim=1, keepdim=True)  # shape (B,1)

        # compute population variance
        pred_var = torch.mean((pred_flat - pred_mean) ** 2, dim=1, keepdim=True) + self.epsilon  # shape (B,1)
        target_var = torch.mean((target_flat - target_mean) ** 2, dim=1, keepdim=True) + self.epsilon  # shape (B,1)

        # standard deviations
        pred_std = torch.sqrt(pred_var)  # shape (B,1)
        target_std = torch.sqrt(target_var)  # shape (B,1)

        # normalize
        pred_hat = (pred_flat - pred_mean) / pred_std  # shape (B, N)
        target_hat = (target_flat - target_mean) / target_std  # shape (B, N)

        # compute population correlation per sample
        corr = torch.sum(pred_hat * target_hat, dim=1, keepdim=True) / N  # shape (B,1)

        # concordance correlation coefficient components
        nom = 2 * corr * pred_std * target_std  # shape (B,1)
        den = pred_var + target_var + (pred_mean - target_mean) ** 2 + self.epsilon  # shape (B,1)

        ccc = nom / den  # shape (B,1)
        ccc = ccc.squeeze(1)  # CHANGED: shape (B,)

        loss = 1 - ccc  # tensor of shape (B,)

        # CHANGED: apply reduction
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:  # default 'mean'
            return loss.mean()  # CHANGED