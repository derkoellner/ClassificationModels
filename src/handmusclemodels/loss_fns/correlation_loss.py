import torch
import torch.nn as nn

class CorrelationLoss(nn.Module):
    """
    Concordance Correlation Coefficient loss: loss = 1 - CCC
    CCC = 2 * cov(x, y) / (var_x + var_y + (mean_x - mean_y)^2 + eps)
    This implementation computes per-sample CCC and reduces per `reduction`.
    """
    def __init__(self, eps: float = 1e-8, reduction: str = "mean", dtype=torch.float32):
        super().__init__()
        self.eps = float(eps)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.dtype = dtype

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        device = pred.device
        pred = pred.to(dtype=self.dtype, device=device)
        target = target.to(dtype=self.dtype, device=device)

        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        mu_x = pred_flat.mean(dim=1, keepdim=True)
        mu_y = target_flat.mean(dim=1, keepdim=True)

        x_centered = pred_flat - mu_x
        y_centered = target_flat - mu_y

        cov_xy = (x_centered * y_centered).mean(dim=1, keepdim=True)
        var_x = (x_centered * x_centered).mean(dim=1, keepdim=True)
        var_y = (y_centered * y_centered).mean(dim=1, keepdim=True)

        numer = 2.0 * cov_xy
        denom = var_x + var_y + (mu_x - mu_y).pow(2) + self.eps
        ccc = (numer / denom).squeeze(1)

        loss_per_sample = 1.0 - ccc

        if self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction == "none":
            return loss_per_sample
        else:
            return loss_per_sample.mean()