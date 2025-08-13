from ..loss_fns import (
    MSELoss,
    CorrelationLoss,
    BCELoss
)

loss_dict = dict(
    MSELoss=MSELoss,
    CorrelationLoss=CorrelationLoss,
    BCELoss=BCELoss
)

def get_loss_fn(loss_fn):
    return loss_dict[loss_fn]