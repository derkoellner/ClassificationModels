from ..loss_fns import (
    MSELoss,
    CorrelationLoss
)

loss_dict = dict(
    MSELoss=MSELoss,
    CorrelationLoss=CorrelationLoss
)

def get_loss_fn(loss_fn):
    return loss_dict[loss_fn]