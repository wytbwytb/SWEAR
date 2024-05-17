import torch
import torch.nn as nn
import numpy
import mmcv
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def kl_loss(pred, target, pred_var, beta=1.0):
    """KL loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss_sml1 = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    loss_kl = torch.exp(-pred_var) * loss_sml1 + 0.5 * pred_var
    # loss_sml1 = loss_sml1 * loss_sml1
    
    pred_var = torch.sigmoid(pred_var)
    # loss_kl = torch.abs(torch.exp(-pred_var) * loss_sml1 + 0.5 * pred_var)
    # loss_kl = torch.exp(-pred_var) * loss_sml1 + 0.5 * pred_var
    loss_kl = - torch.log(torch.exp(- (pred - target) ** 2.0 / pred_var / 2.0) / 
                          torch.sqrt(2.0 * numpy.pi * pred_var) + 1e-9)
    # loss_kl = 0.5 * torch.exp(-pred_var) * loss_sml1 + 0.5 * pred_var
    return loss_kl


@ROTATED_LOSSES.register_module()
class KLLoss(nn.Module):
    
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self,
                pred,
                target,
                variances, 
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_kl = self.loss_weight * kl_loss(
            pred,
            target,
            weight,
            pred_var=variances,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
    
        return loss_kl