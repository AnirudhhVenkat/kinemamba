# Adapted from facebookresearch/jepa app/vjepa/train.py
# https://github.com/facebookresearch/jepa

import torch
import torch.nn.functional as F


def jepa_prediction_loss(
    predictions,
    targets,
    loss_exp: float = 2.0,
) -> torch.Tensor:
    """L^p distance between predictor outputs and target-encoder tokens."""
    loss = 0.0
    for z, h in zip(predictions, targets):
        loss += torch.mean(torch.abs(z - h) ** loss_exp) / loss_exp
    return loss / len(targets)


def predictor_variance_loss(predictions, eps: float = 1e-4) -> torch.Tensor:
    """Encourage predictor outputs to maintain variance across tokens."""
    pstd = sum(torch.sqrt(z.var(dim=1) + eps) for z in predictions) / len(predictions)
    return torch.mean(F.relu(1.0 - pstd))


def jepa_loss(
    predictions,
    targets,
    loss_exp: float = 2.0,
    reg_coeff: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_jepa = jepa_prediction_loss(predictions, targets, loss_exp=loss_exp)
    loss_reg = predictor_variance_loss(predictions)
    total = loss_jepa + reg_coeff * loss_reg
    return total, loss_jepa, loss_reg


@torch.no_grad()
def update_target_encoder(
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    momentum: float,
) -> None:
    """EMA update of target encoder weights."""
    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
        param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)
