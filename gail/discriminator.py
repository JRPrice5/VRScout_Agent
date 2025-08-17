import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [..., T]
    mask: same broadcastable shape with 1 for valid, 0 for pad
    returns scalar mean over masked elements
    """
    m = mask.to(x.dtype)
    return (x * m).sum() / (m.sum() + eps)


class GAILDiscriminator(nn.Module):
    def __init__(self, h_dim, cont_dim, bin_dim, hidden: int = 256, layers: int = 2, label_smooth: float = 0.9):
        super().__init__()
        self.label_smooth = label_smooth
        in_dim = h_dim + cont_dim + (bin_dim or 0)
        mlp = []
        for i in range(layers):
            mlp += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.ReLU()]
        self.net = nn.Sequential(*mlp, nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor, a_cont: torch.Tensor, a_bin: Optional[torch.Tensor], mask: torch.Tensor):
        """
        h: [B, T, H]
        a_cont: [B, T, Cc]
        a_bin: [B, T, Cb] or None
        mask: [B, T] (1/0)
        returns logits [B, T], prob [B, T]
        """
        x = [h, a_cont]
        if a_bin is not None:
            x.append(a_bin)
        x = torch.cat(x, dim=-1)
        logits = self.net(x).squeeze(-1)
        prob = torch.sigmoid(logits)
        return logits, prob


def gail_loss_D(prob_exp: torch.Tensor,
                prob_pol: torch.Tensor,
                mask_exp: torch.Tensor,
                mask_pol: torch.Tensor,
                smooth: float = 0.0,
                eps: float = 1e-6) -> torch.Tensor:
    """
    Proper discriminator loss with optional label smoothing on expert targets.
    - If smooth in (0,1), use BCE with target y_pos=smooth for experts.
    - Policy negatives remain target 0.
    Loss = -E_exp[ y*log D + (1-y)*log(1-D) ] - E_pol[ log(1 - D) ]
    """
    prob_exp_c = torch.clamp(prob_exp, eps, 1.0 - eps)
    prob_pol_c = torch.clamp(prob_pol, eps, 1.0 - eps)
    y_pos = float(smooth) if smooth is not None else 0.0
    # Expert term with smoothed positive target
    pos = -(y_pos * torch.log(prob_exp_c) + (1.0 - y_pos) * torch.log(1.0 - prob_exp_c))
    # Policy term (negative target 0)
    neg = -torch.log(1.0 - prob_pol_c)
    return masked_mean(pos, mask_exp) + masked_mean(neg, mask_pol)


def reward_from_prob(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # r = -log(1 - D(s,a))
    return -torch.log(torch.clamp(1.0 - prob, min=eps))
