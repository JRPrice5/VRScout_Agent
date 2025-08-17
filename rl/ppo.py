import torch
import torch.nn.functional as F


def compute_gae(r: torch.Tensor, v: torch.Tensor, dones: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float):
    """
    Generalized Advantage Estimation.

    Supports two input layouts:
    - Batched sequences: r, v, dones, mask with shape [B, T]
    - Flattened sequence: r, v, dones, mask with shape [N]

    Returns (advantages, returns) with the same shape as inputs.
    """
    print(f"Mask: {mask.shape}")
    print(f"dones: {dones.shape}")
    print(f"r: {r.shape}")
    print(f"v: {v.shape}")
    B, T = r.shape
    adv = torch.zeros_like(r)
    lastgaelam = torch.zeros(B, device=r.device, dtype=r.dtype)
    v_next = torch.zeros(B, device=r.device, dtype=r.dtype)
    for t in reversed(range(T)):
        nonterminal = (1.0 - dones[:, t])
        delta = r[:, t] + gamma * v_next * nonterminal - v[:, t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[:, t] = lastgaelam
        v_next = v[:, t]
    ret = adv + v
    # mask invalid timesteps
    adv = adv * mask
    ret = ret * mask
    return adv, ret


def ppo_clip_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip_eps: float, mask: torch.Tensor):
    """
    logp, logp_old, adv, mask: [B, T]
    returns scalar loss (negative surrogate, to minimize)
    """
    # Clamp the log-prob difference to avoid numerical overflow/instability in exp
    dv = torch.clamp(logp - logp_old, -20.0, 20.0)
    ratio = torch.exp(dv)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    # take min and average over masked positions
    loss = -torch.minimum(unclipped, clipped)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def value_loss(v: torch.Tensor, ret: torch.Tensor, mask: torch.Tensor):
    loss = 0.5 * (v - ret) ** 2 #TODO check whether this is correct for MCE l1 loss
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom
