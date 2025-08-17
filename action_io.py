import numpy as np
import torch
from typing import Sequence, Union
from action_utils import ACTION_INDICES


def preprocess_qpos(qpos_np: np.ndarray, stats: dict) -> torch.Tensor:
    """
    Normalize qpos using dataset stats. Returns torch.float32 tensor (CPU).
    Caller can move to device and add batch dim.
    """
    qpos = torch.as_tensor(qpos_np, dtype=torch.float32)
    qpos_mean = torch.as_tensor(stats['qpos_mean'], dtype=torch.float32)
    qpos_std = torch.as_tensor(stats['qpos_std'], dtype=torch.float32)
    return (qpos - qpos_mean) / qpos_std


def postprocess_action_to_env(
    action_norm: Union[torch.Tensor, np.ndarray],
    stats: dict,
    boolean_indices: Sequence[int]
) -> np.ndarray:
    """
    Convert normalized action to environment space:
    - Apply sigmoid to boolean indices to map to [0,1]
    - Denormalize full action with dataset stats

    Args:
        action_norm: 1D tensor/array of shape [A] in normalized space
        stats: dict with 'action_mean' and 'action_std'
        boolean_indices: indices of boolean action dims

    Returns:
        Numpy array in env action space (same shape as input)
    """
    if isinstance(action_norm, np.ndarray):
        a = torch.from_numpy(action_norm).float()
    else:
        a = action_norm.detach().float().cpu()

    # Handle boolean dims: if values are outside [0,1], treat as logits and apply sigmoid;
    # if already within [0,1] (e.g., Bernoulli samples or probabilities), leave as-is.
    if len(boolean_indices) > 0:
        a_b = a[boolean_indices]
        if (a_b.min() < 0.0) or (a_b.max() > 1.0):
            a[boolean_indices] = torch.sigmoid(a_b)

    # Denormalize
    action_mean = torch.as_tensor(stats['action_mean'], dtype=torch.float32)
    action_std = torch.as_tensor(stats['action_std'], dtype=torch.float32)
    a_denorm = a * action_std + action_mean

    return a_denorm.numpy()


def split_actions_normalized(action_norm: torch.Tensor):
    """
    Split normalized action tensor into continuous and binary parts using ACTION_INDICES.
    Accepts shapes [..., A]. Returns (cont, bin) with matching leading dims.
    """
    cont_idx = ACTION_INDICES['continuous']
    bin_idx = ACTION_INDICES['boolean']
    cont = action_norm[..., cont_idx]
    binv = action_norm[..., bin_idx]
    return cont, binv


def merge_actions_normalized(cont: torch.Tensor, binv: torch.Tensor, device=None) -> torch.Tensor:
    """
    Merge continuous and binary parts back into full normalized action vector using ACTION_INDICES.
    cont: [..., Cc], binv: [..., Cb]
    Returns: [..., A]
    """
    A = max(ACTION_INDICES['continuous'] + ACTION_INDICES['boolean']) + 1
    out_shape = cont.shape[:-1] + (A,)
    out = torch.zeros(out_shape, dtype=cont.dtype, device=device or cont.device)
    out[..., ACTION_INDICES['continuous']] = cont
    out[..., ACTION_INDICES['boolean']] = binv
    return out
