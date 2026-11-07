
import numpy as np
import torch


def preprocess_qpos(qpos_np: np.ndarray, stats: dict) -> torch.Tensor:
    """
    Normalize qpos using dataset stats. Returns torch.float32 tensor (CPU).
    Caller can move to device and add batch dim.
    """
    qpos = torch.as_tensor(qpos_np, dtype=torch.float32)
    qpos_mean = torch.as_tensor(stats["qpos_mean"], dtype=torch.float32)
    qpos_std = torch.as_tensor(stats["qpos_std"], dtype=torch.float32)
    return (qpos - qpos_mean) / qpos_std


def postprocess_action_to_env(
    action_norm: torch.Tensor | np.ndarray,
    stats: dict,
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

    # Denormalize
    action_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32)
    action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32)
    a_denorm = a * action_std + action_mean

    return a_denorm.numpy()
