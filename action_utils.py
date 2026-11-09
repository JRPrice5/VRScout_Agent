import torch


def compute_continuous_loss(pred_actions, target_actions):
    """
    Compute separate losses for continuous components.

    Args:
        pred_actions: Predicted actions from model [batch_size, seq_len, action_dim]
        target_actions: Target actions from dataset [batch_size, seq_len, action_dim]
    Returns:
        loss: continuous loss
    """
    return torch.nn.functional.l1_loss(
        pred_actions, target_actions, reduction="none"
    ).mean()
