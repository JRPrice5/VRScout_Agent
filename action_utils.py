import torch
import numpy as np

# Define the indices for different action components
# These indices are based on the qpos format in convert_new_vr_dataset.py
ACTION_INDICES = {
    # Controller positions and rotations (continuous)
    'continuous': list(range(0, 24)) + list(range(28, 32)),  # All except button states and world position
    
    # Button states (boolean)
    'boolean': list(range(24, 28)),  # Indices 24-27 are the button states
    
    # Specific button indices
    'lefthand_trigger': 24,
    'righthand_trigger': 25,
    'lefthand_grip': 26,
    'righthand_grip': 27,
}

def split_actions_by_type(actions):
    """
    Split action tensor into continuous and boolean components.
    
    Args:
        actions: Tensor of shape [batch_size, seq_len, action_dim]
        
    Returns:
        continuous_actions: Tensor containing only continuous action dimensions
        boolean_actions: Tensor containing only boolean action dimensions
    """
    continuous_indices = ACTION_INDICES['continuous']
    boolean_indices = ACTION_INDICES['boolean']
    
    continuous_actions = actions[:, :, continuous_indices]
    boolean_actions = actions[:, :, boolean_indices]
    
    return continuous_actions, boolean_actions

def compute_mixed_loss(pred_actions, target_actions, is_pad=None, continuous_weight=1.0, boolean_weight=1.0):
    """
    Compute separate losses for continuous and boolean action components.
    
    Args:
        pred_actions: Predicted actions from model [batch_size, seq_len, action_dim]
        target_actions: Target actions from dataset [batch_size, seq_len, action_dim]
        is_pad: Padding mask tensor [batch_size, seq_len]
        continuous_weight: Weight for continuous action loss
        boolean_weight: Weight for boolean action loss
        
    Returns:
        loss_dict: Dictionary containing individual losses and weighted total loss
    """
    
    # Split actions into continuous and boolean components
    pred_continuous, pred_boolean = split_actions_by_type(pred_actions)
    target_continuous, target_boolean = split_actions_by_type(target_actions)
    
    # Compute L1 loss for continuous actions
    continuous_l1 = torch.nn.functional.l1_loss(pred_continuous, target_continuous, reduction='none')
    
    # For boolean actions, ensure targets are strictly in the range [0, 1]
    target_boolean_binary = torch.clamp(target_boolean, min=0.0, max=1.0)
    
    # Apply sigmoid to predictions to ensure they're in [0, 1] range
    pred_boolean_sigmoid = torch.sigmoid(pred_boolean)
    
    # For boolean values, we use a combination of BCE and L1 loss
    # BCE for classification accuracy
    boolean_bce = torch.nn.functional.binary_cross_entropy(
        pred_boolean_sigmoid, target_boolean_binary, reduction='none'
    )
    
    # L1 for regression stability (helps with convergence)
    boolean_l1 = torch.nn.functional.l1_loss(pred_boolean_sigmoid, target_boolean_binary, reduction='none')
    
    # Combined boolean loss
    boolean_loss = 0.7 * boolean_bce + 0.3 * boolean_l1
    
    # Apply padding mask if provided
    if is_pad is not None:
        mask = ~is_pad.unsqueeze(-1)
        
        # For continuous actions: divide by (mask.sum() * continuous_action_dims)
        continuous_action_dims = continuous_l1.size(2)
        boolean_action_dims = boolean_loss.size(2)
        
        # New style normalization with proper dimension scaling
        continuous_l1 = (continuous_l1 * mask[:, :, :continuous_l1.size(2)]).sum() / (mask.sum() * continuous_action_dims)
        boolean_loss = (boolean_loss * mask[:, :, :boolean_loss.size(2)]).sum() / (mask.sum() * boolean_action_dims)
    else:
        continuous_l1 = continuous_l1.mean()
        boolean_loss = boolean_loss.mean()
    
    # Compute weighted total loss
    total_loss = continuous_weight * continuous_l1 + boolean_weight * boolean_loss
    
    return {
        'continuous_l1': continuous_l1,
        'boolean_bce': boolean_loss,  # Keep the name for backward compatibility
        'total_loss': total_loss
    }
