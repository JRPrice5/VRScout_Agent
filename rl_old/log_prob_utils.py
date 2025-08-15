"""
Shared utility functions for consistent log probability calculation in ACT RL training.

This module provides unified log probability computation to ensure consistency between
the Unity VR RL interface and PPO agent, preventing training instabilities caused by
mismatched log probability calculations.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_gaussian_log_prob(
    actions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    aggregate_method: str = "mean",
    numerical_stability: bool = True
) -> torch.Tensor:
    """
    Compute log probability of actions under a Gaussian distribution defined by mu and logvar.
    
    This function provides consistent log probability calculation for ACT RL training,
    handling both single actions and batched actions with multiple queries.
    
    Args:
        actions: Action tensor of shape [batch_size, action_dim] or [batch_size, num_queries, action_dim]
        mu: Mean of Gaussian distribution, shape [batch_size, latent_dim]
        logvar: Log variance of Gaussian distribution, shape [batch_size, latent_dim]
        aggregate_method: How to aggregate across queries ("mean" or "sum")
        numerical_stability: Whether to apply numerical stability measures
        
    Returns:
        log_prob: Log probabilities of shape [batch_size]
    """
    if mu is None or logvar is None:
        raise ValueError("mu and logvar cannot be None for log probability computation")
    
    # Compute standard deviation
    std = torch.exp(0.5 * logvar)
    
    if numerical_stability:
        # Apply numerical stability measures
        std = torch.clamp(std, min=1e-6, max=100.0)
        actions = torch.clamp(actions, min=-10.0, max=10.0)
    
    # Handle different action tensor shapes
    if len(actions.shape) == 2:
        # Single action per batch: [batch_size, action_dim]
        batch_size, action_dim = actions.shape
        
        # Ensure mu and actions have compatible dimensions
        if mu.shape[1] != action_dim:
            # If dimensions don't match, we need to handle the latent space properly
            # For now, we'll assume the action_dim matches the latent_dim
            if mu.shape[1] < action_dim:
                # Pad mu and std to match action dimensions (this is an approximation)
                padding_size = action_dim - mu.shape[1]
                mu_padded = torch.cat([mu, torch.zeros(batch_size, padding_size, device=mu.device)], dim=1)
                std_padded = torch.cat([std, torch.ones(batch_size, padding_size, device=std.device)], dim=1)
                mu, std = mu_padded, std_padded
            else:
                # Truncate actions to match latent dimensions
                actions = actions[:, :mu.shape[1]]
        
        # Compute log probability using Gaussian formula
        # log_prob = -0.5 * ((x - mu)^2 / std^2 + log(2*pi*std^2))
        diff_squared = (actions - mu) ** 2
        variance = std ** 2
        
        if numerical_stability:
            exp_term = torch.clamp(diff_squared / variance, max=50.0)
            log_term = torch.clamp(torch.log(2 * np.pi * variance), min=-50.0, max=50.0)
        else:
            exp_term = diff_squared / variance
            log_term = torch.log(2 * np.pi * variance)
        
        log_prob_per_dim = -0.5 * (exp_term + log_term)
        
        if numerical_stability:
            log_prob_per_dim = torch.clamp(log_prob_per_dim, min=-50.0, max=50.0)
        
        # Sum over action dimensions
        log_prob = log_prob_per_dim.sum(dim=-1)
        
    elif len(actions.shape) == 3:
        # Multiple queries per batch: [batch_size, num_queries, action_dim]
        batch_size, num_queries, action_dim = actions.shape
        
        # Ensure mu and actions have compatible dimensions
        if mu.shape[1] != action_dim:
            if mu.shape[1] < action_dim:
                padding_size = action_dim - mu.shape[1]
                mu_padded = torch.cat([mu, torch.zeros(batch_size, padding_size, device=mu.device)], dim=1)
                std_padded = torch.cat([std, torch.ones(batch_size, padding_size, device=std.device)], dim=1)
                mu, std = mu_padded, std_padded
            else:
                actions = actions[:, :, :mu.shape[1]]
                action_dim = mu.shape[1]
        
        # Expand mu and std to match query dimension
        mu_expanded = mu.unsqueeze(1).expand(-1, num_queries, -1)  # [batch_size, num_queries, latent_dim]
        std_expanded = std.unsqueeze(1).expand(-1, num_queries, -1)  # [batch_size, num_queries, latent_dim]
        
        # Compute log probability for each query
        diff_squared = (actions - mu_expanded) ** 2
        variance = std_expanded ** 2
        
        if numerical_stability:
            exp_term = torch.clamp(diff_squared / variance, max=50.0)
            log_term = torch.clamp(torch.log(2 * np.pi * variance), min=-50.0, max=50.0)
        else:
            exp_term = diff_squared / variance
            log_term = torch.log(2 * np.pi * variance)
        
        log_prob_per_dim = -0.5 * (exp_term + log_term)
        
        if numerical_stability:
            log_prob_per_dim = torch.clamp(log_prob_per_dim, min=-50.0, max=50.0)
        
        # Sum over action dimensions for each query
        query_log_probs = log_prob_per_dim.sum(dim=-1)  # [batch_size, num_queries]
        
        # Aggregate across queries
        if aggregate_method == "mean":
            log_prob = query_log_probs.mean(dim=1)  # [batch_size]
        elif aggregate_method == "sum":
            log_prob = query_log_probs.sum(dim=1)  # [batch_size]
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")
    
    else:
        raise ValueError(f"Unsupported action tensor shape: {actions.shape}")
    
    # Final numerical stability check
    if numerical_stability:
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            print(f"[WARNING] NaN or Inf detected in log probabilities!")
            log_prob = torch.where(
                torch.isnan(log_prob) | torch.isinf(log_prob), 
                torch.tensor(-10.0, device=log_prob.device), 
                log_prob
            )
        log_prob = torch.clamp(log_prob, min=-50.0, max=10.0)
    
    return log_prob


def compute_inference_mode_log_prob(actions: torch.Tensor) -> float:
    """
    Compute approximate log probability for inference mode when mu/logvar are not available.
    
    Args:
        actions: Action tensor of any shape
        
    Returns:
        log_prob: Approximate log probability as a scalar
    """
    # Simple approximation based on action magnitude
    # This is consistent with the original implementation
    action_flat = actions.flatten()
    log_prob = -0.5 * torch.sum(action_flat**2).item()
    return log_prob


def validate_log_prob_inputs(
    actions: torch.Tensor,
    mu: Optional[torch.Tensor],
    logvar: Optional[torch.Tensor]
) -> Tuple[bool, str]:
    """
    Validate inputs for log probability computation.
    
    Returns:
        (is_valid, error_message)
    """
    if actions is None:
        return False, "actions cannot be None"
    
    if len(actions.shape) < 2 or len(actions.shape) > 3:
        return False, f"actions must be 2D or 3D tensor, got shape {actions.shape}"
    
    if mu is not None and logvar is not None:
        if mu.shape != logvar.shape:
            return False, f"mu and logvar must have same shape, got {mu.shape} and {logvar.shape}"
        
        if len(mu.shape) != 2:
            return False, f"mu and logvar must be 2D tensors, got shape {mu.shape}"
        
        if actions.shape[0] != mu.shape[0]:
            return False, f"batch size mismatch: actions {actions.shape[0]} vs mu {mu.shape[0]}"
    
    return True, ""
