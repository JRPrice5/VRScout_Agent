"""
Debug utilities for log probability computation in ACT RL training.
"""

import torch
import numpy as np
from .log_prob_utils import compute_gaussian_log_prob


def debug_log_prob_computation(actions, mu, logvar, name=""):
    """
    Debug log probability computation with detailed logging.
    """
    print(f"\n=== DEBUG LOG PROB COMPUTATION ({name}) ===")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions range: [{actions.min():.6f}, {actions.max():.6f}]")
    print(f"Actions mean: {actions.mean():.6f}, std: {actions.std():.6f}")
    
    print(f"Mu shape: {mu.shape}")
    print(f"Mu range: [{mu.min():.6f}, {mu.max():.6f}]")
    print(f"Mu mean: {mu.mean():.6f}, std: {mu.std():.6f}")
    
    print(f"Logvar shape: {logvar.shape}")
    print(f"Logvar range: [{logvar.min():.6f}, {logvar.max():.6f}]")
    print(f"Logvar mean: {logvar.mean():.6f}, std: {logvar.std():.6f}")
    
    # Compute std
    std = torch.exp(0.5 * logvar)
    print(f"Std range: [{std.min():.6f}, {std.max():.6f}]")
    print(f"Std mean: {std.mean():.6f}, std: {std.std():.6f}")
    
    # Compute log probability
    log_prob = compute_gaussian_log_prob(
        actions=actions,
        mu=mu,
        logvar=logvar,
        aggregate_method="mean",
        numerical_stability=True
    )
    
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Log prob range: [{log_prob.min():.6f}, {log_prob.max():.6f}]")
    print(f"Log prob mean: {log_prob.mean():.6f}, std: {log_prob.std():.6f}")
    
    # Check for potential issues
    if torch.isnan(log_prob).any():
        print("WARNING: NaN detected in log probabilities!")
    if torch.isinf(log_prob).any():
        print("WARNING: Inf detected in log probabilities!")
    if (log_prob < -100).any():
        print("WARNING: Very negative log probabilities detected!")
    if (log_prob > 10).any():
        print("WARNING: Very positive log probabilities detected!")
    
    print("=== END DEBUG ===\n")
    
    return log_prob


def analyze_ppo_ratio(old_log_probs, new_log_probs, name=""):
    """
    Analyze PPO ratio computation for potential issues.
    """
    print(f"\n=== ANALYZE PPO RATIO ({name}) ===")
    print(f"Old log probs range: [{old_log_probs.min():.6f}, {old_log_probs.max():.6f}]")
    print(f"Old log probs mean: {old_log_probs.mean():.6f}, std: {old_log_probs.std():.6f}")
    
    print(f"New log probs range: [{new_log_probs.min():.6f}, {new_log_probs.max():.6f}]")
    print(f"New log probs mean: {new_log_probs.mean():.6f}, std: {new_log_probs.std():.6f}")
    
    log_prob_diff = new_log_probs - old_log_probs
    print(f"Log prob diff range: [{log_prob_diff.min():.6f}, {log_prob_diff.max():.6f}]")
    print(f"Log prob diff mean: {log_prob_diff.mean():.6f}, std: {log_prob_diff.std():.6f}")
    
    # Clamp like in PPO
    log_prob_diff_clamped = torch.clamp(log_prob_diff, min=-20.0, max=20.0)
    ratio = torch.exp(log_prob_diff_clamped)
    
    print(f"Ratio range: [{ratio.min():.6f}, {ratio.max():.6f}]")
    print(f"Ratio mean: {ratio.mean():.6f}, std: {ratio.std():.6f}")
    
    # Check for potential issues
    if (torch.abs(log_prob_diff) > 10).any():
        print("WARNING: Large log probability differences detected!")
        print(f"Number of large diffs: {(torch.abs(log_prob_diff) > 10).sum().item()}")
    
    if (ratio > 10).any() or (ratio < 0.1).any():
        print("WARNING: Extreme ratios detected!")
        print(f"Number of ratios > 10: {(ratio > 10).sum().item()}")
        print(f"Number of ratios < 0.1: {(ratio < 0.1).sum().item()}")
    
    print("=== END ANALYZE ===\n")
    
    return ratio


def suggest_log_prob_fixes():
    """
    Suggest potential fixes for log probability issues.
    """
    print("\n=== SUGGESTED FIXES ===")
    print("1. Check if action and latent dimensions match properly")
    print("2. Verify that mu and logvar are in reasonable ranges")
    print("3. Consider normalizing actions before log prob computation")
    print("4. Check if the latent space is properly initialized")
    print("5. Verify that the DETR VAE is producing reasonable latent distributions")
    print("6. Consider using a smaller learning rate for PPO")
    print("7. Check if rewards are properly normalized")
    print("8. Verify that advantages are computed correctly")
    print("========================\n")
