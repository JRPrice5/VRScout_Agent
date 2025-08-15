"""
PPO Training Diagnostics and Fixes for ACT RL Training
"""

import torch
import numpy as np


def diagnose_ppo_training_issues(policy_loss, value_loss, kl_div, entropy):
    """
    Diagnose common PPO training issues based on loss values.
    """
    print("\n=== PPO TRAINING DIAGNOSTICS ===")
    
    issues_found = []
    suggestions = []
    
    # Check policy loss
    if abs(policy_loss) >= 1000.0:
        issues_found.append("Policy loss hitting clamp limits (±1000)")
        suggestions.append("Reduce learning rate or clip gradient norms")
        suggestions.append("Check for extreme log probability differences")
    
    if policy_loss < 0 and abs(policy_loss) > 1.0:
        issues_found.append("Large negative policy loss (unusual for PPO)")
        suggestions.append("Check if advantages are computed correctly")
        suggestions.append("Verify reward normalization")
    
    # Check value loss
    if value_loss > 100.0:
        issues_found.append("Very high value loss")
        suggestions.append("Reduce value function learning rate")
        suggestions.append("Check if returns are properly computed")
        suggestions.append("Consider value function clipping")
    
    # Check KL divergence
    if kl_div > 1.0:
        issues_found.append("Very high KL divergence")
        suggestions.append("Increase KL weight to constrain policy updates")
        suggestions.append("Reduce policy learning rate")
        suggestions.append("Check if pretrained policy is loading correctly")
    
    # Check entropy
    if entropy < 0.1:
        issues_found.append("Very low entropy (policy becoming deterministic)")
        suggestions.append("Increase entropy coefficient")
        suggestions.append("Check if policy is collapsing")
    
    if entropy > 10.0:
        issues_found.append("Very high entropy (policy too random)")
        suggestions.append("Decrease entropy coefficient")
        suggestions.append("Check if policy is properly initialized")
    
    # Print results
    if issues_found:
        print("ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        
        print("\nSUGGESTED FIXES:")
        for i, suggestion in enumerate(set(suggestions), 1):
            print(f"  {i}. {suggestion}")
    else:
        print("No obvious issues detected in loss values.")
    
    print("=== END DIAGNOSTICS ===\n")
    
    return issues_found, suggestions


def suggest_hyperparameter_adjustments(current_lr=3e-4, current_clip=0.2, current_kl_weight=0.05):
    """
    Suggest hyperparameter adjustments for stable training.
    """
    print("\n=== HYPERPARAMETER SUGGESTIONS ===")
    print("For more stable training, consider:")
    print(f"1. Reduce learning rate from {current_lr} to {current_lr * 0.1}")
    print(f"2. Reduce clip parameter from {current_clip} to {current_clip * 0.5}")
    print(f"3. Increase KL weight from {current_kl_weight} to {current_kl_weight * 2}")
    print("4. Add gradient clipping (max_grad_norm=0.5)")
    print("5. Use smaller batch sizes initially")
    print("6. Normalize advantages (subtract mean, divide by std)")
    print("7. Clip value function updates")
    print("================================\n")


def check_data_quality(rewards, advantages, returns):
    """
    Check the quality of training data.
    """
    print("\n=== DATA QUALITY CHECK ===")
    
    # Check rewards
    print(f"Rewards - Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}")
    print(f"Rewards - Range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    
    if torch.isnan(rewards).any():
        print("WARNING: NaN detected in rewards!")
    
    if (rewards == 0).all():
        print("WARNING: All rewards are zero!")
    
    # Check advantages
    print(f"Advantages - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
    print(f"Advantages - Range: [{advantages.min():.4f}, {advantages.max():.4f}]")
    
    if torch.isnan(advantages).any():
        print("WARNING: NaN detected in advantages!")
    
    if advantages.std() < 1e-6:
        print("WARNING: Advantages have very low variance!")
    
    # Check returns
    print(f"Returns - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
    print(f"Returns - Range: [{returns.min():.4f}, {returns.max():.4f}]")
    
    if torch.isnan(returns).any():
        print("WARNING: NaN detected in returns!")
    
    print("=== END DATA CHECK ===\n")


def emergency_training_fixes():
    """
    Emergency fixes for unstable training.
    """
    print("\n=== EMERGENCY FIXES ===")
    print("If training is completely unstable:")
    print("1. Set learning_rate = 1e-5 (very small)")
    print("2. Set clip_param = 0.05 (very conservative)")
    print("3. Set kl_weight = 0.1 (strong constraint)")
    print("4. Set entropy_coef = 0.001 (minimal exploration)")
    print("5. Use batch_size = 16 (small batches)")
    print("6. Set max_grad_norm = 0.1 (strong gradient clipping)")
    print("7. Normalize all inputs (actions, states, rewards)")
    print("8. Start with pretrained policy frozen for first few episodes")
    print("========================\n")
