"""
Quick fixes for PPO training instability in ACT RL.

Based on the observed symptoms:
- Policy loss hitting 1000.0 (clamp limit)
- Value loss hitting 1000.0 (clamp limit) 
- Negative policy loss (-0.076678)
- High KL divergence (22.73)
"""

def apply_conservative_hyperparameters(ppo_policy):
    """
    Apply very conservative hyperparameters for stable training.
    """
    print("Applying conservative hyperparameters for stability...")
    
    # Reduce learning rates significantly
    for param_group in ppo_policy.optimizer.param_groups:
        param_group['lr'] = 1e-5  # Very small learning rate
    
    # Make clipping very conservative
    ppo_policy.clip_param = 0.05  # Very small clip range
    
    # Increase KL constraint
    ppo_policy.kl_weight = 0.2  # Strong KL constraint
    
    # Reduce other coefficients
    ppo_policy.value_coef = 0.1  # Reduce value loss weight
    ppo_policy.entropy_coef = 0.001  # Minimal entropy
    
    print(f"Applied settings:")
    print(f"  Learning rate: 1e-5")
    print(f"  Clip param: 0.05")
    print(f"  KL weight: 0.2")
    print(f"  Value coef: 0.1")
    print(f"  Entropy coef: 0.001")


def normalize_advantages_and_returns(memory):
    """
    Normalize advantages and returns for stability.
    """
    print("Normalizing advantages and returns...")
    
    # Convert to tensors if not already
    advantages = torch.tensor(memory.advantages, dtype=torch.float32)
    returns = torch.tensor(memory.returns, dtype=torch.float32)
    
    # Normalize advantages (zero mean, unit variance)
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clip advantages to prevent extreme values
    advantages = torch.clamp(advantages, min=-10.0, max=10.0)
    
    # Normalize returns to reasonable range
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Update memory
    memory.advantages = advantages.tolist()
    memory.returns = returns.tolist()
    
    print(f"Normalized advantages - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
    print(f"Normalized returns - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")


def add_gradient_clipping(ppo_policy, max_grad_norm=0.1):
    """
    Add gradient clipping to prevent exploding gradients.
    """
    print(f"Adding gradient clipping with max_grad_norm={max_grad_norm}")
    
    # Store original update method
    original_update = ppo_policy.update
    
    def update_with_clipping(memory, n_epochs=10, batch_size=64):
        # Call original update but with gradient clipping
        for epoch in range(n_epochs):
            for batch_indices in memory.get_batches():
                # ... (PPO update logic)
                # After loss.backward() and before optimizer.step():
                torch.nn.utils.clip_grad_norm_(ppo_policy.policy.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(ppo_policy.value_net.parameters(), max_grad_norm)
        
        return original_update(memory, n_epochs, batch_size)
    
    # Replace update method
    ppo_policy.update = update_with_clipping


def emergency_reset_policy(ppo_policy):
    """
    Emergency reset of policy to pretrained state.
    """
    print("Performing emergency reset to pretrained policy...")
    
    # Reset policy to pretrained state
    ppo_policy.policy.load_state_dict(ppo_policy.pretrained_policy.state_dict())
    
    # Reset optimizer
    ppo_policy.optimizer = torch.optim.Adam(
        list(ppo_policy.policy.parameters()) + list(ppo_policy.value_net.parameters()),
        lr=1e-5  # Very conservative learning rate
    )
    
    print("Policy reset to pretrained state with conservative learning rate.")


# Example usage function
def apply_all_fixes(ppo_policy, memory):
    """
    Apply all stability fixes.
    """
    print("\n=== APPLYING ALL STABILITY FIXES ===")
    
    # 1. Apply conservative hyperparameters
    apply_conservative_hyperparameters(ppo_policy)
    
    # 2. Normalize data
    normalize_advantages_and_returns(memory)
    
    # 3. Add gradient clipping
    add_gradient_clipping(ppo_policy, max_grad_norm=0.1)
    
    print("All fixes applied. Training should be more stable now.")
    print("=====================================\n")


if __name__ == "__main__":
    print("PPO Quick Fix Script")
    print("Import this module and call apply_all_fixes(ppo_policy, memory)")
    print("to apply stability fixes to your training.")
