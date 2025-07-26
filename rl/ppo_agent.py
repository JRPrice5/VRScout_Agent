#!/usr/bin/env python
"""
PPO Agent for fine-tuning ACT model with KL constraint to original behavior policy.

This module implements Proximal Policy Optimization (PPO) with a KL divergence
constraint to the original behavior cloning policy to prevent excessive deviation
during reinforcement learning fine-tuning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.distributions import Normal
from collections import deque
import time
from .log_prob_utils import compute_gaussian_log_prob, validate_log_prob_inputs
from .debug_log_prob import debug_log_prob_computation, analyze_ppo_ratio
from rl.ppo_diagnostics import diagnose_ppo_training_issues, check_data_quality

# Add the parent directory to path to import ACT modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ACT modules
from policy import ACTPolicy, kl_divergence


class PPOMemory:
    """
    Memory buffer for PPO algorithm with support for storing trajectories
    and computing advantages.
    """
    def __init__(self, batch_size=64, max_eps_len=1000, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.images = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_log_probs = []
        self.advantages = []
        self.returns = []
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # For matching rewards from Unity client
        self.episode_start_times = []
        self.timestep_to_buffer_idx = {}
        
    def store(self, state, image, action, reward, done, value, action_log_prob, timestep):
        """Store a transition"""
        self.states.append(state)
        self.images.append(image)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_log_probs.append(action_log_prob)
        
        # Store mapping from timestep to buffer index for reward matching
        self.timestep_to_buffer_idx[timestep] = len(self.states) - 1
        
    def mark_episode_start(self, timestep):
        """Mark the start of a new episode"""
        self.episode_start_times.append(timestep)
        
    def match_rewards(self, timesteps, rewards):
        """Match rewards from Unity client to stored transitions"""
        for timestep, reward in zip(timesteps, rewards):
            if timestep in self.timestep_to_buffer_idx:
                idx = self.timestep_to_buffer_idx[timestep]
                self.rewards[idx] = reward
            else:
                print(f"Warning: Timestep {timestep} not found in buffer")
                
    def compute_advantages_and_returns(self):
        """Compute advantages and returns using GAE"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert lists to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        
        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        
        last_advantage = 0
        last_return = 0
        next_return = 0
        
        # Process episodes in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or self.dones[t]:
                next_value = 0
                next_advantage = 0
            else:
                next_value = values[t + 1]
                next_advantage = advantages[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * next_advantage
            
            # Returns
            returns[t] = rewards[t] + self.gamma * (1 - dones[t]) * next_return
            next_return = returns[t]
            
        self.advantages = advantages.cpu().numpy().tolist()
        self.returns = returns.cpu().numpy().tolist()
        
    def get_batches(self):
        """Get batches for PPO update"""
        batch_size = self.batch_size
        n_states = len(self.states)
        batch_starts = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]
        
        return batches
    
    def get_batch_data(self, indices):
        """Get batch data for given indices"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[PPOMemory] get_batch_data called with {len(indices)} indices")
        print(f"[PPOMemory] Memory sizes - states: {len(self.states)}, images: {len(self.images)}, actions: {len(self.actions)}")
        
        states = torch.tensor(np.array([self.states[i] for i in indices]), dtype=torch.float32).to(device)
        print(f"[PPOMemory] Loaded states with shape: {states.shape}")
        
        images = torch.stack([self.images[i] for i in indices]).to(device)
        print(f"[PPOMemory] Loaded images with shape: {images.shape}")
        
        # Check if actions are already PyTorch tensors
        if isinstance(self.actions[0], torch.Tensor):
            # Stack tensors directly instead of converting to numpy first
            print(f"[PPOMemory] Actions are PyTorch tensors with shape: {self.actions[0].shape}")
            actions = torch.stack([self.actions[i].to(device) for i in indices])
        else:
            # Original behavior for non-tensor actions
            print(f"[PPOMemory] Actions are not PyTorch tensors, converting from type: {type(self.actions[0])}")
            actions = torch.tensor(np.array([self.actions[i] for i in indices]), dtype=torch.float32).to(device)
        
        print(f"[PPOMemory] Final actions shape: {actions.shape}")
        
        old_log_probs = torch.tensor(np.array([self.action_log_probs[i] for i in indices]), dtype=torch.float32).to(device)
        print(f"[PPOMemory] Loaded old_log_probs with shape: {old_log_probs.shape}")
        
        returns = torch.tensor(np.array([self.returns[i] for i in indices]), dtype=torch.float32).to(device)
        print(f"[PPOMemory] Loaded returns with shape: {returns.shape}")
        
        advantages = torch.tensor(np.array([self.advantages[i] for i in indices]), dtype=torch.float32).to(device)
        print(f"[PPOMemory] Loaded advantages with shape: {advantages.shape}")
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        print(f"[PPOMemory] Returning batch data with shapes:")
        print(f"  - states: {states.shape}")
        print(f"  - images: {images.shape}")
        print(f"  - actions: {actions.shape}")
        print(f"  - old_log_probs: {old_log_probs.shape}")
        print(f"  - returns: {returns.shape}")
        print(f"  - advantages: {advantages.shape}")
        
        return states, images, actions, old_log_probs, returns, advantages
    
    def clear(self):
        """Clear memory"""
        self.states = []
        self.images = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_log_probs = []
        self.advantages = []
        self.returns = []
        self.timestep_to_buffer_idx = {}
        # Keep episode_start_times for reference


class PPOACTPolicy(nn.Module):
    """
    PPO policy for fine-tuning ACT model with KL constraint to original behavior policy.
    """
    def __init__(self, pretrained_policy, kl_weight=0.05, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        
        # Store the pretrained policy for KL constraint
        self.pretrained_policy = pretrained_policy
        for param in self.pretrained_policy.parameters():
            param.requires_grad = False
        self.pretrained_policy.eval()
        
        # Create a copy of the pretrained policy for fine-tuning
        # We'll use the same configuration but create a new instance
        args_override = {
            'num_queries': pretrained_policy.model.num_queries,
            'kl_weight': pretrained_policy.kl_weight,
            'hidden_dim': pretrained_policy.model.transformer.d_model,
            'dim_feedforward': pretrained_policy.model.transformer.encoder.layers[0].linear1.out_features,
            'lr': pretrained_policy.optimizer.param_groups[0]['lr'],
            'lr_backbone': pretrained_policy.optimizer.param_groups[1]['lr'],
            'backbone': 'resnet18',  # Assuming resnet18 is used
            'enc_layers': pretrained_policy.model.encoder.num_layers,
            'dec_layers': pretrained_policy.model.transformer.decoder.num_layers,
            'nheads': pretrained_policy.model.transformer.nhead,
            'chunk_size': pretrained_policy.model.num_queries,
            # CRITICAL FIX: Copy camera_names and other missing parameters
            'camera_names': getattr(pretrained_policy.model, 'camera_names', ['camera1']),
            'state_dim': getattr(pretrained_policy.model, 'state_dim', 32),
            'latent_dim': getattr(pretrained_policy.model, 'latent_dim', 32),
            # Add other required parameters with defaults
            'dropout': 0.1,
            'pre_norm': False,
            'weight_decay': 1e-4,
        }
        
        # Print the extracted configuration for debugging
        print(f"PPO Policy Configuration:")
        print(f"  dim_feedforward: {args_override['dim_feedforward']}")
        print(f"  camera_names: {args_override['camera_names']}")
        print(f"  state_dim: {args_override['state_dim']}")
        print(f"  latent_dim: {args_override['latent_dim']}")

        # Create a trainable policy
        self.policy = ACTPolicy(args_override)
        
        # Load weights from pretrained policy
        self.policy.load_state_dict(pretrained_policy.state_dict())
        
        # Create value network (separate from policy network)
        # Use latent_dim instead of hidden_dim since we're feeding mu (latent representation)
        latent_dim = args_override['latent_dim']
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        
        # Move to GPU
        self.value_net.cuda()
        
        # PPO hyperparameters
        self.kl_weight = kl_weight
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def forward(self, qpos, image):
        """Forward pass for inference"""
        return self.policy(qpos, image)
    
    def evaluate_actions(self, qpos, image, actions=None, is_pad=None):
        """
        Evaluate actions for PPO update.
        Returns action log probs, values, and entropy.
        """
        # Get features from policy network
        env_state = None
        
        # Debug print for original shape
        print(f"[evaluate_actions] Original image shape: {image.shape}")
        
        # Handle image tensor dimensions for conv2d compatibility
        # Handle 6D tensor [batch_size, dim1, dim2, channels, height, width]
        # This happens because we have a redundant dimension in the camera dimension
        if len(image.shape) == 6:
            batch_size, dim1, dim2, channels, height, width = image.shape
            # Squeeze out the redundant dimension (dim2 is usually 1)
            # This converts [batch_size, dim1, 1, channels, height, width] to [batch_size, dim1, channels, height, width]
            image = image.squeeze(2)
            print(f"[evaluate_actions] Squeezed redundant dim: {image.shape}")
        
        # Normalize image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        # Debug print for input tensor shapes
        print(f"[evaluate_actions] Input shapes - qpos: {qpos.shape}, image: {image.shape}")
        if actions is not None:
            print(f"[evaluate_actions] Actions shape: {actions.shape}")
        if is_pad is not None:
            print(f"[evaluate_actions] is_pad shape: {is_pad.shape}")
        
        # Check if actions is provided (training mode)
        if actions is not None:
            print(f"[evaluate_actions] Actions shape: {actions.shape}")
            # Check if we have an extra dimension [batch_size, 1, num_queries, action_dim]
            if len(actions.shape) == 4 and actions.shape[1] == 1:
                print(f"[evaluate_actions] Detected extra dimension in actions tensor, squeezing it")
                actions = actions.squeeze(1)  # Remove the extra dimension
                print(f"[evaluate_actions] After squeezing: {actions.shape}")
            
            if len(actions.shape) == 2:
                # Handle 2D actions (batch_size, action_dim)
                batch_size, action_dim = actions.shape
                model_batch_size = qpos.shape[0]
                num_queries = self.policy.model.num_queries
                
                print(f"[evaluate_actions] 2D actions detected - shape: {actions.shape}, model batch size: {model_batch_size}, num_queries: {num_queries}")
                
                # Check if batch_size matches model's expected batch size
                if batch_size != model_batch_size:
                    print(f"[evaluate_actions] WARNING: Batch size mismatch - actions: {batch_size}, model input: {model_batch_size}")
                    # Adjust actions tensor to match model's batch size
                    if batch_size < model_batch_size:
                        # Pad by repeating the last action
                        pad_size = model_batch_size - batch_size
                        padding = actions[-1:].repeat(pad_size, 1)
                        actions = torch.cat([actions, padding], dim=0)
                        print(f"[evaluate_actions] After padding: actions shape: {actions.shape}")
                    else:
                        # Truncate to match model's batch size
                        actions = actions[:model_batch_size]
                        print(f"[evaluate_actions] After truncating: actions shape: {actions.shape}")
                
                # Reshape to [batch_size, num_queries, action_dim] by repeating the action
                actions = actions.unsqueeze(1).repeat(1, num_queries, 1)
                print(f"[evaluate_actions] After reshaping to 3D: actions shape: {actions.shape}")
            elif len(actions.shape) == 3:
                # Actions are already 3D [batch_size, num_queries, action_dim]
                # Check if batch_size matches model's expected batch size
                batch_size, curr_queries, action_dim = actions.shape
                model_batch_size = qpos.shape[0]
                num_queries = self.policy.model.num_queries
                
                print(f"[evaluate_actions] 3D actions detected - shape: {actions.shape}, model batch size: {model_batch_size}, num_queries: {num_queries}")
                
                # Handle batch size mismatch
                if batch_size != model_batch_size:
                    print(f"[evaluate_actions] WARNING: Batch size mismatch - actions: {batch_size}, model input: {model_batch_size}")
                    if batch_size < model_batch_size:
                        # Pad by repeating the last batch
                        pad_size = model_batch_size - batch_size
                        padding = actions[-1:].repeat(pad_size, 1, 1)
                        actions = torch.cat([actions, padding], dim=0)
                        print(f"[evaluate_actions] After padding: actions shape: {actions.shape}")
                    else:
                        # Truncate to match model's batch size
                        actions = actions[:model_batch_size]
                        print(f"[evaluate_actions] After truncating: actions shape: {actions.shape}")
                
                # Handle query dimension mismatch
                if curr_queries != num_queries:
                    print(f"[evaluate_actions] WARNING: Query dimension mismatch - actions: {curr_queries}, model: {num_queries}")
                    if curr_queries < num_queries:
                        # Pad by repeating the last query
                        pad_size = num_queries - curr_queries
                        padding = actions[:, -1:].repeat(1, pad_size, 1)
                        actions = torch.cat([actions, padding], dim=1)
                        print(f"[evaluate_actions] After padding queries: actions shape: {actions.shape}")
                    else:
                        # Truncate to match model's num_queries
                        actions = actions[:, :num_queries]
                        print(f"[evaluate_actions] After truncating queries: actions shape: {actions.shape}")
            
            actions = actions.float()
            print(f"[evaluate_actions] Actions after float conversion: shape: {actions.shape}")
            
            if is_pad is None:
                # Create is_pad with correct shape [batch_size, num_queries]
                if len(actions.shape) == 3:  # [batch_size, num_queries, action_dim]
                    is_pad = torch.zeros((actions.shape[0], actions.shape[1]), dtype=torch.bool, device=actions.device)
                else:  # Handle unexpected shapes gracefully
                    is_pad = torch.zeros_like(actions[..., 0], dtype=torch.bool)
                print(f"[evaluate_actions] Created default is_pad with shape: {is_pad.shape}")
            else:
                print(f"[evaluate_actions] Processing provided is_pad with shape: {is_pad.shape}")
                if len(is_pad.shape) == 2:
                    # is_pad is 2D, slice appropriately
                    original_shape = is_pad.shape
                    is_pad = is_pad[:, :self.policy.model.num_queries]
                    print(f"[evaluate_actions] Sliced is_pad from {original_shape} to {is_pad.shape}")
                else:
                    # is_pad might be 1D, expand it
                    original_shape = is_pad.shape
                    is_pad = is_pad.unsqueeze(1).repeat(1, self.policy.model.num_queries)
                    print(f"[evaluate_actions] Expanded is_pad from {original_shape} to {is_pad.shape}")
            
            print(f"[evaluate_actions] Before model call - qpos: {qpos.shape}, image: {image.shape}, actions: {actions.shape}, is_pad: {is_pad.shape}")
            a_hat, is_pad_hat, (mu, logvar) = self.policy.model(qpos, image, env_state, actions, is_pad)
            print(f"[evaluate_actions] After model call - a_hat: {a_hat.shape}, mu: {mu.shape}, logvar: {logvar.shape}")
            
            # Check if mu and logvar are valid
            if mu is None or logvar is None:
                raise RuntimeError("Model forward pass returned None for mu or logvar in training mode")
            
            # Compute log probability using the shared utility function
            # This ensures consistency with Unity VR RL interface log probability calculation
            print(f"[evaluate_actions] Computing log probability with mu: {mu.shape}, logvar: {logvar.shape}")
            
            # Validate inputs
            is_valid, error_msg = validate_log_prob_inputs(a_hat, mu, logvar)
            if not is_valid:
                raise ValueError(f"Invalid inputs for log probability computation: {error_msg}")
            
            # Use the shared utility function with mean aggregation (consistent with Unity interface)
            action_log_probs = compute_gaussian_log_prob(
                actions=a_hat,  # Use model predictions (a_hat) for log probability
                mu=mu,
                logvar=logvar,
                aggregate_method="mean",  # Average across queries (consistent with Unity interface)
                numerical_stability=True
            )
            
            print(f"[evaluate_actions] Final log probs shape: {action_log_probs.shape}, range: [{action_log_probs.min():.4f}, {action_log_probs.max():.4f}]")
            
            # Debug log probability computation if values seem extreme
            if (action_log_probs < -50).any() or (action_log_probs > 10).any():
                print("[DEBUG] Extreme log probabilities detected, running debug analysis...")
                debug_log_prob_computation(a_hat, mu, logvar, "PPO_evaluate_actions")
            
            # Get value estimate
            # We'll use the latent representation from the model for value estimation
            # This is a simplification; in practice you might want a separate value network
            print(f"[evaluate_actions] Computing value with mu: {mu.shape}")
            value = self.value_net(mu).squeeze(-1)
            print(f"[evaluate_actions] Value shape: {value.shape}")
            
            # Compute entropy (approximation for continuous actions)
            print(f"[evaluate_actions] Computing entropy with logvar: {logvar.shape}")
            entropy = 0.5 * logvar.sum(dim=-1).mean()
            print(f"[evaluate_actions] Entropy: {entropy}")
            
            print(f"[evaluate_actions] Returning - action_log_probs: {action_log_probs.shape}, value: {value.shape}, mu: {mu.shape}, logvar: {logvar.shape}")
            return action_log_probs, value, entropy, mu, logvar
        else:
            # Inference mode - DETR VAE now always returns mu and logvar
            try:
                a_hat, _, (mu, logvar) = self.policy.model(qpos, image, env_state)
                
                # DETR VAE should now always return mu and logvar, even in inference mode
                if mu is None or logvar is None:
                    raise RuntimeError("DETR VAE returned None for mu or logvar - this should not happen with updated model!")
                
                # Compute value using the latent representation
                print(f"[evaluate_actions] Computing value with mu: {mu.shape}")
                value = self.value_net(mu).squeeze(-1)
                print(f"[evaluate_actions] Value shape: {value.shape}")
                return a_hat, value, mu, logvar
                
            except Exception as e:
                print(f"[PPO ERROR] Error in model forward pass: {e}")
                # Create normal distribution
                print(f"[evaluate_actions] Creating distribution with mu: {mu.shape}, std: {std.shape}")
                dist = torch.distributions.Normal(mu, std)
                
                # Create fallback values
                batch_size = qpos.shape[0]
                action_dim = getattr(self.policy.model, 'state_dim', 32)
                num_queries = getattr(self.policy.model, 'num_queries', 8)
                
                # Create dummy action and value
                dummy_action = torch.zeros(batch_size, num_queries, action_dim).cuda()
                dummy_value = torch.zeros(batch_size).cuda()
                
                print(f"[PPO ERROR] Using fallback action shape: {dummy_action.shape}, value shape: {dummy_value.shape}")
                return dummy_action, dummy_value
    
    def get_kl_divergence(self, qpos, image, actions=None, is_pad=None):
        """
        Compute KL divergence between current policy and pretrained policy.
        """
        env_state = None
        
        # Debug print for original shape
        print(f"[get_kl_divergence] Original image shape: {image.shape}")
        
        # Handle image tensor dimensions for conv2d compatibility
        # Handle 6D tensor [batch_size, dim1, dim2, channels, height, width]
        # This happens because we have a redundant dimension in the camera dimension
        if len(image.shape) == 6:
            batch_size, dim1, dim2, channels, height, width = image.shape
            # Squeeze out the redundant dimension (dim2 is usually 1)
            # This converts [batch_size, dim1, 1, channels, height, width] to [batch_size, dim1, channels, height, width]
            image = image.squeeze(2)
            print(f"[get_kl_divergence] Squeezed redundant dim: {image.shape}")
        
        # Handle actions tensor if provided
        if actions is not None:
            print(f"[get_kl_divergence] Actions shape: {actions.shape}")
            # Check if we have an extra dimension [batch_size, 1, num_queries, action_dim]
            if len(actions.shape) == 4 and actions.shape[1] == 1:
                print(f"[get_kl_divergence] Detected extra dimension in actions tensor, squeezing it")
                actions = actions.squeeze(1)  # Remove the extra dimension
                print(f"[get_kl_divergence] After squeezing: {actions.shape}")
            
            # Handle is_pad tensor if needed
            if is_pad is None:
                # Create is_pad with correct shape [batch_size, num_queries]
                if len(actions.shape) == 3:  # [batch_size, num_queries, action_dim]
                    is_pad = torch.zeros((actions.shape[0], actions.shape[1]), dtype=torch.bool, device=actions.device)
                else:  # Handle unexpected shapes gracefully
                    is_pad = torch.zeros_like(actions[..., 0], dtype=torch.bool)
                print(f"[get_kl_divergence] Created default is_pad with shape: {is_pad.shape}")
        
        # Normalize image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        print(f"[get_kl_divergence] Before model call - qpos: {qpos.shape}, image: {image.shape}")
        if actions is not None:
            print(f"[get_kl_divergence] actions: {actions.shape}, is_pad: {is_pad.shape}")
        
        # Forward pass through current policy
        _, _, (mu_current, logvar_current) = self.policy.model(qpos, image, env_state, actions, is_pad)
        
        # Forward pass through pretrained policy (no gradient)
        with torch.no_grad():
            _, _, (mu_pretrained, logvar_pretrained) = self.pretrained_policy.model(qpos, image, env_state, actions, is_pad)
        
        # Compute KL divergence between the two distributions
        # We'll use a simplified approach here
        # In practice, you might want to use a more sophisticated KL divergence calculation
        kl_div = 0.5 * (
            torch.exp(logvar_current - logvar_pretrained) +
            (mu_pretrained - mu_current).pow(2) / torch.exp(logvar_pretrained) -
            1 +
            logvar_pretrained - logvar_current
        ).sum(dim=-1).mean()
        
        return kl_div
    
    def update(self, memory, n_epochs=10, batch_size=64):
        """
        Update policy using PPO with KL constraint.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Compute advantages and returns
        memory.compute_advantages_and_returns()
        
        # Create optimizer for policy and value networks
        optimizer = optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=1e-4)
        
        # PPO update loop
        for epoch in range(n_epochs):
            # Get batches
            batches = memory.get_batches()
            
            # Process each batch
            for batch_indices in batches:
                # Get batch data
                states, images, actions, old_log_probs, returns, advantages = memory.get_batch_data(batch_indices)
                
                # Debug print for all tensor shapes before processing
                print(f"[PPO Update] Batch data shapes:")
                print(f"  - states: {states.shape}")
                print(f"  - images: {images.shape}, dtype: {images.dtype}")
                print(f"  - actions: {actions.shape}")
                print(f"  - old_log_probs: {old_log_probs.shape}")
                print(f"  - returns: {returns.shape}")
                print(f"  - advantages: {advantages.shape}")
                print(f"  - batch_indices length: {len(batch_indices)}")
                
                # Evaluate actions
                print(f"[PPO Update] Calling evaluate_actions with states: {states.shape}, images: {images.shape}, actions: {actions.shape}")
                action_log_probs, values, entropy, mu, logvar = self.evaluate_actions(states, images, actions)
                print(f"[PPO Update] After evaluate_actions - action_log_probs: {action_log_probs.shape}, values: {values.shape}")
                
                # Compute KL divergence to pretrained policy
                kl_div = self.get_kl_divergence(states, images, actions)
                
                # Compute ratio and clipped ratio
                print(f"[PPO Update] Computing ratio with action_log_probs: {action_log_probs.shape}, old_log_probs: {old_log_probs.shape}")
                
                # Add numerical stability checks for log probabilities
                if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
                    print(f"[WARNING] NaN or Inf detected in action_log_probs!")
                    action_log_probs = torch.where(torch.isnan(action_log_probs) | torch.isinf(action_log_probs), 
                                                  torch.tensor(-10.0, device=action_log_probs.device), 
                                                  action_log_probs)
                
                if torch.isnan(old_log_probs).any() or torch.isinf(old_log_probs).any():
                    print(f"[WARNING] NaN or Inf detected in old_log_probs!")
                    old_log_probs = torch.where(torch.isnan(old_log_probs) | torch.isinf(old_log_probs), 
                                               torch.tensor(-10.0, device=old_log_probs.device), 
                                               old_log_probs)
                
                # Clip log probability differences to prevent extreme ratios
                log_prob_diff = action_log_probs - old_log_probs
                log_prob_diff = torch.clamp(log_prob_diff, min=-20.0, max=20.0)  # Prevents extreme ratios
                
                ratio = torch.exp(log_prob_diff)
                
                # Additional safety check for ratio values
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print(f"[WARNING] NaN or Inf detected in ratio computation!")
                    ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio), 
                                       torch.tensor(1.0, device=ratio.device), 
                                       ratio)
                
                # Clip ratio to reasonable bounds to prevent extreme policy updates
                ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                
                print(f"[PPO Update] Ratio shape: {ratio.shape}, range: [{ratio.min():.6f}, {ratio.max():.6f}]")
                
                # Debug ratio computation if values seem extreme
                if (ratio > 10).any() or (ratio < 0.1).any():
                    print("[DEBUG] Extreme ratios detected, running debug analysis...")
                    analyze_ppo_ratio(old_log_probs, action_log_probs, "PPO_update")
                
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                
                # Check advantages for numerical stability
                if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                    print(f"[WARNING] NaN or Inf detected in advantages!")
                    advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages), 
                                           torch.tensor(0.0, device=advantages.device), 
                                           advantages)
                
                # Compute policy loss with additional stability checks
                policy_obj1 = ratio * advantages
                policy_obj2 = clipped_ratio * advantages
                
                # Check for NaN/Inf in policy objectives
                if torch.isnan(policy_obj1).any() or torch.isinf(policy_obj1).any():
                    print(f"[WARNING] NaN or Inf detected in policy_obj1!")
                    policy_obj1 = torch.where(torch.isnan(policy_obj1) | torch.isinf(policy_obj1), 
                                             torch.tensor(0.0, device=policy_obj1.device), 
                                             policy_obj1)
                
                if torch.isnan(policy_obj2).any() or torch.isinf(policy_obj2).any():
                    print(f"[WARNING] NaN or Inf detected in policy_obj2!")
                    policy_obj2 = torch.where(torch.isnan(policy_obj2) | torch.isinf(policy_obj2), 
                                             torch.tensor(0.0, device=policy_obj2.device), 
                                             policy_obj2)
                
                policy_loss = -torch.min(policy_obj1, policy_obj2).mean()
                
                temp_policy_loss = policy_loss

                # Final check and clipping for policy loss
                if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                    print(f"[WARNING] NaN or Inf detected in policy_loss! Setting to 0.0")
                    policy_loss = torch.tensor(0.0, device=policy_loss.device, requires_grad=True)
                else:
                    # Clip policy loss to prevent extreme values
                    policy_loss = torch.clamp(policy_loss, min=-1000.0, max=1000.0)
                
                print(f"[PPO Update] Policy loss: {policy_loss.item():.6f}")
                
                # Compute value loss with numerical stability
                if torch.isnan(values).any() or torch.isinf(values).any():
                    print(f"[WARNING] NaN or Inf detected in values!")
                    values = torch.where(torch.isnan(values) | torch.isinf(values), 
                                        torch.tensor(0.0, device=values.device), 
                                        values)
                
                if torch.isnan(returns).any() or torch.isinf(returns).any():
                    print(f"[WARNING] NaN or Inf detected in returns!")
                    returns = torch.where(torch.isnan(returns) | torch.isinf(returns), 
                                         torch.tensor(0.0, device=returns.device), 
                                         returns)
                
                value_loss = F.mse_loss(values, returns)
                
                # Diagnose training issues
                issues, suggestions = diagnose_ppo_training_issues(
                    policy_loss=temp_policy_loss,
                    value_loss=value_loss,
                    kl_div=kl_div,
                    entropy=entropy
                )
                
                # Clip value loss to prevent extreme values
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    print(f"[WARNING] NaN or Inf detected in value_loss! Setting to 0.0")
                    value_loss = torch.tensor(0.0, device=value_loss.device, requires_grad=True)
                else:
                    value_loss = torch.clamp(value_loss, min=0.0, max=1000.0)
                
                print(f"[PPO Update] Value loss: {value_loss.item():.6f}")
                
                # Compute total loss with numerical stability
                # Check entropy for numerical stability
                if torch.isnan(entropy) or torch.isinf(entropy):
                    print(f"[WARNING] NaN or Inf detected in entropy! Setting to 0.0")
                    entropy = torch.tensor(0.0, device=entropy.device, requires_grad=True)
                
                # Check KL divergence for numerical stability
                if torch.isnan(kl_div) or torch.isinf(kl_div):
                    print(f"[WARNING] NaN or Inf detected in KL divergence! Setting to 0.0")
                    kl_div = torch.tensor(0.0, device=kl_div.device, requires_grad=True)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.kl_weight * kl_div
                
                # Final check for total loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN or Inf detected in total loss! Setting to policy_loss only")
                    loss = policy_loss  # Fall back to just policy loss if total loss is unstable
                
                # Update networks
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Return statistics for logging
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item()
        }


class KLScheduler:
    """
    Scheduler for KL weight during training.
    """
    def __init__(self, initial_kl_weight=0.05, min_kl_weight=0.001, decay_factor=0.95, decay_steps=100):
        self.initial_kl_weight = initial_kl_weight
        self.min_kl_weight = min_kl_weight
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.current_kl_weight = initial_kl_weight
        self.step_count = 0
        
    def step(self):
        """Update KL weight based on schedule"""
        if self.step_count % self.decay_steps == 0 and self.step_count > 0:
            self.current_kl_weight = max(self.current_kl_weight * self.decay_factor, self.min_kl_weight)
        self.step_count += 1
        return self.current_kl_weight
    
    def get_kl_weight(self):
        """Get current KL weight"""
        return self.current_kl_weight
    
    def reset(self):
        """Reset scheduler"""
        self.current_kl_weight = self.initial_kl_weight
        self.step_count = 0
