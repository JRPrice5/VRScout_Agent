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
                is_pad = torch.zeros_like(actions[:, :, 0], dtype=torch.bool)
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
            
            # Compute log probability of actions using the latent Gaussian distribution
            print(f"[evaluate_actions] Computing log probability with mu: {mu.shape}, logvar: {logvar.shape}")
            std = torch.exp(0.5 * logvar)
            
            # For proper PPO, we need to compute the log probability of the actual actions
            # under the current policy's distribution. Since ACT uses a latent space,
            # we approximate this by computing the log prob in the latent space
            
            # Create normal distribution
            print(f"[evaluate_actions] Creating distribution with mu: {mu.shape}, std: {std.shape}")
            dist = torch.distributions.Normal(mu, std)
            
            # Compute log probability for all queries and average them
            print(f"[evaluate_actions] Computing log_prob with actions: {actions.shape}")
            
            # The issue is that actions is [batch_size, num_queries, action_dim] but
            # distribution is [batch_size, action_dim]
            # We'll compute log probabilities for each query and average them
            
            # Create a list to store log probabilities for each query
            query_log_probs = []
            
            # For each query position, compute log probability
            for q in range(actions.shape[1]):
                query_actions = actions[:, q, :]
                query_a_hat = a_hat[:, q, :]
                print(f"[evaluate_actions] Computing log_prob for query {q}, actions shape: {query_actions.shape}")
                
                # Compute log probability using the Gaussian formula
                log_prob_per_dim = -0.5 * (((query_a_hat - mu)**2 / (std**2 + 1e-8)) + 
                                          torch.log(2 * np.pi * std**2 + 1e-8))
                query_log_prob = log_prob_per_dim.sum(dim=-1)
                query_log_probs.append(query_log_prob)
            
            # Stack and average log probabilities across queries
            stacked_log_probs = torch.stack(query_log_probs, dim=1)  # [batch_size, num_queries]
            print(f"[evaluate_actions] Stacked log probs shape: {stacked_log_probs.shape}")
            
            # Average log probabilities across queries
            action_log_probs = stacked_log_probs.mean(dim=1)  # [batch_size]
            print(f"[evaluate_actions] Final log probs shape: {action_log_probs.shape}")
            
            # Check if we have a sequence dimension to mean over
            if len(action_log_probs.shape) > 1 and action_log_probs.shape[-1] > 1:
                action_log_probs = action_log_probs.mean(dim=-1)
            # Otherwise, we already have the correct shape
            
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
            # Inference mode
            try:
                a_hat, _, (mu, logvar) = self.policy.model(qpos, image, env_state)
                
                # In inference mode, DETRVAE returns None for mu and logvar
                # We need to create a latent representation for value estimation
                if mu is None or logvar is None:
                    # For inference, we need to extract features from the model to create a latent representation
                    # We'll use the transformer's output features as a proxy for the latent representation
                    batch_size = qpos.shape[0]
                    latent_dim = getattr(self.policy.model, 'latent_dim', 32)
                    
                    # Create a zero latent sample (as done in DETRVAE inference mode)
                    latent_sample = torch.zeros([batch_size, latent_dim], dtype=torch.float32).cuda()
                    
                    # Use this as input to value network
                    mu = latent_sample
                    print(f"[PPO DEBUG] Created inference latent with shape: {mu.shape}")
                
                # Compute value
                print(f"[evaluate_actions] Computing value with mu: {mu.shape}")
                value = self.value_net(mu).squeeze(-1)
                print(f"[evaluate_actions] Value shape: {value.shape}")
                return a_hat, value
                
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
                ratio = torch.exp(action_log_probs - old_log_probs)
                print(f"[PPO Update] Ratio shape: {ratio.shape}")
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                
                # Compute policy loss
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, returns)
                
                # Compute total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.kl_weight * kl_div
                
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
