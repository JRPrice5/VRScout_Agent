#!/usr/bin/env python
"""
Unity VR Reinforcement Learning Interface for ACT Policy Fine-tuning

This script extends the Unity VR interface to support reinforcement learning
data collection and fine-tuning of pre-trained ACT models using PPO with KL-to-BC constraint.
"""

import os
import sys
import time
import json
import socket
import threading
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
from collections import deque
import argparse

# Add the parent directory to path to import ACT modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ACT modules and Unity VR interface
from unity_vr_inference import UnityVRInterface
from policy import ACTPolicy
from rl.ppo_agent import PPOACTPolicy, PPOMemory, KLScheduler


class UnityVRRLInterface(UnityVRInterface):
    """
    Extended Unity VR interface for reinforcement learning training.
    Inherits from UnityVRInterface and adds RL-specific functionality.
    """
    
    def __init__(self, 
                 pretrained_ckpt_path,
                 rl_config=None,
                 **kwargs):
        """
        Initialize the RL interface.
        
        Args:
            pretrained_ckpt_path: Path to the pre-trained ACT model checkpoint
            rl_config: Dictionary containing RL-specific configuration
            **kwargs: Arguments passed to parent UnityVRInterface
        """
        # Initialize parent class
        super().__init__(**kwargs)
        
        # RL-specific configuration
        self.rl_config = rl_config or self._get_default_rl_config()
        
        # Load pretrained policy for KL constraint
        self.pretrained_policy = self._load_pretrained_policy(pretrained_ckpt_path)
        
        # Initialize PPO policy
        self.ppo_policy = PPOACTPolicy(
            pretrained_policy=self.pretrained_policy,
            kl_weight=self.rl_config['kl_weight'],
            clip_param=self.rl_config['clip_param'],
            value_coef=self.rl_config['value_coef'],
            entropy_coef=self.rl_config['entropy_coef']
        )
        
        # Initialize PPO memory buffer
        self.memory = PPOMemory(
            batch_size=self.rl_config['batch_size'],
            max_eps_len=self.rl_config['max_episode_length'],
            gamma=self.rl_config['gamma'],
            gae_lambda=self.rl_config['gae_lambda']
        )
        
        # Initialize KL scheduler
        self.kl_scheduler = KLScheduler(
            initial_kl_weight=self.rl_config['kl_weight'],
            min_kl_weight=self.rl_config['min_kl_weight'],
            decay_factor=self.rl_config['kl_decay_factor'],
            decay_steps=self.rl_config['kl_decay_steps']
        )
        
        # Training state
        self.is_training = False
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = []
        
        # Episode tracking
        self.episode_states = []
        self.episode_images = []
        self.episode_actions = []
        self.episode_values = []
        self.episode_log_probs = []
        self.episode_timesteps = []
        self.episode_start_time = None
        
        # Replace the policy with PPO policy for training
        if self.is_training:
            self.policy = self.ppo_policy
        
        print(f"RL Interface initialized with config: {self.rl_config}")
        
    def _get_default_rl_config(self):
        """Get default RL configuration"""
        return {
            'kl_weight': 0.05,
            'min_kl_weight': 0.001,
            'kl_decay_factor': 0.95,
            'kl_decay_steps': 100,
            'clip_param': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'batch_size': 64,
            'max_episode_length': 1000,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'n_epochs': 10,
            'update_frequency': 2048,  # Update after collecting this many steps
            'max_episodes': 1000,
            'save_frequency': 100,  # Save model every N episodes
        }
    
    def _load_pretrained_policy(self, ckpt_path):
        """Load the pretrained policy for KL constraint"""
        print(f"Loading pretrained policy from {ckpt_path}")
        
        # CRITICAL FIX: Load the checkpoint first to get the actual model state
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Try to extract configuration from checkpoint if available
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            print(f"Found saved config in checkpoint: {saved_config}")
            policy_config = saved_config
        else:
            # Fallback: Use interface configuration but ensure all required fields are present
            policy_config = {
                'num_queries': self.chunk_size,
                'kl_weight': self.kl_weight,
                'hidden_dim': self.hidden_dim,
                'dim_feedforward': self.dim_feedforward,
                'lr': self.lr,
                'lr_backbone': self.lr_backbone,
                'backbone': self.backbone,
                'enc_layers': self.enc_layers,
                'dec_layers': self.dec_layers,
                'nheads': self.nheads,
                'chunk_size': self.chunk_size,
                'ckpt_dir': self.ckpt_dir,
                'policy_class': self.policy_class,
                'task_name': self.task_name,
                'seed': self.seed,
                'num_epochs': self.num_epochs,
                'camera_names': self.camera_names,
                # Add missing required parameters with defaults
                'dropout': 0.1,
                'pre_norm': False,
                'weight_decay': 1e-4,
                'state_dim': 32,  # Default robot state dimension
                'latent_dim': 32,  # Default latent dimension
            }
            print(f"Using interface config with defaults: {policy_config}")
        
        # Create policy with the configuration
        pretrained_policy = ACTPolicy(policy_config)
        
        # Load the state dict
        if 'model_state_dict' in checkpoint:
            pretrained_policy.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model_state_dict from checkpoint")
        elif 'state_dict' in checkpoint:
            pretrained_policy.load_state_dict(checkpoint['state_dict'])
            print("Loaded state_dict from checkpoint")
        else:
            # Assume the entire checkpoint is the state dict
            pretrained_policy.load_state_dict(checkpoint)
            print("Loaded checkpoint as state_dict")
        
        pretrained_policy.cuda()
        pretrained_policy.eval()
        
        # Print model info for debugging
        print(f"Loaded pretrained policy:")
        print(f"  Model type: {type(pretrained_policy.model)}")
        print(f"  Camera names: {getattr(pretrained_policy.model, 'camera_names', 'NOT SET')}")
        print(f"  Num queries: {getattr(pretrained_policy.model, 'num_queries', 'NOT SET')}")
        print(f"  Hidden dim: {getattr(pretrained_policy.model.transformer, 'd_model', 'NOT SET')}")
        
        return pretrained_policy
    
    def start_rl_training(self):
        """Start reinforcement learning training mode"""
        self.is_training = True
        self.policy = self.ppo_policy
        print("Started RL training mode")
        
    def stop_rl_training(self):
        """Stop reinforcement learning training mode"""
        self.is_training = False
        self.policy = self.pretrained_policy
        print("Stopped RL training mode")
        
    def start_new_episode(self):
        """Start a new episode"""
        self.episode_start_time = time.time()
        self.episode_states = []
        self.episode_images = []
        self.episode_actions = []
        self.episode_values = []
        self.episode_log_probs = []
        self.episode_timesteps = []
        
        # Mark episode start in memory
        self.memory.mark_episode_start(self.timestep)
        
        print(f"Started episode {self.current_episode}")
        
    def _process_state_and_get_action(self, state_data):
        """
        Override parent method to support RL data collection.
        """
        # If not in training mode, use parent implementation
        if not self.is_training:
            return super()._process_state_and_get_action(state_data)
        
        with torch.no_grad():
            # Extract qpos from state data
            qpos_numpy = np.array(state_data['qpos'], dtype=np.float32)
            print(f"[RL DEBUG] Input qpos_numpy shape: {qpos_numpy.shape}, dtype: {qpos_numpy.dtype}")
            print(f"[RL DEBUG] Input qpos values (first 10): {qpos_numpy[:10]}")
            
            # Preprocess qpos
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            print(f"[RL DEBUG] After preprocessing qpos shape: {qpos.shape}")
            
            # Process image data
            curr_image = self._process_image_data(state_data)
            
            # DEBUG: Print tensor shapes and model configuration
            print(f"[RL DEBUG] qpos shape: {qpos.shape}")
            print(f"[RL DEBUG] curr_image shape: {curr_image.shape}")
            print(f"[RL DEBUG] self.camera_names: {self.camera_names}")
            print(f"[RL DEBUG] model.camera_names: {getattr(self.ppo_policy.policy.model, 'camera_names', 'NOT SET')}")
            
            # CRITICAL FIX: Ensure the PPO policy model has the correct camera_names
            if hasattr(self.ppo_policy.policy.model, 'camera_names'):
                if self.ppo_policy.policy.model.camera_names != self.camera_names:
                    print(f"[RL DEBUG] Fixing model camera_names: {self.ppo_policy.policy.model.camera_names} -> {self.camera_names}")
                    self.ppo_policy.policy.model.camera_names = self.camera_names
            else:
                print(f"[RL DEBUG] Setting model camera_names to: {self.camera_names}")
                self.ppo_policy.policy.model.camera_names = self.camera_names
            
            # Store state and image for episode
            self.episode_states.append(qpos_numpy)
            self.episode_images.append(curr_image)
            self.episode_timesteps.append(self.timestep)
            
            # Get action and value from PPO policy
            if self.policy_class == 'ACT':
                try:
                    # Get action from policy
                    action_tensor, value = self.ppo_policy.evaluate_actions(qpos, curr_image)
                    print(f"[RL DEBUG] Raw action_tensor shape: {action_tensor.shape}, dtype: {action_tensor.dtype}")
                    print(f"[RL DEBUG] Value: {value}")
                    
                    # Handle like parent class
                    all_actions = action_tensor  # Shape: [batch, num_queries, action_dim]
                    time_index = 0  # Always use first timestep when no temporal agg
                    raw_action = all_actions[:, time_index]  # Shape: [batch, action_dim]
                    
                    # Keep as tensor for log probability computation
                    raw_action = raw_action.squeeze(0)  # Shape: [action_dim]
                    
                    # Store value for episode
                    self.episode_values.append(value.item())
                    
                    # Compute log probability using the policy's latent distribution
                    # We need to get the latent distribution parameters (mu, logvar) from the policy
                    with torch.no_grad():
                        # Get the latent distribution parameters from the policy
                        env_state = None
                        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                        normalized_image = normalize(curr_image)
                        
                        # Forward pass to get latent distribution
                        _, _, (mu, logvar) = self.ppo_policy.policy.model(qpos, normalized_image, env_state)
                        
                        # In inference mode, mu and logvar are None
                        if mu is None or logvar is None:
                            # For inference mode, we can't compute exact log probabilities
                            # Use a simplified approximation based on action magnitude
                            action_flat = raw_action
                            # Simple approximation: negative squared magnitude as log prob
                            log_prob = -0.5 * torch.sum(action_flat**2).item()
                            print(f"[RL DEBUG] Using approximated log_prob in inference mode: {log_prob}")
                        else:
                            # Training mode - compute proper log probability
                            # For continuous actions, we assume a Gaussian distribution
                            std = torch.exp(0.5 * logvar)
                            
                            # Convert action back to latent space for log prob computation
                            # This is an approximation - in practice you might want to store the original latent sample
                            action_flat = raw_action
                            
                            # Compute log probability under Gaussian distribution
                            # log_prob = -0.5 * ((action - mu)^2 / std^2 + log(2*pi*std^2))
                            log_prob_tensor = -0.5 * (((action_flat.unsqueeze(0) - mu)**2 / (std**2 + 1e-8)) + 
                                                     torch.log(2 * np.pi * std**2 + 1e-8))
                            log_prob = log_prob_tensor.sum().item()  # Sum over action dimensions
                        
                    self.episode_log_probs.append(log_prob)
                    
                    # Store action for episode - convert to CPU tensor for storage
                    self.episode_actions.append(raw_action.detach().cpu())
                    
                except Exception as e:
                    print(f"[RL ERROR] Error in PPO policy evaluation: {e}")
                    print(f"[RL ERROR] Falling back to parent implementation")
                    return super()._process_state_and_get_action(state_data)
                
            else:
                # For CNNMLP, use parent implementation
                return super()._process_state_and_get_action(state_data)
            
            # Convert tensor to numpy for post-processing
            raw_action_np = raw_action.detach().cpu().numpy()
            
            # Post-process action
            action = self.post_process(raw_action_np)
            
            # Convert to list for JSON serialization
            action_list = action.tolist()
            
            return {
                'action': action_list,
                'timestep': self.timestep
            }
    
    def process_episode_rewards(self, episode_data):
        """
        Process rewards received from Unity client after episode completion.
        
        Args:
            episode_data: Dictionary containing:
                - 'timesteps': List of timesteps
                - 'rewards': List of rewards corresponding to timesteps
                - 'done': Whether episode is complete
        """
        timesteps = episode_data.get('timesteps', [])
        rewards = episode_data.get('rewards', [])
        done = episode_data.get('done', True)
        
        print(f"Received episode data: {len(timesteps)} timesteps, {len(rewards)} rewards")
        
        # Match rewards to stored transitions
        self.memory.match_rewards(timesteps, rewards)
        
        # Store transitions in memory
        for i, (state, image, action, value, log_prob, timestep) in enumerate(
            zip(self.episode_states, self.episode_images, self.episode_actions, 
                self.episode_values, self.episode_log_probs, self.episode_timesteps)):
            
            # Get reward for this timestep (default to 0 if not provided)
            reward = 0.0
            if timestep in dict(zip(timesteps, rewards)):
                reward = dict(zip(timesteps, rewards))[timestep]
            
            # Determine if this is the last step
            is_done = (i == len(self.episode_states) - 1) and done
            
            # Store in memory
            self.memory.store(
                state=state,
                image=image,
                action=action,
                reward=reward,
                done=is_done,
                value=value,
                action_log_prob=log_prob,
                timestep=timestep
            )
        
        # Update episode statistics
        total_reward = sum(rewards) if rewards else 0.0
        episode_length = len(self.episode_states)
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.current_episode += 1
        self.total_episodes += 1
        
        print(f"Episode {self.current_episode} completed: reward={total_reward:.2f}, length={episode_length}, is_training={self.is_training}")
        
        # Check if we should update the policy
        if self.is_training and len(self.memory.states) >= self.rl_config['update_frequency']:
            self.update_policy()
        
        # Check if we should save the model
        if self.is_training and self.current_episode % self.rl_config['save_frequency'] == 0:
            self.save_model()
        
        return {
            'episode': self.current_episode,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'average_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        }
    
    def update_policy(self):
        """Update the PPO policy"""
        print("Updating PPO policy...")
        
        # Update KL weight based on schedule
        new_kl_weight = self.kl_scheduler.step()
        self.ppo_policy.kl_weight = new_kl_weight
        
        # Perform PPO update
        stats = self.ppo_policy.update(
            memory=self.memory,
            n_epochs=self.rl_config['n_epochs'],
            batch_size=self.rl_config['batch_size']
        )
        
        # Add KL weight to stats
        stats['kl_weight'] = new_kl_weight
        stats['episode'] = self.current_episode
        stats['total_steps'] = len(self.memory.states)
        
        # Store training statistics
        self.training_stats.append(stats)
        
        # Clear memory after update
        self.memory.clear()
        
        print(f"Policy updated: {stats}")
        
        return stats
    
    def save_model(self, path=None):
        """Save the current PPO model"""
        if path is None:
            path = os.path.join(self.ckpt_dir, f'ppo_policy_episode_{self.current_episode}.ckpt')
        
        # Save PPO policy state
        torch.save({
            'policy_state_dict': self.ppo_policy.policy.state_dict(),
            'value_net_state_dict': self.ppo_policy.value_net.state_dict(),
            'episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_stats': self.training_stats,
            'rl_config': self.rl_config,
            'kl_scheduler_state': {
                'current_kl_weight': self.kl_scheduler.current_kl_weight,
                'step_count': self.kl_scheduler.step_count
            }
        }, path)
        
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load a saved PPO model"""
        checkpoint = torch.load(path)
        
        # Load policy and value network states
        self.ppo_policy.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_policy.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        
        # Load training state
        self.current_episode = checkpoint.get('episode', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.training_stats = checkpoint.get('training_stats', [])
        
        # Load KL scheduler state
        kl_scheduler_state = checkpoint.get('kl_scheduler_state', {})
        self.kl_scheduler.current_kl_weight = kl_scheduler_state.get('current_kl_weight', self.rl_config['kl_weight'])
        self.kl_scheduler.step_count = kl_scheduler_state.get('step_count', 0)
        
        print(f"Model loaded from {path}")
        
    def get_training_stats(self):
        """Get current training statistics"""
        return {
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'average_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'average_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0,
            'current_kl_weight': self.kl_scheduler.get_kl_weight(),
            'total_training_steps': sum(len(stats.get('policy_loss', [])) for stats in self.training_stats),
            'memory_size': len(self.memory.states)
        }
    
    def _handle_client(self):
        """
        Override parent method to handle RL-specific messages.
        """
        # Use the client socket that was already accepted by the parent class
        client_socket = self.client_socket
        
        try:
            # Reset timestep counter when a new client connects
            self.timestep = 0
            
            # Set timeout on client socket for clean shutdown
            client_socket.settimeout(0.5)
            print("Client connected and ready for RL communication")
            
            # Initialize data buffer for message handling (like parent class)
            buffer_size = 4096
            data_buffer = b""
            
            try:
                while self.running:
                    try:
                        # Receive data from client with timeout
                        chunk = client_socket.recv(buffer_size)
                        if not chunk:
                            print("Unity client disconnected")
                            # Clear memory after client disconnect
                            self.memory.clear()
                            break
                        
                        data_buffer += chunk
                        
                        # Check if we have a complete message (Unity client uses END_OF_MESSAGE delimiter)
                        if b"END_OF_MESSAGE" in data_buffer:
                            # Extract the complete message
                            message_end = data_buffer.find(b"END_OF_MESSAGE")
                            message_data = data_buffer[:message_end]
                            data_buffer = data_buffer[message_end + len(b"END_OF_MESSAGE"):]
                            
                            try:
                                # Parse JSON data
                                message = json.loads(message_data.decode('utf-8'))
                                
                                # Detect message format: check if it has 'type' field (RL format) or direct state data (Unity format)
                                if 'type' in message:
                                    # RL-specific message format: {"type": "...", "data": {...}}
                                    message_type = message.get('type', 'state')
                                    
                                    if message_type == 'episode_data':
                                        # Handle episode completion with rewards
                                        episode_data = message.get('data', {})
                                        
                                        # Process episode rewards and update policy
                                        episode_stats = self.process_episode_rewards(episode_data)
                                        
                                        # Send episode statistics back to client
                                        response = {
                                            'type': 'episode_response',
                                            'data': episode_stats
                                        }
                                        response_json = json.dumps(response) + "END_OF_MESSAGE"
                                        client_socket.sendall(response_json.encode('utf-8'))
                                        
                                        # Start new episode if training continues
                                        if self.is_training and self.current_episode < self.rl_config['max_episodes']:
                                            self.start_new_episode()
                                        
                                    elif message_type == 'training_stats':
                                        # Handle request for training statistics
                                        stats = self.get_training_stats()
                                        response = {
                                            'type': 'stats_response',
                                            'data': stats
                                        }
                                        response_json = json.dumps(response) + "END_OF_MESSAGE"
                                        client_socket.sendall(response_json.encode('utf-8'))
                                        
                                    elif message_type == 'start_training':
                                        # Handle start training command
                                        self.start_rl_training()
                                        self.start_new_episode()
                                        response = {
                                            'type': 'training_started',
                                            'data': {'status': 'Training started'}
                                        }
                                        response_json = json.dumps(response) + "END_OF_MESSAGE"
                                        client_socket.sendall(response_json.encode('utf-8'))
                                        
                                    elif message_type == 'stop_training':
                                        # Handle stop training command
                                        self.stop_rl_training()
                                        response = {
                                            'type': 'training_stopped',
                                            'data': {'status': 'Training stopped'}
                                        }
                                        response_json = json.dumps(response) + "END_OF_MESSAGE"
                                        client_socket.sendall(response_json.encode('utf-8'))
                                        
                                    else:
                                        # Default to state handling for RL format
                                        state_data = message.get('data', {})
                                        
                                        # Update timestep
                                        self.timestep = state_data.get('timestep', self.timestep + 1)
                                        
                                        # Process state and get action
                                        response = self._process_state_and_get_action(state_data)
                                        
                                        # Send response back to client
                                        response_json = json.dumps(response) + "END_OF_MESSAGE"
                                        client_socket.sendall(response_json.encode('utf-8'))
                                        
                                else:
                                    # Unity client direct format: {"qpos": [...], "image": "...", "timestep": 123}
                                    # This is the standard format from InferencePoseClient.cs
                                    state_data = message
                                    
                                    # Update timestep
                                    if 'timestep' in state_data:
                                        self.timestep = state_data['timestep']
                                    
                                    # Process state and get action (same as parent class)
                                    action = self._process_state_and_get_action(state_data)
                                    
                                    # Send action back to Unity (same format as parent class)
                                    response = json.dumps(action) + "END_OF_MESSAGE"
                                    client_socket.sendall(response.encode('utf-8'))
                                    
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON: {e}")
                                error_response = json.dumps({'error': 'Invalid JSON format'}) + "END_OF_MESSAGE"
                                client_socket.sendall(error_response.encode('utf-8'))
                            except Exception as e:
                                print(f"Error processing request: {e}")
                                error_response = json.dumps({'error': str(e)}) + "END_OF_MESSAGE"
                                client_socket.sendall(error_response.encode('utf-8'))
                            
                    except socket.timeout:
                        # This is expected with the timeout, just continue
                        continue
                    except ConnectionResetError:
                        print("Connection was reset by the client")
                        break
                    except ConnectionAbortedError:
                        print("Connection was aborted")
                        break
                    except Exception as e:
                        print(f"Error receiving data: {e}")
                        break
                        
            except Exception as e:
                print(f"Client communication error: {e}")
            finally:
                print("Closing client connection")
                try:
                    client_socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass  # Socket might already be closed
                client_socket.close()
        
        except Exception as e:
            print(f"Error handling client: {e}")


def main():
    parser = argparse.ArgumentParser(description='Unity VR RL Interface for ACT Policy Fine-tuning')
    parser.add_argument('--pretrained_ckpt_path', type=str, required=True, help='Path to pretrained ACT model checkpoint')
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=5557, help='Server port (different from inference port)')
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['camera1'], help='Camera names')
    
    # Add required parameters for DETR module
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--task_name', type=str, default='rl_task', help='Task name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--kl_weight', type=float, default=0.05, help='Initial KL Weight for RL')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for ACT policy')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for transformer')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension for transformer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model')
    parser.add_argument('--enc_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='Number of decoder layers')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--query_frequency', type=int, default=1, help='How often to query the policy')
    
    # RL-specific arguments
    parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum episode length')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum number of training episodes')
    parser.add_argument('--update_frequency', type=int, default=2048, help='Update policy every N steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PPO updates')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--min_kl_weight', type=float, default=0.001, help='Minimum KL weight')
    parser.add_argument('--kl_decay_factor', type=float, default=0.95, help='KL weight decay factor')
    parser.add_argument('--kl_decay_steps', type=int, default=100, help='KL weight decay steps')
    
    args = parser.parse_args()
    
    # Create RL configuration
    rl_config = {
        'kl_weight': args.kl_weight,
        'min_kl_weight': args.min_kl_weight,
        'kl_decay_factor': args.kl_decay_factor,
        'kl_decay_steps': args.kl_decay_steps,
        'clip_param': args.clip_param,
        'value_coef': args.value_coef,
        'entropy_coef': args.entropy_coef,
        'batch_size': args.batch_size,
        'max_episode_length': args.max_episode_length,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'n_epochs': args.n_epochs,
        'update_frequency': args.update_frequency,
        'max_episodes': args.max_episodes,
        'save_frequency': 100,
    }
    
    # Create RL interface
    interface = UnityVRRLInterface(
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        rl_config=rl_config,
        policy_class=args.policy_class,
        host=args.host,
        port=args.port,
        camera_names=args.camera_names,
        temporal_agg=args.temporal_agg,
        ckpt_dir=args.ckpt_dir,
        task_name=args.task_name,
        seed=args.seed,
        num_epochs=args.num_epochs,
        kl_weight=args.kl_weight,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        backbone=args.backbone,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        nheads=args.nheads,
        query_frequency=args.query_frequency
    )
    
    # Start the server
    interface.start_server()


if __name__ == "__main__":
    main()
