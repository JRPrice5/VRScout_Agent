#!/usr/bin/env python
"""
Training utilities for ACT RL fine-tuning.

This module provides utilities for logging, monitoring, and managing
the reinforcement learning training process.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import pickle
from datetime import datetime


class TrainingLogger:
    """
    Logger for RL training metrics and statistics.
    """
    
    def __init__(self, log_dir, experiment_name=None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"rl_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize logs
        self.episode_logs = []
        self.training_logs = []
        self.system_logs = []
        
        # Initialize metrics tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.policy_losses = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        self.kl_divergences = deque(maxlen=1000)
        self.entropies = deque(maxlen=1000)
        self.kl_weights = deque(maxlen=1000)
        
        print(f"Training logger initialized: {self.experiment_dir}")
        
    def log_episode(self, episode, reward, length, timestep):
        """Log episode statistics"""
        episode_data = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'timestep': timestep,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        self.episode_logs.append(episode_data)
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Save episode log
        self._save_logs()
        
    def log_training_step(self, episode, stats):
        """Log training step statistics"""
        training_data = {
            'episode': episode,
            'policy_loss': stats.get('policy_loss', 0.0),
            'value_loss': stats.get('value_loss', 0.0),
            'entropy': stats.get('entropy', 0.0),
            'kl_div': stats.get('kl_div', 0.0),
            'kl_weight': stats.get('kl_weight', 0.0),
            'total_steps': stats.get('total_steps', 0),
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        self.training_logs.append(training_data)
        self.policy_losses.append(stats.get('policy_loss', 0.0))
        self.value_losses.append(stats.get('value_loss', 0.0))
        self.kl_divergences.append(stats.get('kl_div', 0.0))
        self.entropies.append(stats.get('entropy', 0.0))
        self.kl_weights.append(stats.get('kl_weight', 0.0))
        
        # Save training log
        self._save_logs()
        
    def log_system_event(self, event_type, message, data=None):
        """Log system events"""
        system_data = {
            'event_type': event_type,
            'message': message,
            'data': data,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        self.system_logs.append(system_data)
        print(f"[{event_type}] {message}")
        
        # Save system log
        self._save_logs()
        
    def _save_logs(self):
        """Save logs to files"""
        # Save episode logs
        with open(os.path.join(self.experiment_dir, 'episode_logs.json'), 'w') as f:
            json.dump(self.episode_logs, f, indent=2)
            
        # Save training logs
        with open(os.path.join(self.experiment_dir, 'training_logs.json'), 'w') as f:
            json.dump(self.training_logs, f, indent=2)
            
        # Save system logs
        with open(os.path.join(self.experiment_dir, 'system_logs.json'), 'w') as f:
            json.dump(self.system_logs, f, indent=2)
            
    def plot_training_curves(self, save_plot=True, show_plot=False):
        """Plot training curves"""
        if len(self.episode_rewards) < 2:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'RL Training Progress - {self.experiment_name}')
        
        # Episode rewards
        episodes = range(len(self.episode_rewards))
        axes[0, 0].plot(episodes, list(self.episode_rewards))
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Moving average of rewards
        if len(self.episode_rewards) > 10:
            window_size = min(50, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(list(self.episode_rewards), 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(range(window_size-1, len(self.episode_rewards)), moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[0, 2].plot(episodes, list(self.episode_lengths))
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Length')
        axes[0, 2].grid(True)
        
        # Policy loss
        if len(self.policy_losses) > 0:
            training_steps = range(len(self.policy_losses))
            axes[1, 0].plot(training_steps, list(self.policy_losses))
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # KL divergence
        if len(self.kl_divergences) > 0:
            axes[1, 1].plot(training_steps, list(self.kl_divergences))
            axes[1, 1].set_title('KL Divergence')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('KL Div')
            axes[1, 1].grid(True)
        
        # KL weight schedule
        if len(self.kl_weights) > 0:
            axes[1, 2].plot(training_steps, list(self.kl_weights))
            axes[1, 2].set_title('KL Weight Schedule')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('KL Weight')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.experiment_dir, 'plots', 'training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {plot_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    def get_summary_stats(self):
        """Get summary statistics"""
        if len(self.episode_rewards) == 0:
            return {}
            
        return {
            'total_episodes': len(self.episode_rewards),
            'total_training_steps': len(self.policy_losses),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'recent_mean_reward': np.mean(list(self.episode_rewards)[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'recent_mean_length': np.mean(list(self.episode_lengths)[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths),
            'current_kl_weight': list(self.kl_weights)[-1] if len(self.kl_weights) > 0 else 0.0,
            'mean_policy_loss': np.mean(self.policy_losses) if len(self.policy_losses) > 0 else 0.0,
            'mean_kl_div': np.mean(self.kl_divergences) if len(self.kl_divergences) > 0 else 0.0
        }
        
    def save_checkpoint(self, model_state, episode, additional_data=None):
        """Save training checkpoint"""
        checkpoint_data = {
            'model_state': model_state,
            'episode': episode,
            'episode_logs': self.episode_logs,
            'training_logs': self.training_logs,
            'summary_stats': self.get_summary_stats(),
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
            
        checkpoint_path = os.path.join(self.experiment_dir, 'checkpoints', f'checkpoint_episode_{episode}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path


class MemoryMonitor:
    """
    Monitor memory usage during training to ensure memory efficiency.
    """
    
    def __init__(self, max_memory_mb=8000):
        self.max_memory_mb = max_memory_mb
        self.memory_history = deque(maxlen=1000)
        
    def check_memory(self):
        """Check current memory usage"""
        if torch.cuda.is_available():
            # GPU memory
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            
            self.memory_history.append({
                'gpu_allocated': gpu_memory,
                'gpu_cached': gpu_memory_cached,
                'timestamp': time.time()
            })
            
            # Check if memory usage is too high
            if gpu_memory > self.max_memory_mb:
                print(f"Warning: GPU memory usage high: {gpu_memory:.1f} MB")
                return False
                
            return True
        return True
        
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")
            
    def get_memory_stats(self):
        """Get memory statistics"""
        if not self.memory_history:
            return {}
            
        recent_memory = list(self.memory_history)[-10:]  # Last 10 measurements
        
        return {
            'current_gpu_allocated': recent_memory[-1]['gpu_allocated'],
            'current_gpu_cached': recent_memory[-1]['gpu_cached'],
            'avg_gpu_allocated': np.mean([m['gpu_allocated'] for m in recent_memory]),
            'max_gpu_allocated': np.max([m['gpu_allocated'] for m in recent_memory]),
            'memory_efficiency': (recent_memory[-1]['gpu_allocated'] / self.max_memory_mb) * 100
        }


class EarlyStoppingMonitor:
    """
    Monitor training progress and implement early stopping if needed.
    """
    
    def __init__(self, patience=50, min_delta=0.01, monitor='reward'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('-inf') if monitor == 'reward' else float('inf')
        self.wait = 0
        self.stopped_episode = 0
        
    def check(self, current_value, episode):
        """Check if training should be stopped early"""
        if self.monitor == 'reward':
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
            
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_episode = episode
            return True
            
        return False
        
    def get_status(self):
        """Get early stopping status"""
        return {
            'best_value': self.best_value,
            'wait': self.wait,
            'patience': self.patience,
            'should_stop': self.wait >= self.patience
        }


def create_training_config(args):
    """Create training configuration from command line arguments"""
    return {
        'experiment_name': getattr(args, 'experiment_name', None),
        'max_episodes': getattr(args, 'max_episodes', 1000),
        'update_frequency': getattr(args, 'update_frequency', 2048),
        'batch_size': getattr(args, 'batch_size', 64),
        'n_epochs': getattr(args, 'n_epochs', 10),
        'gamma': getattr(args, 'gamma', 0.99),
        'gae_lambda': getattr(args, 'gae_lambda', 0.95),
        'clip_param': getattr(args, 'clip_param', 0.2),
        'value_coef': getattr(args, 'value_coef', 0.5),
        'entropy_coef': getattr(args, 'entropy_coef', 0.01),
        'kl_weight': getattr(args, 'kl_weight', 0.05),
        'min_kl_weight': getattr(args, 'min_kl_weight', 0.001),
        'kl_decay_factor': getattr(args, 'kl_decay_factor', 0.95),
        'kl_decay_steps': getattr(args, 'kl_decay_steps', 100),
        'save_frequency': getattr(args, 'save_frequency', 100),
        'plot_frequency': getattr(args, 'plot_frequency', 50),
        'max_memory_mb': getattr(args, 'max_memory_mb', 8000),
        'early_stopping_patience': getattr(args, 'early_stopping_patience', 50),
        'early_stopping_min_delta': getattr(args, 'early_stopping_min_delta', 0.01)
    }


def load_training_checkpoint(checkpoint_path):
    """Load training checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def compare_experiments(experiment_dirs, metrics=['reward', 'policy_loss']):
    """Compare multiple experiments"""
    results = {}
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        episode_log_path = os.path.join(exp_dir, 'episode_logs.json')
        training_log_path = os.path.join(exp_dir, 'training_logs.json')
        
        if os.path.exists(episode_log_path):
            with open(episode_log_path, 'r') as f:
                episode_logs = json.load(f)
                
        if os.path.exists(training_log_path):
            with open(training_log_path, 'r') as f:
                training_logs = json.load(f)
                
        results[exp_name] = {
            'episode_logs': episode_logs,
            'training_logs': training_logs
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        for exp_name, data in results.items():
            if metric == 'reward' and data['episode_logs']:
                episodes = [log['episode'] for log in data['episode_logs']]
                values = [log['reward'] for log in data['episode_logs']]
                axes[i].plot(episodes, values, label=exp_name, alpha=0.7)
            elif metric == 'policy_loss' and data['training_logs']:
                steps = range(len(data['training_logs']))
                values = [log['policy_loss'] for log in data['training_logs']]
                axes[i].plot(steps, values, label=exp_name, alpha=0.7)
                
        axes[i].set_title(f'{metric.title()} Comparison')
        axes[i].set_xlabel('Episode' if metric == 'reward' else 'Training Step')
        axes[i].set_ylabel(metric.title())
        axes[i].legend()
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.show()
    
    return results
