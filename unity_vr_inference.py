#!/usr/bin/env python
"""
Unity VR Environment Interface for ACT Policy Inference

This script loads a trained ACT policy and establishes a server that communicates
with a Unity VR environment. It receives state observations from Unity, runs inference
with the policy, and sends action predictions back to Unity.

Usage:
    python unity_vr_inference.py --ckpt_path path/to/policy_best.ckpt --stats_path path/to/dataset_stats.pkl
"""

import os
import sys
import time
import json
import socket
import argparse
import threading
import numpy as np
import torch
import pickle
from PIL import Image
import io
import base64

# Add the current directory to path to import ACT modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ACT modules
from policy import ACTPolicy, CNNMLPPolicy
from action_utils import ACTION_INDICES
from action_io import preprocess_qpos, postprocess_action_to_env


class UnityVRInterface:
    """Interface between Unity VR environment and ACT policy"""
    
    def __init__(self, 
                 policy_class='ACT',
                 host='127.0.0.1', 
                 port=5556,
                 camera_names=['camera1'],
                 temporal_agg=False,
                 ckpt_dir=None,
                 task_name='default_task',
                 seed=0,
                 num_epochs=100,
                 kl_weight=5.0,
                 chunk_size=8,
                 hidden_dim=256,
                 dim_feedforward=1024,
                 lr=1e-4,
                 lr_backbone=1e-5,
                 backbone='resnet18',
                 enc_layers=4,
                 dec_layers=7,
                 nheads=8,
                 query_frequency=1,
                 reverse_weighting=False,
                 activation='relu',
                 use_dynamic_window=False,
                 # Optimized parameter ranges
                 min_window_size=8,
                 max_window_size=24,
                 base_window_size=16,
                 action_history_size=10,
                 # Unified responsiveness parameter (0.0=smooth, 1.0=responsive)
                 responsiveness=0.5,
                 # Computational throttling
                 stability_threshold=0.1,
                 throttle_interval=5,
                 base_decay_factor=0.04,
                 min_decay_factor=0.02,
                 max_decay_factor=0.08,
                 window_adjustment_rate=0.1,
                 weight_smoothing_factor=0.3,
                 target_window_history_size=5,
                 update_rate_presets=[0.018, 0.02, 0.022, 0.024, 0.026, 0.032, 0.036],
                 base_update_rate=0.02,
                 update_rate_smoothing_factor=0.3,
                 update_rate_history_size=5,
                 # Adaptive normalization parameters
                 adaptive_scale_beta_up=0.3,
                 adaptive_scale_beta_down=0.0,
                 adaptive_min_scale=1e-4,
                 # Debugging
                 dw_debug=True,
                 dw_debug_interval=20,
                 # Dynamic component toggles
                 enable_dynamic_window_size=True,
                 enable_dynamic_decay_factor=True,
                 enable_dynamic_update_rate=True):
        """
        Initialize the interface.
        
        Args:
            policy_class: Type of policy ('ACT' or 'CNNMLP')
            host: Host address for the server
            port: Port for the server
            camera_names: List of camera names
            temporal_agg: Whether to use temporal aggregation
            ckpt_dir: Checkpoint directory (required by DETR)
            task_name: Task name (required by DETR)
            seed: Random seed (required by DETR)
            num_epochs: Number of epochs (required by DETR)
            kl_weight: KL Weight for ACT policy
            chunk_size: Chunk size for ACT policy
            hidden_dim: Hidden dimension for transformer
            dim_feedforward: Feedforward dimension for transformer
            lr: Learning rate
            lr_backbone: Learning rate for backbone
            backbone: Backbone model (e.g., 'resnet18')
            enc_layers: Number of encoder layers
            dec_layers: Number of decoder layers
            nheads: Number of attention heads
            kl_weight: KL Weight for ACT policy
            chunk_size: Chunk size for ACT policy
            reverse_weighting: Whether to put more weight on older actions instead of recent ones
            activation: Activation function to use
            use_dynamic_window: Whether to use dynamic window sizing
            min_window_size: Minimum window size for dynamic sizing
            max_window_size: Maximum window size for dynamic sizing
            base_window_size: Base window size when not using dynamic sizing
            action_history_size: Length of recent action buffer for dynamic window computation
            responsiveness: Unified responsiveness parameter (0.0=smooth, 1.0=responsive)
            stability_threshold: Threshold for detecting stable periods (for throttling)
            throttle_interval: Number of timesteps to skip calculations during stable periods
            base_decay_factor: Base decay factor for exponential weighting
            min_decay_factor: Minimum decay factor (for large windows)
            max_decay_factor: Maximum decay factor (for small windows)
            window_adjustment_rate: Rate of gradual window size adjustment (0-1)
            weight_smoothing_factor: Smoothing factor for historical weighting updates (0-1)
            target_window_history_size: Number of recent target window sizes to track
            update_rate_presets: List of preset update rates to choose from
            base_update_rate: Base update rate when not using dynamic adjustment
            update_rate_smoothing_factor: Smoothing factor for update rate adjustments (0-1)
            update_rate_history_size: Number of recent target update rates to track
            """
        self.host = host
        self.port = port
        self.camera_names = camera_names
        self.temporal_agg = temporal_agg
        
        # Store DETR required parameters
        self.ckpt_dir = ckpt_dir
        self.task_name = task_name
        self.seed = seed
        self.num_epochs = num_epochs
        self.kl_weight = kl_weight
        self.chunk_size = chunk_size
        self.reverse_weighting = reverse_weighting
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.backbone = backbone
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nheads = nheads
        self.activation = activation
        
        # Dynamic window sizing parameters
        self.use_dynamic_window = use_dynamic_window
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.base_window_size = base_window_size
        self.action_history_size = action_history_size
        
        # Unified responsiveness system
        self.responsiveness = responsiveness
        self.stability_threshold = stability_threshold
        self.throttle_interval = throttle_interval
        
        # Decay factor parameters
        self.base_decay_factor = base_decay_factor
        self.min_decay_factor = min_decay_factor
        self.max_decay_factor = max_decay_factor
        
        # Adaptive normalization config
        self.adaptive_scale_beta_up = adaptive_scale_beta_up
        self.adaptive_scale_beta_down = adaptive_scale_beta_down
        self.adaptive_min_scale = adaptive_min_scale
        
        # Debugging
        self.dw_debug = dw_debug
        self.dw_debug_interval = max(1, int(dw_debug_interval))
        
        # Dynamic component toggles
        self.enable_dynamic_window_size = enable_dynamic_window_size
        self.enable_dynamic_decay_factor = enable_dynamic_decay_factor
        self.enable_dynamic_update_rate = enable_dynamic_update_rate
        
        # Dynamic update rate parameters
        self.update_rate_presets = update_rate_presets
        self.base_update_rate = base_update_rate
        self.update_rate_smoothing_factor = update_rate_smoothing_factor
        self.update_rate_history_size = update_rate_history_size
        
        # Gradual adjustment parameters
        self.window_adjustment_rate = window_adjustment_rate
        self.weight_smoothing_factor = weight_smoothing_factor
        self.target_window_history_size = target_window_history_size
        
        # Load dataset statistics
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Load persistent normalization statistics
        self.normalization_stats_path = os.path.join(self.ckpt_dir, 'normalization_stats.pkl')
        self.normalization_stats = self._load_normalization_stats()
        self.save_counter = 0  # Counter for periodic saving
        self.save_frequency = 1000  # Save every N timesteps
        
        # Centralized preprocessing/postprocessing
        self.boolean_indices = ACTION_INDICES['boolean']
        
        # Load policy
        self.policy_class = policy_class
        self.policy = self._load_policy()
        
        # Initialize inference variables
        self.timestep = 0
        self.query_frequency = query_frequency
        self.last_actions = None
        
        # For dynamic window sizing sent to Unity
        self.window_size = self.base_window_size  # Current window size
        self.decay_factor = self.base_decay_factor  # Current decay factor (dynamically adjusted)
        self.current_update_rate = self.base_update_rate  # Current update rate

        # Initialize temporal aggregation variables if needed
        if self.temporal_agg:
            self.state_dim = len(self.stats['qpos_mean'])
            self.prediction_buffer = None  # Will store predictions from past policy calls
            self.prediction_timesteps = None  # Will store the timestep when each prediction was made
            self.buffer_index = 0  # Current position in the circular buffer
            self.last_unique_timestep = -1  # Track the last unique timestep to handle repeats
            self.current_weight = 0.0  # Weight given to the current prediction (0-1)
            self.reverse_weighting = reverse_weighting

            # Initialize action history for dynamic window sizing
            if self.use_dynamic_window:
                self.recent_actions = torch.zeros([self.action_history_size, self.state_dim]).cuda()
                self.action_history_index = 0
                self.previous_qpos = None
                
                # Gradual adjustment state tracking
                self.target_window_size = self.base_window_size  # Target window size from calculations
                self.target_window_history = [self.base_window_size] * self.target_window_history_size
                self.smoothed_decay_factor = self.base_decay_factor  # Smoothed decay factor
                self.decay_factor_history = [self.base_decay_factor] * self.target_window_history_size
                
                # Dynamic update rate state tracking
                self.smoothed_update_rate = self.base_update_rate  # Smoothed update rate
                self.update_rate_history = [self.base_update_rate] * self.update_rate_history_size
                
                # Initialize adaptive normalization scales from persistent stats or defaults
                self.scale_variance = self.normalization_stats.get('scale_variance', 0.1)
                self.scale_magnitude = self.normalization_stats.get('scale_magnitude', 2.0)
                self.scale_consistency = self.normalization_stats.get('scale_consistency', 0.1)
                self.scale_state_change = self.normalization_stats.get('scale_state_change', 0.5)
                self.scale_speed = self.normalization_stats.get('scale_speed', 0.5)
                
                # Computational throttling state
                self.last_calculation_timestep = -1
                self.cached_target_window = self.base_window_size
                self.stability_counter = 0
                self.recent_window_changes = []
        
        # Initialize socket
        self.socket = None
        self.client_socket = None
        self.running = False
    
    def _load_normalization_stats(self):
        """Load persistent normalization statistics from file, with fallback to defaults"""
        try:
            if os.path.exists(self.normalization_stats_path):
                with open(self.normalization_stats_path, 'rb') as f:
                    stats = pickle.load(f)
                print(f"Loaded normalization stats from {self.normalization_stats_path}")
                return stats
            else:
                print(f"No normalization stats file found at {self.normalization_stats_path}, using defaults")
                return {}
        except Exception as e:
            print(f"Error loading normalization stats: {e}, using defaults")
            return {}
    
    def _save_normalization_stats(self):
        """Save current normalization statistics to file"""
        try:
            stats = {
                'scale_variance': self.scale_variance,
                'scale_magnitude': self.scale_magnitude,
                'scale_consistency': self.scale_consistency,
                'scale_state_change': self.scale_state_change,
                'scale_speed': self.scale_speed,
                'last_updated': self.timestep,
                'total_updates': getattr(self, 'total_updates', 0) + 1
            }
            
            # Atomic write to prevent corruption
            temp_path = self.normalization_stats_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(stats, f)
            
            # Replace original file atomically
            if os.path.exists(self.normalization_stats_path):
                os.remove(self.normalization_stats_path)
            os.rename(temp_path, self.normalization_stats_path)
            
            self.total_updates = stats['total_updates']
            print(f"Saved normalization stats to {self.normalization_stats_path}")
            
        except Exception as e:
            print(f"Error saving normalization stats: {e}")
    
    def _load_policy(self):
        """Load the trained policy from checkpoint"""
        print(f"Loading policy from {self.ckpt_dir}")
        
        # Determine policy configuration based on policy class
        if self.policy_class == 'ACT':
            # ACT policy configuration with DETR required parameters
            policy_config = {
                'state_dim': len(self.stats['qpos_mean']),
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
                # Add DETR required parameters
                'ckpt_dir': self.ckpt_dir,
                'policy_class': self.policy_class,
                'task_name': self.task_name,
                'seed': self.seed,
                'num_epochs': self.num_epochs,
                'camera_names': self.camera_names,
                'activation': self.activation,
            }
            policy = ACTPolicy(policy_config)
        elif self.policy_class == 'CNNMLP':
            policy_config = {
                'state_dim': len(self.stats['qpos_mean']),
                'action_dim': len(self.stats['action_mean']),
                'lr': self.lr,
                'lr_backbone': self.lr_backbone,
                'backbone': self.backbone,
                'num_queries': 1,
                # Add DETR required parameters
                'ckpt_dir': self.ckpt_dir,
                'policy_class': self.policy_class,
                'task_name': self.task_name,
                'seed': self.seed,
                'num_epochs': self.num_epochs,
                'camera_names': self.camera_names,
            }
            policy = CNNMLPPolicy(policy_config)
        else:
            raise ValueError(f"Unsupported policy class: {self.policy_class}")
        
        # Load checkpoint
        ckpt_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        policy.load_state_dict(torch.load(ckpt_path), strict=False) # CAUTION : test only
        policy.cuda()
        policy.eval()

        # Optionally compile the model for faster inference (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                policy = torch.compile(policy)
            except Exception as e:
                # Fall back gracefully if compile is unsupported on this system
                print(f"torch.compile failed, continuing without compile: {e}")

        return policy
    
    def _get_responsiveness_weights(self):
        """
        Convert unified responsiveness parameter to individual factor weights
        
        Returns:
            dict: Factor weights based on responsiveness level
        """
        # Responsiveness curves: 0.0=smooth, 1.0=responsive
        r = self.responsiveness
        
        return {
            'variance': 0.0,               # disabled
            'magnitude': 0.0,              # disabled
            'consistency': 0.0,            # disabled
            'state_change': 0.0,           # disabled  
            'speed': 2.0 * r               # 0.0 to 2.0 (speed dominates at high responsiveness)
        }
    
    def _should_skip_calculation(self):
        """
        Determine if we should skip expensive factor calculations due to stability
        
        Returns:
            bool: True if calculation should be skipped
        """
        if not self.use_dynamic_window:
            return False
            
        # Always calculate on first few timesteps
        if self.timestep < 10:
            return False
            
        # Check stability based on recent window size changes
        if len(self.recent_window_changes) >= 8:  # Need more history for stable detection
            recent_std = np.std(self.recent_window_changes[-8:])
            if recent_std < self.stability_threshold * 2.0:  # More lenient threshold
                self.stability_counter += 1
                # Only throttle if we've been stable AND enough time has passed
                timesteps_since_calc = self.timestep - self.last_calculation_timestep
                if self.stability_counter > 1 and timesteps_since_calc < self.throttle_interval:  # Less strict counter
                    return True
            else:
                self.stability_counter = max(0, self.stability_counter - 1)  # Gradual decay instead of reset
                
        return False
    
    def _calculate_target_window_size(self, raw_action, qpos, actions_for_current=None):
        """
        Calculate target window size using unified responsiveness parameter and computational throttling
        
        Args:
            raw_action: Current raw action tensor
            qpos: Current joint positions
            actions_for_current: Past predictions for current timestep (optional)
            
        Returns:
            int: Target window size between min_window_size and max_window_size
        """
        if not self.use_dynamic_window:
            return self.base_window_size
        
        # Check if we should skip expensive calculations
        if self._should_skip_calculation():
            return self.cached_target_window
            
        # Update calculation timestamp
        self.last_calculation_timestep = self.timestep
        
        # Get factor weights from unified responsiveness parameter
        weights = self._get_responsiveness_weights()
        
        factors = []
        total_weight = 0
        debug_info = {
            'responsiveness': self.responsiveness,
            'variance': None, 'var_scale': None, 'var_norm': None,
            'magnitude': None, 'mag_scale': None, 'mag_norm': None,
            'pred_var': None, 'cons_scale': None, 'cons_norm': None,
            'state_change': None, 'state_scale': None, 'state_norm': None,
            'speed': None, 'speed_scale': None, 'speed_norm': None,
        }
        
        # Helper for adaptive scale update (fast-up, slow-down EMA)
        def _update_scale(name: str, value: float) -> float:
            scale_attr = f"scale_{name}"
            current = getattr(self, scale_attr)
            beta = self.adaptive_scale_beta_up if value > current else self.adaptive_scale_beta_down
            new_scale = (1.0 - beta) * current + beta * value
            # Ensure a minimal floor to avoid division by near-zero
            new_scale = max(new_scale, self.adaptive_min_scale)
            setattr(self, scale_attr, new_scale)
            return new_scale

        # Factor 1: Action Variance (higher variance -> smaller window for responsiveness)
        if weights['variance'] > 0 and self.timestep >= 5:  # Need some history
            # Get valid recent actions
            valid_actions = self.recent_actions[:min(self.timestep, self.action_history_size)]
            if len(valid_actions) > 1:
                action_variance = torch.var(valid_actions, dim=0).mean().item()
                # Adaptive normalization
                variance_scale = _update_scale('variance', action_variance)
                variance_factor = min(action_variance / (variance_scale + 1e-8), 1.0)
                factors.append(weights['variance'] * (1.0 - variance_factor))
                total_weight += weights['variance']
                # Debug capture
                debug_info['variance'] = action_variance
                debug_info['var_scale'] = variance_scale
                debug_info['var_norm'] = variance_factor
        
        # Factor 2: Action Magnitude (higher magnitude -> smaller window for responsiveness)
        if weights['magnitude'] > 0:
            action_magnitude = torch.norm(raw_action).item()
            # Adaptive normalization
            magnitude_scale = _update_scale('magnitude', action_magnitude)
            magnitude_factor = min(action_magnitude / (magnitude_scale + 1e-8), 1.0)
            factors.append(weights['magnitude'] * (1.0 - magnitude_factor))
            total_weight += weights['magnitude']
            # Debug capture
            debug_info['magnitude'] = action_magnitude
            debug_info['mag_scale'] = magnitude_scale
            debug_info['mag_norm'] = magnitude_factor
        
        # Factor 3: Prediction Consistency (higher consistency -> larger window for smoothness)
        if weights['consistency'] > 0 and actions_for_current is not None and len(actions_for_current) > 1:
            prediction_variance = torch.var(actions_for_current, dim=0).mean().item()
            # Adaptive normalization (higher variance -> less consistency)
            consistency_scale = _update_scale('consistency', prediction_variance)
            consistency_factor = 1.0 - min(prediction_variance / (consistency_scale + 1e-8), 1.0)
            factors.append(weights['consistency'] * consistency_factor)
            total_weight += weights['consistency']
            # Debug capture
            debug_info['pred_var'] = prediction_variance
            debug_info['cons_scale'] = consistency_scale
            debug_info['cons_norm'] = 1.0 - consistency_factor  # normalized variance component
        
        # Factor 4: State Change Rate (higher change rate -> smaller window)
        if weights['state_change'] > 0 and self.previous_qpos is not None:
            state_change = torch.norm(qpos.squeeze() - self.previous_qpos).item()
            # Adaptive normalization
            state_change_scale = _update_scale('state_change', state_change)
            change_factor = min(state_change / (state_change_scale + 1e-8), 1.0)
            factors.append(weights['state_change'] * (1.0 - change_factor))
            total_weight += weights['state_change']
            # Debug capture
            debug_info['state_change'] = state_change
            debug_info['state_scale'] = state_change_scale
            debug_info['state_norm'] = change_factor
        
        # Factor 5: Action Speed (higher speed -> smaller window for responsiveness)
        if weights['speed'] > 0 and self.timestep >= 2:  # Need at least 2 actions
            # Calculate mean speed using vectorized operations
            num_actions = min(self.timestep, self.action_history_size)
            if num_actions >= 2:
                # Get slice of recent actions (most recent first)
                end_idx = self.action_history_index
                start_idx = (end_idx - num_actions) % self.action_history_size
                
                if start_idx < end_idx:
                    # Contiguous slice
                    recent_actions = self.recent_actions[start_idx:end_idx]
                else:
                    # Wrap-around slice
                    recent_actions = torch.cat([
                        self.recent_actions[start_idx:],
                        self.recent_actions[:end_idx]
                    ], dim=0)
                
                # Calculate differences between consecutive actions
                action_diffs = recent_actions[1:] - recent_actions[:-1]
                
                # Calculate speeds (L2 norms) for all differences at once
                speeds = torch.norm(action_diffs, dim=1)
                
                # Calculate mean speed
                mean_action_speed = speeds.mean().item()
                
                # Adaptive normalization
                speed_scale = _update_scale('speed', mean_action_speed)
                speed_factor = min(mean_action_speed / (speed_scale + 1e-8), 1.0)
                factors.append(weights['speed'] * (1.0 - speed_factor))
                total_weight += weights['speed']
                # Debug capture
                debug_info['speed'] = mean_action_speed
                debug_info['speed_scale'] = speed_scale
                debug_info['speed_norm'] = speed_factor
        
        # Combine factors (weighted average of available factors)
        if factors and total_weight > 0:
            combined_factor = sum(factors) / total_weight
        else:
            combined_factor = 0.5  # Default to middle ground
        
        # Map combined factor to window size range
        target_window_size = int(self.min_window_size + 
                                combined_factor * (self.max_window_size - self.min_window_size))
    
        if self.dw_debug and self.timestep % self.dw_debug_interval == 0:
            print(f"Debug info: {debug_info}")
            print(f"Combined factor: {combined_factor:.3f}, Target window: {target_window_size}")

        # Cache the result for potential throttling
        self.cached_target_window = target_window_size
        
        # Update recent window changes for stability detection
        self.recent_window_changes.append(target_window_size)
        if len(self.recent_window_changes) > 10:
            self.recent_window_changes.pop(0)

        return max(self.min_window_size, min(target_window_size, self.max_window_size))
    
    def _update_gradual_window_size(self, target_window_size):
        """
        Apply gradual adjustment to window size with smoothing
        
        Args:
            target_window_size: The target window size calculated from factors
            
        Returns:
            int: Gradually adjusted window size
        """
        if not self.use_dynamic_window or not self.enable_dynamic_window_size:
            return self.base_window_size
        
        # Update target window history for smoothing
        self.target_window_history.pop(0)
        self.target_window_history.append(target_window_size)
        
        # Calculate smoothed target (moving average)
        smoothed_target = sum(self.target_window_history) / len(self.target_window_history)
        
        # Apply gradual adjustment toward smoothed target
        current_window = float(self.window_size)
        adjustment = (smoothed_target - current_window) * self.window_adjustment_rate
        
        # Ensure minimum adjustment of 1 when there's a significant difference
        if abs(smoothed_target - current_window) >= 1.0 and abs(adjustment) < 1.0:
            adjustment = 1.0 if smoothed_target > current_window else -1.0
            
        new_window_size = int(round(current_window + adjustment))
        
        return max(self.min_window_size, min(new_window_size, self.max_window_size))
    
    def _calculate_target_decay_factor(self, window_size):
        """
        Calculate target decay factor based on window size
        Smaller windows -> higher decay factor (more focus on recent)
        Larger windows -> lower decay factor (more smoothing)
        """
        if not self.use_dynamic_window or not self.enable_dynamic_decay_factor:
            return self.base_decay_factor
        
        # Normalize window size to [0, 1] range
        window_ratio = (window_size - self.min_window_size) / (self.max_window_size - self.min_window_size)
        
        # Invert the ratio so smaller windows get higher decay factors
        target_decay_factor = self.max_decay_factor - (window_ratio * (self.max_decay_factor - self.min_decay_factor))
        
        return target_decay_factor
    
    def _update_smoothed_decay_factor(self, target_decay_factor):
        """
        Apply smoothed historical weighting update to decay factor
        
        Args:
            target_decay_factor: The target decay factor calculated from window size
            
        Returns:
            float: Smoothed decay factor
        """
        if not self.use_dynamic_window or not self.enable_dynamic_decay_factor:
            return self.base_decay_factor
        
        # Update decay factor history for smoothing
        self.decay_factor_history.pop(0)
        self.decay_factor_history.append(target_decay_factor)
        
        # Calculate smoothed decay factor using exponential moving average
        # This provides more responsive smoothing than simple moving average
        smoothed_target = sum(self.decay_factor_history) / len(self.decay_factor_history)
        
        # Apply exponential smoothing
        self.smoothed_decay_factor = (self.weight_smoothing_factor * smoothed_target + 
                                     (1 - self.weight_smoothing_factor) * self.smoothed_decay_factor)
        
        return self.smoothed_decay_factor
    
    def _calculate_target_update_rate(self, window_size):
        """
        Calculate target update rate based on window size using preset rates
        Smaller windows -> faster update rates (more responsive)
        Larger windows -> slower update rates (more stable)
        
        Args:
            window_size: Current window size
            
        Returns:
            float: Target update rate from preset list
        """
        if not self.use_dynamic_window or not self.enable_dynamic_update_rate:
            return self.base_update_rate
        
        # Normalize window size to [0, 1] range
        window_ratio = (window_size - self.min_window_size) / (self.max_window_size - self.min_window_size)
        
        # Map to preset index (larger windows get slower rates)
        preset_index = round(window_ratio * (len(self.update_rate_presets) - 1))
        preset_index = max(0, min(preset_index, len(self.update_rate_presets) - 1))
        
        return self.update_rate_presets[preset_index]
    
    def _update_smoothed_update_rate(self, target_update_rate):
        """
        Apply smoothed historical weighting update to update rate - allow fast decreases and slow increases
        
        Args:
            target_update_rate: The target update rate calculated from window size
            
        Returns:
            float: Smoothed update rate (monotonically decreasing)
        """
        if not self.use_dynamic_window or not self.enable_dynamic_update_rate:
            return self.base_update_rate
        
        # Update update rate history for smoothing
        self.update_rate_history.pop(0)
        self.update_rate_history.append(target_update_rate)
        
        # Calculate smoothed update rate using exponential moving average
        smoothed_target = sum(self.update_rate_history) / len(self.update_rate_history)
        
        # Allow both increases and decreases with asymmetric rates: fast decrease, slow increase
        if smoothed_target < self.smoothed_update_rate:
            # Decreasing: skip smoothing, use target directly for immediate response
            self.smoothed_update_rate = smoothed_target
        else:
            # Increasing: use lower smoothing factor for slower response
            effective_smoothing_factor = self.update_rate_smoothing_factor
            # Apply exponential smoothing for increases only
            self.smoothed_update_rate = (effective_smoothing_factor * smoothed_target + 
                                       (1 - effective_smoothing_factor) * self.smoothed_update_rate)
        
        # Find nearest preset to ensure smoothed rate stays within preset values
        distances = [abs(self.smoothed_update_rate - preset) for preset in self.update_rate_presets]
        nearest_preset_index = distances.index(min(distances))
        
        return self.update_rate_presets[nearest_preset_index]
    
    def _handle_unity_toggle_commands(self, state_data):
        """
        Handle toggle commands from Unity client for dynamic window components
        
        Args:
            state_data: Dictionary containing state data from Unity
            
        Returns:
            bool: True if this is a toggle-only message, False otherwise
        """
        is_toggle_message = False
        # Check for toggle commands in state_data (now using int format: -1=not set, 0=false, 1=true)
        if 'toggle_window_size' in state_data and state_data['toggle_window_size'] != -1:
            self.enable_dynamic_window_size = bool(state_data['toggle_window_size'])
            is_toggle_message = True
            if self.dw_debug:
                print(f"Dynamic window size toggled: {self.enable_dynamic_window_size}")
        
        if 'toggle_decay_factor' in state_data and state_data['toggle_decay_factor'] != -1:
            self.enable_dynamic_decay_factor = bool(state_data['toggle_decay_factor'])
            is_toggle_message = True
            if self.dw_debug:
                print(f"Dynamic decay factor toggled: {self.enable_dynamic_decay_factor}")
        
        if 'toggle_update_rate' in state_data and state_data['toggle_update_rate'] != -1:
            self.enable_dynamic_update_rate = bool(state_data['toggle_update_rate'])
            is_toggle_message = True
            if self.dw_debug:
                print(f"Dynamic update rate toggled: {self.enable_dynamic_update_rate}")
        
        return is_toggle_message
    
    def _update_action_history(self, raw_action, qpos):
        """Update the action and state history for dynamic window sizing"""
        if self.temporal_agg and self.use_dynamic_window:
            # Store current action in circular buffer
            self.recent_actions[self.action_history_index] = raw_action.squeeze()
            self.action_history_index = (self.action_history_index + 1) % self.action_history_size
            
            # Update previous qpos for next iteration
            self.previous_qpos = qpos.squeeze().clone()
    
    def start_server(self):
        """Start the server to listen for Unity connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Set a timeout to allow for keyboard interrupts
        self.socket.settimeout(1.0)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Server started on {self.host}:{self.port}")
            print("Press Ctrl+C to stop the server")
            
            self.running = True
            while self.running:
                try:
                    # Accept with timeout to allow for keyboard interrupts
                    self.client_socket, addr = self.socket.accept()
                    print(f"Connected to Unity client at {addr}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(target=self._handle_client)
                    client_thread.daemon = True
                    client_thread.start()
                    client_thread.join()
                except socket.timeout:
                    # This is expected, just continue the loop to check if we should still be running
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {e}")
                    # Short sleep to avoid CPU spinning if there's a persistent error
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        except Exception as e:
            print(f"Error in server: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the server and clean up resources"""
        print("Stopping server...")
        self.running = False
        
        # Close client socket if it exists
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass  # Socket might already be closed or not connected
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        # Close server socket if it exists
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        print("Server stopped successfully")
        
        # Release CUDA resources if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _handle_client(self):
        """Handle communication with a connected Unity client"""
        try:
            # Reset timestep counter when a new client connects
            self.timestep = 0
            if self.temporal_agg:
                # Initialize prediction buffer with sliding window approach
                # [window_size, chunk_size, state_dim] - each entry stores chunk_size predictions
                self.prediction_buffer = torch.zeros([self.max_window_size, self.chunk_size, self.state_dim]).cuda()
                # Track which timestep each buffer entry was made at
                self.prediction_timesteps = torch.full([self.max_window_size], -1, dtype=torch.long).cuda()
                
                # Reset dynamic window sizing buffers and state
                if self.use_dynamic_window:
                    # Reset action history buffer
                    self.recent_actions = torch.zeros([self.action_history_size, self.state_dim]).cuda()
                    self.action_history_index = 0
                    self.previous_qpos = None
                    
                    # Reset window size state
                    self.window_size = self.base_window_size
                    self.target_window_size = self.base_window_size
                    self.target_window_history = [self.base_window_size] * self.target_window_history_size
                    
                    # Reset decay factor state
                    self.decay_factor = self.base_decay_factor
                    self.smoothed_decay_factor = self.base_decay_factor
                    self.decay_factor_history = [self.base_decay_factor] * self.target_window_history_size
                    
                    # Reset update rate state
                    self.current_update_rate = self.base_update_rate
                    self.smoothed_update_rate = self.base_update_rate
                    self.update_rate_history = [self.base_update_rate] * self.update_rate_history_size
            
            buffer_size = 4096
            data_buffer = b""
            
            # Set a timeout on the client socket to allow for clean shutdown
            self.client_socket.settimeout(0.5)
            print("Client connected and ready for communication")
            
            while self.running:
                try:
                    # Receive data from Unity with timeout
                    chunk = self.client_socket.recv(buffer_size)
                    if not chunk:
                        print("Unity client disconnected")
                        break
                    
                    data_buffer += chunk
                    
                    # Check if we have a complete message
                    if b"END_OF_MESSAGE" in data_buffer:
                        # Extract the complete message
                        message_end = data_buffer.find(b"END_OF_MESSAGE")
                        message = data_buffer[:message_end]
                        data_buffer = data_buffer[message_end + len(b"END_OF_MESSAGE"):]
                        
                        # Process the message
                        try:
                            state_data = json.loads(message.decode('utf-8'))
                            
                            # Use client timestep for synchronization if available
                            if 'timestep' in state_data:
                                self.timestep = state_data['timestep']
                            
                            # Handle toggle commands from Unity client
                            if self._handle_unity_toggle_commands(state_data):
                                # This is a toggle-only message, no response needed
                                continue
                                
                            action = self._process_state_and_get_action(state_data)
                            
                            # Send action back to Unity
                            response = json.dumps(action) + "END_OF_MESSAGE"
                            self.client_socket.sendall(response.encode('utf-8'))
                        except json.JSONDecodeError:
                            print("Error decoding JSON from Unity")
                        except Exception as e:
                            print(f"Error processing state data: {e}")
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
            print(f"Error handling client: {e}")
        finally:
            print("Closing client connection")
            if self.client_socket:
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass  # Socket might already be closed
                self.client_socket.close()
                self.client_socket = None
    
    def _process_state_and_get_action(self, state_data):
        """
        Process state data from Unity and get action from policy
        
        Args:
            state_data: Dictionary containing state information from Unity
                Expected keys:
                - 'qpos': List of joint positions/state variables
                - 'image': Base64 encoded image data or dictionary of images
                - 'timestep': Client timestep for synchronization
                
        Returns:
            Dictionary containing action data to send back to Unity
        """
        with torch.inference_mode():
            # Extract and preprocess qpos using shared utility
            qpos_numpy = np.array(state_data['qpos'], dtype=np.float32)
            qpos = preprocess_qpos(qpos_numpy, self.stats).cuda().unsqueeze(0)
            
            # Process image data
            curr_image = self._process_image_data(state_data)
            
            # Get action from policy
            if self.policy_class == 'ACT':
                # Only query the policy when needed based on query_frequency
                time_index = self.timestep % self.query_frequency
                if time_index == 0:
                    all_actions = self.policy(qpos, curr_image)
                    self.last_actions = all_actions  # Store for future timesteps
                else:
                    # Use previously computed actions if not querying on this timestep
                    if self.last_actions is None:
                        # First timestep or in case there's no previous action
                        all_actions = self.policy(qpos, curr_image)
                        self.last_actions = all_actions
                        time_index = 0
                    else:
                        all_actions = self.last_actions
                
                if self.temporal_agg:
                    # Check if this is a new timestep or a repeated one
                    is_new_timestep = self.timestep > self.last_unique_timestep
                    if is_new_timestep:
                        self.last_unique_timestep = self.timestep
                    
                    # Store the full chunk of predictions in the circular buffer
                    self.prediction_buffer[self.buffer_index] = all_actions.squeeze(0)  # Remove batch dimension
                    self.prediction_timesteps[self.buffer_index] = self.timestep
                    
                    # Update buffer index for circular buffer behavior
                    self.buffer_index = (self.buffer_index + 1) % self.max_window_size
                    
                    # Find all valid entries in the buffer (timestep >= 0) within current window size
                    valid_indices = (self.prediction_timesteps >= 0)
                    # Limit to current window size by finding the most recent entries
                    valid_timesteps = self.prediction_timesteps[valid_indices]
                    if len(valid_timesteps) > self.window_size:
                        # Get indices of the most recent window_size entries
                        sorted_indices = torch.argsort(valid_timesteps, descending=True)
                        keep_indices = sorted_indices[:self.window_size]
                        # Create new valid_indices mask
                        new_valid_indices = torch.zeros_like(valid_indices)
                        valid_positions = torch.where(valid_indices)[0]
                        new_valid_indices[valid_positions[keep_indices]] = True
                        valid_indices = new_valid_indices
                    
                    # Get the current action from the most recent prediction
                    current_prediction_idx = (self.buffer_index - 1) % self.max_window_size
                    if self.prediction_timesteps[current_prediction_idx] >= 0:
                        current_action = self.prediction_buffer[current_prediction_idx, time_index]
                    else:
                        # Fallback if buffer is not yet populated
                        current_action = all_actions[:, time_index].squeeze(0)
                    
                    # For each valid past prediction, extract the action predicted for the current timestep
                    if torch.any(valid_indices):
                        valid_timesteps = self.prediction_timesteps[valid_indices]
                        valid_predictions = self.prediction_buffer[valid_indices]
                        
                        # Calculate relative position in each prediction's chunk
                        # This finds where in each past prediction's chunk our current timestep would be
                        relative_positions = self.timestep - valid_timesteps
                        
                        # Filter to only include predictions that have a valid prediction for current timestep
                        valid_for_current = (relative_positions >= 0) & (relative_positions < self.chunk_size)
                        
                        if torch.any(valid_for_current):
                            # Get predictions that include the current timestep
                            usable_timesteps = valid_timesteps[valid_for_current]
                            usable_predictions = valid_predictions[valid_for_current]
                            usable_positions = relative_positions[valid_for_current]
                            
                            # Sort by recency (most recent first)
                            sorted_indices = torch.argsort(usable_timesteps, descending=True)
                            sorted_timesteps = usable_timesteps[sorted_indices]
                            sorted_predictions = usable_predictions[sorted_indices]
                            sorted_positions = usable_positions[sorted_indices]
                            
                            # Extract the specific action for current timestep from each past prediction
                            actions_for_current = torch.stack([
                                pred[pos] for pred, pos in zip(sorted_predictions, sorted_positions)
                            ])
                            
                            # Apply exponential weighting based on how old the prediction is
                            time_deltas = self.timestep - sorted_timesteps
                            
                            # Use smoothed decay factor for more stable weighting
                            current_decay = self.smoothed_decay_factor if self.use_dynamic_window else self.decay_factor
                            
                            if self.reverse_weighting:
                                # Put more weight on older actions (reverse the weighting)
                                # Invert time_deltas so older predictions get higher weights
                                max_delta = time_deltas.max()
                                inverted_deltas = max_delta - time_deltas
                                weights = torch.exp(-current_decay * inverted_deltas.float())
                            else:
                                # Default: put more weight on recent actions
                                weights = torch.exp(-current_decay * time_deltas.float())
                                
                            weights = weights / weights.sum()

                            # Compute weighted average of past predictions (vectorized)
                            history_action = (actions_for_current * weights.unsqueeze(-1)).sum(dim=0)

                            # Blend current action with history using the current_weight parameter
                            raw_action = (self.current_weight * current_action +
                                          (1 - self.current_weight) * history_action).unsqueeze(0)
                        else:
                            # No valid predictions for current timestep
                            raw_action = current_action.unsqueeze(0)
                            actions_for_current = None
                    else:
                        # No valid predictions yet
                        raw_action = current_action.unsqueeze(0)
                        actions_for_current = None
                else:
                    raw_action = all_actions[:, time_index]
                    actions_for_current = None
            
            elif self.policy_class == 'CNNMLP':
                raw_action = self.policy(qpos, curr_image)
                actions_for_current = None
            
            # Update action history for dynamic window sizing
            self._update_action_history(raw_action, qpos)
            
            # Calculate dynamic window size and decay factor with gradual adjustment if needed
            if self.temporal_agg and self.use_dynamic_window:
                # Calculate target values based on current conditions (only if respective components are enabled)
                target_window_size = self._calculate_target_window_size(raw_action, qpos, actions_for_current)
                target_decay_factor = self._calculate_target_decay_factor(target_window_size) if self.enable_dynamic_decay_factor else self.base_decay_factor
                target_update_rate = self._calculate_target_update_rate(target_window_size) if self.enable_dynamic_update_rate else self.base_update_rate
                
                # Apply gradual adjustments with smoothing (only if respective components are enabled)
                self.window_size = self._update_gradual_window_size(target_window_size) if self.enable_dynamic_window_size else self.base_window_size
                self.decay_factor = self._update_smoothed_decay_factor(target_decay_factor) if self.enable_dynamic_decay_factor else self.base_decay_factor
                self.current_update_rate = self._update_smoothed_update_rate(target_update_rate) if self.enable_dynamic_update_rate else self.base_update_rate
                
                # Optional debug output
                # print(f"Target window: {target_window_size}, Actual window: {self.window_size}, "
                #       f"Target decay: {target_decay_factor:.4f}, Smoothed decay: {self.decay_factor:.4f}")
            
            # Post-process action using shared utility
            raw_action = raw_action.squeeze(0)
            action = postprocess_action_to_env(raw_action, self.stats, self.boolean_indices)
            
            # Convert to list for JSON serialization
            action_list = action.tolist()
            
            # Periodic saving of normalization stats
            if self.use_dynamic_window:
                self.save_counter += 1
                if self.save_counter >= self.save_frequency:
                    self._save_normalization_stats()
                    self.save_counter = 0
            
            return {
                'action': action_list,
                'timestep': self.timestep,
                'window_size': self.window_size,
                'decay_factor': f"{self.decay_factor:.4f}",
                'stepped_inference_update_rate': f"{self.current_update_rate:.4f}",
                'enable_dynamic_window_size': self.enable_dynamic_window_size,
                'enable_dynamic_decay_factor': self.enable_dynamic_decay_factor,
                'enable_dynamic_update_rate': self.enable_dynamic_update_rate
            }
    
    def _process_image_data(self, state_data):
        """
        Process image data from Unity
        
        Args:
            state_data: Dictionary containing state data from Unity
            
        Returns:
            Processed image tensor ready for the policy
        """
        # Check the image format (raw, jpg, png, etc.)
        image_format = state_data.get('imageFormat', 'raw')
        is_raw_format = image_format == 'raw'
        is_jpg_format = image_format == 'jpg'
        
        # Get image dimensions from state data
        width = state_data.get('imageWidth', 224)
        height = state_data.get('imageHeight', 224)
        
        # Check if we have multiple cameras or just one
        if 'images' in state_data:
            images_dict = state_data['images']
            all_cam_images = []
            
            for cam_name in self.camera_names:
                if cam_name in images_dict:
                    # Decode base64 image
                    img_data = base64.b64decode(images_dict[cam_name])
                    
                    if is_raw_format:
                        # Process raw texture data
                        # Reshape the raw bytes into a numpy array (RGB24 format - 3 bytes per pixel)
                        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
                    else:
                        # Process encoded image (JPG/PNG)
                        img = Image.open(io.BytesIO(img_data)).convert('RGB')
                        img_array = np.array(img)
                    
                    all_cam_images.append(img_array)
                else:
                    print(f"Warning: Camera {cam_name} not found in Unity data")
                    # Create a blank image
                    blank_img = np.zeros((height, width, 3), dtype=np.uint8)
                    all_cam_images.append(blank_img)
        else:
            # Single camera case
            img_data = base64.b64decode(state_data['image'])
            
            if is_raw_format:
                # Process raw texture data
                # Reshape the raw bytes into a numpy array (RGB24 format - 3 bytes per pixel)
                img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
            else:
                # Process encoded image (JPG/PNG)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                img_array = np.array(img)
                
            all_cam_images = [img_array]
        
        # Stack images and convert to tensor
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # Ensure all images are 3 channel RGB before converting to tensor
        # Check if images are grayscale (1 channel)
        if all_cam_images.ndim == 3 or (all_cam_images.ndim == 4 and all_cam_images.shape[-1] == 1):
            # If grayscale, duplicate to 3 channels
            if all_cam_images.ndim == 3:  # No channel dimension
                all_cam_images = np.repeat(all_cam_images[:, :, :, np.newaxis], 3, axis=-1)
            else:  # Has channel dim but it's 1
                all_cam_images = np.repeat(all_cam_images, 3, axis=-1)
        
        # Make sure the shape is [batch, height, width, 3]
        # print(f"NumPy image shape: {all_cam_images.shape}")
        
        # Convert to tensor (uint8) to minimize CPU-side work and copies
        image_data = torch.from_numpy(all_cam_images)  # uint8 CPU tensor
        
        # Permute to channel-first format expected by PyTorch: [batch, channels, height, width]
        image_data = image_data.permute(0, 3, 1, 2)  # avoid .contiguous() unless required
        
        # print(f"Final tensor shape (before adding camera dim): {image_data.shape}")
        
        # CRITICAL FIX: Reshape to add camera dimension expected by DETR model
        # The model expects [batch, num_cam, channel, height, width]
        # Currently we have [num_cam, channel, height, width]
        # Add a batch dimension at the start and reshape the tensor to have a camera dimension
        if image_data.shape[0] > 1:
            # We already have multiple cameras
            # Need to add batch dimension: [num_cam, C, H, W] -> [1, num_cam, C, H, W]
            image_data = image_data.unsqueeze(0)
        else:
            # We have a single camera view
            # Need to reshape: [1, C, H, W] -> [1, 1, C, H, W]
            image_data = image_data.unsqueeze(1)
        
        # print(f"Final tensor shape (with camera dim): {image_data.shape}")
        
        # Move to GPU as uint8 (smaller transfer), then convert to float and normalize IN-PLACE on GPU
        image_data = image_data.to(device='cuda', non_blocking=True)
        image_data = image_data.float()
        image_data.div_(255.0)
        
        return image_data


def main():
    parser = argparse.ArgumentParser(description='Unity VR Interface for ACT Policy')
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=5556, help='Server port')
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['camera1'], help='Camera names')
    
    # Add required parameters for DETR module
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory (required by DETR)')
    parser.add_argument('--task_name', type=str, default='default_task', help='Task name (required by DETR)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (required by DETR)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs (required by DETR)')
    parser.add_argument('--kl_weight', type=float, default=5.0, help='KL Weight for ACT policy')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for ACT policy (must match model training config)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for transformer')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension for transformer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model')
    parser.add_argument('--enc_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='Number of decoder layers')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--query_frequency', type=int, default=1, help='How often to query the policy (in timesteps)')
    parser.add_argument('--reverse_weighting', action='store_true', help='Use reverse weighting for temporal aggregation')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function to use')
    
    # Dynamic window sizing arguments
    parser.add_argument('--use_dynamic_window', action='store_true', help='Use dynamic window sizing')
    parser.add_argument('--min_window_size', type=int, default=6, help='Minimum window size for dynamic sizing')
    parser.add_argument('--max_window_size', type=int, default=30, help='Maximum window size for dynamic sizing')
    parser.add_argument('--base_window_size', type=int, default=12, help='Base window size when not using dynamic sizing')
    parser.add_argument('--action_history_size', type=int, default=100, help='Size of action history buffer for dynamic window sizing')
    parser.add_argument('--responsiveness', type=float, default=0.8, help='Unified responsiveness parameter (0.0=smooth, 1.0=responsive)')
    parser.add_argument('--stability_threshold', type=float, default=0.1, help='Threshold for detecting stable periods (for throttling)')
    parser.add_argument('--throttle_interval', type=int, default=5, help='Number of timesteps to skip calculations during stable periods')
    parser.add_argument('--base_decay_factor', type=float, default=0.02, help='Base decay factor for exponential weighting')
    parser.add_argument('--min_decay_factor', type=float, default=0.01, help='Minimum decay factor (for large windows)')
    parser.add_argument('--max_decay_factor', type=float, default=0.022, help='Maximum decay factor (for small windows)')
    
    # Gradual adjustment arguments - 3-tier control system
    # Tier 1: Window Size (Fastest adaptation)
    parser.add_argument('--window_adjustment_rate', type=float, default=0.15, help='Rate of gradual window size adjustment (0-1) - FASTEST')
    parser.add_argument('--target_window_history_size', type=int, default=100, help='Number of recent target window sizes to track - FASTEST')
    # Tier 2: Weight Factor (Medium adaptation)
    parser.add_argument('--weight_smoothing_factor', type=float, default=0.05, help='Smoothing factor for historical weighting updates (0-1) - MEDIUM')
    # Tier 3: Update Rate (Slowest adaptation)
    parser.add_argument('--update_rate_smoothing_factor', type=float, default=0.2, help='Smoothing factor for historical update rate updates (0-1) - SLOWEST')
    parser.add_argument('--update_rate_history_size', type=int, default=200, help='Number of recent update rates to track - SLOWEST')

    # Adaptive normalization (EMA) arguments
    parser.add_argument('--adaptive_scale_beta_up', type=float, default=0.3, help='EMA rate when metric exceeds current scale (fast increase)')
    parser.add_argument('--adaptive_scale_beta_down', type=float, default=0.00005, help='EMA rate when metric is below current scale (slow decrease)')
    parser.add_argument('--adaptive_min_scale', type=float, default=1e-4, help='Minimum allowed scale to avoid division by zero')
    
    # Dynamic component toggle arguments
    # parser.add_argument('--enable_dynamic_window_size', action='store_true', help='Enable dynamic window size adjustment')
    # parser.add_argument('--enable_dynamic_decay_factor', action='store_true', help='Enable dynamic decay factor adjustment')
    # parser.add_argument('--enable_dynamic_update_rate', action='store_true', help='Enable dynamic update rate adjustment')

    args = parser.parse_args()
    
    # If ckpt_dir is not provided, use the directory of ckpt_path
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.dirname(args.ckpt_path)
    
    interface = UnityVRInterface(
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
        reverse_weighting=args.reverse_weighting,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        backbone=args.backbone,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        nheads=args.nheads,
        query_frequency=args.query_frequency,
        activation=args.activation,
        use_dynamic_window=args.use_dynamic_window,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        base_window_size=args.base_window_size,
        action_history_size=args.action_history_size,
        responsiveness=args.responsiveness,
        stability_threshold=args.stability_threshold,
        throttle_interval=args.throttle_interval,
        base_decay_factor=args.base_decay_factor,
        min_decay_factor=args.min_decay_factor,
        max_decay_factor=args.max_decay_factor,
        window_adjustment_rate=args.window_adjustment_rate,
        weight_smoothing_factor=args.weight_smoothing_factor,
        target_window_history_size=args.target_window_history_size,
        update_rate_smoothing_factor=args.update_rate_smoothing_factor,
        update_rate_history_size=args.update_rate_history_size,
        adaptive_scale_beta_up=args.adaptive_scale_beta_up,
        adaptive_scale_beta_down=args.adaptive_scale_beta_down,
        adaptive_min_scale=args.adaptive_min_scale,
        # enable_dynamic_window_size=args.enable_dynamic_window_size,
        # enable_dynamic_decay_factor=args.enable_dynamic_decay_factor,
        # enable_dynamic_update_rate=args.enable_dynamic_update_rate
    )
    
    interface.start_server()


if __name__ == "__main__":
    main()
