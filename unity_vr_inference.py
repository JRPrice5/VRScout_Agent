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
                 query_frequency=1):
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
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.backbone = backbone
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nheads = nheads
        
        # Load dataset statistics
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Set up preprocessing and postprocessing functions
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        
        # Load policy
        self.policy_class = policy_class
        self.policy = self._load_policy()
        
        # Initialize inference variables
        self.timestep = 0
        self.query_frequency = query_frequency
        self.last_actions = None
        
        # Initialize temporal aggregation variables if needed
        if self.temporal_agg:
            self.state_dim = len(self.stats['qpos_mean'])
            self.window_size = 30  # Number of past predictions to store
            self.prediction_buffer = None  # Will store predictions from past policy calls
            self.prediction_timesteps = None  # Will store the timestep when each prediction was made
            self.buffer_index = 0  # Current position in the circular buffer
            self.last_unique_timestep = -1  # Track the last unique timestep to handle repeats
            self.decay_factor = 0.5  # Decay factor for exponential weighting (higher = more focus on recent)
            self.current_weight = 0.7  # Weight given to the current prediction (0-1)
        
        # Initialize socket
        self.socket = None
        self.client_socket = None
        self.running = False
    
    def _load_policy(self):
        """Load the trained policy from checkpoint"""
        print(f"Loading policy from {self.ckpt_dir}")
        
        # Determine policy configuration based on policy class
        if self.policy_class == 'ACT':
            # ACT policy configuration with DETR required parameters
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
                # Add DETR required parameters
                'ckpt_dir': self.ckpt_dir,
                'policy_class': self.policy_class,
                'task_name': self.task_name,
                'seed': self.seed,
                'num_epochs': self.num_epochs,
                'camera_names': self.camera_names,
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
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        policy.eval()
        
        return policy
    
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
                self.prediction_buffer = torch.zeros([self.window_size, self.chunk_size, self.state_dim]).cuda()
                # Track which timestep each buffer entry was made at
                self.prediction_timesteps = torch.full([self.window_size], -1, dtype=torch.long).cuda()
            
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
            # Extract qpos from state data
            qpos_numpy = np.array(state_data['qpos'], dtype=np.float32)
            
            # Preprocess qpos
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
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
                    self.buffer_index = (self.buffer_index + 1) % self.window_size
                    
                    # Find all valid entries in the buffer (timestep >= 0)
                    valid_indices = (self.prediction_timesteps >= 0)
                    
                    # Get the current action from the most recent prediction
                    current_prediction_idx = (self.buffer_index - 1) % self.window_size
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
                            weights = torch.exp(-self.decay_factor * time_deltas.float())
                            weights = weights / weights.sum()
                            
                            # Compute weighted average of past predictions
                            history_action = torch.zeros_like(current_action)
                            for action, weight in zip(actions_for_current, weights):
                                history_action += action * weight
                            
                            # Blend current action with history using the current_weight parameter
                            raw_action = (self.current_weight * current_action + 
                                         (1 - self.current_weight) * history_action).unsqueeze(0)
                        else:
                            # No valid predictions for current timestep
                            raw_action = current_action.unsqueeze(0)
                    else:
                        # No valid predictions yet
                        raw_action = current_action.unsqueeze(0)
                else:
                    raw_action = all_actions[:, time_index]
            elif self.policy_class == 'CNNMLP':
                raw_action = self.policy(qpos, curr_image)
            
            # Post-process action
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            
            # Convert to list for JSON serialization
            action_list = action.tolist()
            
            return {
                'action': action_list,
                'timestep': self.timestep
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
            print("Input is grayscale, converting to RGB")
            # If grayscale, duplicate to 3 channels
            if all_cam_images.ndim == 3:  # No channel dimension
                all_cam_images = np.repeat(all_cam_images[:, :, :, np.newaxis], 3, axis=-1)
            else:  # Has channel dim but it's 1
                all_cam_images = np.repeat(all_cam_images, 3, axis=-1)
        
        # Make sure the shape is [batch, height, width, 3]
        # print(f"NumPy image shape: {all_cam_images.shape}")
        
        # Convert to tensor
        image_data = torch.from_numpy(all_cam_images).float()
        
        # Permute to channel-first format expected by PyTorch: [batch, channels, height, width]
        image_data = image_data.permute(0, 3, 1, 2).contiguous()
        
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
        
        # Normalize pixel values to [0,1]
        image_data = image_data / 255.0
        
        return image_data.cuda()


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
    
    interface.start_server()


if __name__ == "__main__":
    main()
