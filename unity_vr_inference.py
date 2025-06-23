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
                 port=5555,
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
            self.all_time_actions = None
            self.max_timesteps = 1000  # Can be adjusted based on expected episode length
            self.num_queries = self.chunk_size  # Use chunk_size for consistency
        
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
                self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim]).cuda()
            
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
                            action = self._process_state_and_get_action(state_data)
                            
                            # Send action back to Unity
                            response = json.dumps(action) + "END_OF_MESSAGE"
                            self.client_socket.sendall(response.encode('utf-8'))
                            
                            self.timestep += 1
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
                if self.timestep % self.query_frequency == 0:
                    all_actions = self.policy(qpos, curr_image)
                    self.last_actions = all_actions  # Store for future timesteps
                else:
                    # Use previously computed actions if not querying on this timestep
                    if self.last_actions is None:
                        # First timestep or in case there's no previous action
                        all_actions = self.policy(qpos, curr_image)
                        self.last_actions = all_actions
                    else:
                        all_actions = self.last_actions
                
                if self.temporal_agg:
                    self.all_time_actions[[self.timestep], self.timestep:self.timestep+self.num_queries] = all_actions
                    actions_for_curr_step = self.all_time_actions[:, self.timestep]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    # Apply exponential weighting
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # Safe indexing - ensure we're not accessing beyond bounds
                    time_index = self.timestep % self.chunk_size
                    raw_action = all_actions[:, time_index]
            elif self.policy_class == 'CNNMLP':
                raw_action = self.policy(qpos, curr_image)
            
            # Post-process action
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            
            # Increment timestep
            self.timestep += 1
            
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
        # Check if we have multiple cameras or just one
        if 'images' in state_data:
            images_dict = state_data['images']
            all_cam_images = []
            
            for cam_name in self.camera_names:
                if cam_name in images_dict:
                    # Decode base64 image
                    img_data = base64.b64decode(images_dict[cam_name])
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    img_array = np.array(img)
                    all_cam_images.append(img_array)
                else:
                    print(f"Warning: Camera {cam_name} not found in Unity data")
                    # Create a blank image
                    blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    all_cam_images.append(blank_img)
        else:
            # Single camera case
            img_data = base64.b64decode(state_data['image'])
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
        print(f"NumPy image shape: {all_cam_images.shape}")
        
        # Convert to tensor
        image_data = torch.from_numpy(all_cam_images).float()
        
        # Permute to channel-first format expected by PyTorch: [batch, channels, height, width]
        image_data = image_data.permute(0, 3, 1, 2).contiguous()
        
        print(f"Final tensor shape (before adding camera dim): {image_data.shape}")
        
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
        
        print(f"Final tensor shape (with camera dim): {image_data.shape}")
        
        # Normalize pixel values to [0,1]
        image_data = image_data / 255.0
        
        return image_data.cuda()


def main():
    parser = argparse.ArgumentParser(description='Unity VR Interface for ACT Policy')
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
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
