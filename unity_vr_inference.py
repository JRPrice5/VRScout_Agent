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
                 ckpt_path, 
                 stats_path, 
                 policy_class='ACT',
                 host='127.0.0.1', 
                 port=5555,
                 camera_names=['camera1'],
                 temporal_agg=False):
        """
        Initialize the interface.
        
        Args:
            ckpt_path: Path to the trained policy checkpoint
            stats_path: Path to the dataset statistics
            policy_class: Type of policy ('ACT' or 'CNNMLP')
            host: Host address for the server
            port: Port for the server
            camera_names: List of camera names
            temporal_agg: Whether to use temporal aggregation
        """
        self.host = host
        self.port = port
        self.camera_names = camera_names
        self.temporal_agg = temporal_agg
        
        # Load dataset statistics
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Set up preprocessing and postprocessing functions
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        
        # Load policy
        self.policy_class = policy_class
        self.policy = self._load_policy(ckpt_path)
        
        # Initialize temporal aggregation variables if needed
        if self.temporal_agg:
            self.state_dim = len(self.stats['qpos_mean'])
            self.all_time_actions = None
            self.max_timesteps = 1000  # Can be adjusted based on expected episode length
            self.num_queries = 16  # Default from ACT, can be adjusted
        
        self.timestep = 0
        self.query_frequency = 16  # Default from ACT, can be adjusted
        
        # Initialize socket
        self.socket = None
        self.client_socket = None
        self.running = False
    
    def _load_policy(self, ckpt_path):
        """Load the trained policy from checkpoint"""
        print(f"Loading policy from {ckpt_path}")
        
        # Determine policy configuration based on policy class
        if self.policy_class == 'ACT':
            # Default ACT policy configuration, adjust as needed
            policy_config = {
                'num_queries': 16,
                'kl_weight': 5,
                'hidden_dim': 512,
                'dim_feedforward': 2048,
                'chunk_size': 64,
            }
            policy = ACTPolicy(policy_config)
        elif self.policy_class == 'CNNMLP':
            policy_config = {
                'state_dim': len(self.stats['qpos_mean']),
                'action_dim': len(self.stats['action_mean']),
            }
            policy = CNNMLPPolicy(policy_config)
        else:
            raise ValueError(f"Unsupported policy class: {self.policy_class}")
        
        # Load checkpoint
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        policy.eval()
        
        return policy
    
    def start_server(self):
        """Start the server to listen for Unity connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Server started on {self.host}:{self.port}")
            
            self.running = True
            while self.running:
                print("Waiting for Unity client to connect...")
                self.client_socket, addr = self.socket.accept()
                print(f"Connected to Unity client at {addr}")
                
                # Handle client in a separate thread
                client_thread = threading.Thread(target=self._handle_client)
                client_thread.daemon = True
                client_thread.start()
                client_thread.join()
                
        except KeyboardInterrupt:
            print("Server stopped by user")
        except Exception as e:
            print(f"Error in server: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the server"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        print("Server stopped")
    
    def _handle_client(self):
        """Handle communication with a connected Unity client"""
        try:
            # Reset timestep counter when a new client connects
            self.timestep = 0
            if self.temporal_agg:
                self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim]).cuda()
            
            buffer_size = 4096
            data_buffer = b""
            
            while self.running:
                # Receive data from Unity
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
        
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            if self.client_socket:
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
                if self.timestep % self.query_frequency == 0:
                    all_actions = self.policy(qpos, curr_image)
                
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
                    raw_action = all_actions[:, self.timestep % self.query_frequency]
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
        # Check if we have multiple cameras or just one
        if 'images' in state_data:
            images_dict = state_data['images']
            all_cam_images = []
            
            for cam_name in self.camera_names:
                if cam_name in images_dict:
                    # Decode base64 image
                    img_data = base64.b64decode(images_dict[cam_name])
                    img = Image.open(io.BytesIO(img_data))
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
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            all_cam_images = [img_array]
        
        # Stack images and convert to tensor
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images).float()
        
        # Convert to channel-first format (N, C, H, W)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        # Normalize image
        image_data = image_data / 255.0
        
        return image_data.cuda()


def main():
    parser = argparse.ArgumentParser(description='Unity VR Interface for ACT Policy')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to policy checkpoint')
    parser.add_argument('--stats_path', type=str, required=True, help='Path to dataset statistics')
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['camera1'], help='Camera names')
    
    args = parser.parse_args()
    
    interface = UnityVRInterface(
        ckpt_path=args.ckpt_path,
        stats_path=args.stats_path,
        policy_class=args.policy_class,
        host=args.host,
        port=args.port,
        camera_names=args.camera_names,
        temporal_agg=args.temporal_agg
    )
    
    interface.start_server()


if __name__ == "__main__":
    main()
