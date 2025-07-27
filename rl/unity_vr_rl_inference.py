import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from unity_vr_inference import UnityVRInterface
from policy import ACTPolicy

class UnityVRRLInference(UnityVRInterface):
    """
    Extension of UnityVRInterface to load and run RL-trained models.
    """
    
    def _load_policy(self):
        """Override to load policy from PPO checkpoint format"""
        # Load the checkpoint
        checkpoint_path = os.path.join(self.ckpt_dir, f'policy_best.ckpt')
        if os.path.exists(checkpoint_path):
            print(f"Loading policy from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Check if this is a PPO checkpoint (has policy_state_dict)
            if 'policy_state_dict' in checkpoint:
                print("Detected PPO checkpoint format")
                # Create policy
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
                    # Load state dict from policy_state_dict
                    policy.model.load_state_dict(checkpoint['policy_state_dict'])
                    print("Successfully loaded PPO policy")
                    return policy
                else:
                    raise ValueError(f"Unsupported policy class for PPO: {self.policy_class}")
            else:
                # Fall back to parent implementation for regular checkpoints
                return super()._load_policy()
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

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
    
    interface = UnityVRRLInference(
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