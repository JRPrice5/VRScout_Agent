#!/usr/bin/env python
"""
Example script for running ACT RL fine-tuning with PPO and KL-to-BC constraint.

This script demonstrates how to set up and run reinforcement learning fine-tuning
of a pre-trained ACT model using the Unity VR environment.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add the parent directory to path to import ACT modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.unity_vr_rl_interface import UnityVRRLInterface
from rl.training_utils import TrainingLogger, MemoryMonitor, EarlyStoppingMonitor, create_training_config


def main():
    parser = argparse.ArgumentParser(description='ACT RL Fine-tuning Example')
    
    # Required arguments
    parser.add_argument('--pretrained_ckpt_path', type=str, required=True, 
                       help='Path to pretrained ACT model checkpoint (e.g., policy_best.ckpt)')
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                       help='Directory containing pretrained model and dataset stats')
    
    # Basic configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment (auto-generated if not provided)')
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'])
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=5557, help='Server port for RL training')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['camera1'], help='Camera names')
    
    # Model configuration (should match pretrained model)
    parser.add_argument('--task_name', type=str, default='rl_task', help='Task name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs (legacy parameter)')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for ACT policy')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for transformer')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Backbone learning rate')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model')
    parser.add_argument('--enc_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='Number of decoder layers')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--query_frequency', type=int, default=1, help='Policy query frequency')
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    
    # RL-specific configuration
    parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum episode length')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum training episodes')
    parser.add_argument('--update_frequency', type=int, default=2048, help='Policy update frequency (steps)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PPO updates')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    
    # KL constraint configuration
    parser.add_argument('--kl_weight', type=float, default=0.05, help='Initial KL constraint weight')
    parser.add_argument('--min_kl_weight', type=float, default=0.001, help='Minimum KL weight')
    parser.add_argument('--kl_decay_factor', type=float, default=0.95, help='KL weight decay factor')
    parser.add_argument('--kl_decay_steps', type=int, default=100, help='Steps between KL weight decay')
    
    # Training management
    parser.add_argument('--save_frequency', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--plot_frequency', type=int, default=50, help='Plot curves every N episodes')
    parser.add_argument('--max_memory_mb', type=int, default=8000, help='Maximum GPU memory usage (MB)')
    parser.add_argument('--early_stopping_patience', type=int, default=50, 
                       help='Early stopping patience (episodes)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.01,
                       help='Minimum improvement for early stopping')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./rl_logs', help='Directory for logging')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pretrained_ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_ckpt_path}")
    
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"act_rl_{timestamp}"
    
    print(f"Starting ACT RL Fine-tuning Experiment: {args.experiment_name}")
    print(f"Pretrained model: {args.pretrained_ckpt_path}")
    print(f"Server will run on {args.host}:{args.port}")
    
    # Create training configuration
    rl_config = create_training_config(args)
    
    # Initialize training utilities
    logger = TrainingLogger(args.log_dir, args.experiment_name)
    memory_monitor = MemoryMonitor(args.max_memory_mb)
    early_stopping = EarlyStoppingMonitor(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        monitor='reward'
    )
    
    # Log experiment configuration
    logger.log_system_event('EXPERIMENT_START', 'Starting RL fine-tuning experiment', {
        'args': vars(args),
        'rl_config': rl_config
    })
    
    try:
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
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.log_system_event('RESUME_CHECKPOINT', f'Resuming from {args.resume_from}')
            interface.load_model(args.resume_from)
        
        # Start training
        logger.log_system_event('TRAINING_START', 'Starting RL training mode')
        interface.start_rl_training()
        
        # Start the server (this will block and handle Unity connections)
        logger.log_system_event('SERVER_START', f'Starting server on {args.host}:{args.port}')
        print("\n" + "="*60)
        print("RL TRAINING SERVER READY")
        print("="*60)
        print(f"Experiment: {args.experiment_name}")
        print(f"Server: {args.host}:{args.port}")
        print(f"Max Episodes: {args.max_episodes}")
        print(f"Update Frequency: {args.update_frequency} steps")
        print(f"Initial KL Weight: {args.kl_weight}")
        print("="*60)
        print("\nWaiting for Unity VR client connection...")
        print("Connect your Unity VR environment to start training.")
        print("Press Ctrl+C to stop the server.\n")
        
        # Start the server (this will run until interrupted)
        interface.start_server()
        
    except KeyboardInterrupt:
        logger.log_system_event('TRAINING_INTERRUPTED', 'Training interrupted by user')
        print("\nTraining interrupted by user")
        
    except Exception as e:
        logger.log_system_event('TRAINING_ERROR', f'Training error: {str(e)}', {'error': str(e)})
        print(f"Training error: {e}")
        raise
        
    finally:
        # Final cleanup and logging
        if 'interface' in locals():
            # Save final model
            final_model_path = os.path.join(logger.experiment_dir, 'checkpoints', 'final_model.ckpt')
            interface.save_model(final_model_path)
            logger.log_system_event('MODEL_SAVED', f'Final model saved to {final_model_path}')
            
            # Generate final plots
            logger.plot_training_curves(save_plot=True, show_plot=False)
            
            # Log final statistics
            final_stats = logger.get_summary_stats()
            logger.log_system_event('EXPERIMENT_END', 'Experiment completed', final_stats)
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED")
            print("="*60)
            print(f"Total Episodes: {final_stats.get('total_episodes', 0)}")
            print(f"Mean Reward: {final_stats.get('mean_reward', 0.0):.3f}")
            print(f"Recent Mean Reward: {final_stats.get('recent_mean_reward', 0.0):.3f}")
            print(f"Experiment Directory: {logger.experiment_dir}")
            print("="*60)
            
            # Stop the interface
            interface.stop_server()


if __name__ == "__main__":
    main()
