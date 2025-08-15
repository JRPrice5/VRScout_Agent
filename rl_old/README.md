# ACT Reinforcement Learning Fine-tuning

This module implements reinforcement learning fine-tuning for pre-trained ACT (Action Chunking with Transformers) models using Proximal Policy Optimization (PPO) with KL divergence constraint to the original behavior cloning policy.

## Overview

The RL fine-tuning system allows you to improve a pre-trained ACT model through interaction with the Unity VR environment while preventing excessive deviation from the original behavior using a KL constraint (KL-to-BC).

## Key Features

- **PPO with KL Constraint**: Uses PPO algorithm with KL divergence constraint to the original behavior cloning policy
- **Memory Efficient**: Optimized for memory usage during training
- **Unity VR Integration**: Seamlessly integrates with the existing Unity VR environment
- **Reward Matching**: Handles reward collection from Unity client and matches them to training steps
- **KL Scheduling**: Implements adjustable KL constraint scheduling during training
- **Comprehensive Logging**: Detailed logging and visualization of training progress
- **Checkpointing**: Save and resume training from checkpoints

## Files Structure

```
rl/
├── ppo_agent.py              # PPO agent implementation with KL constraint
├── unity_vr_rl_interface.py  # Unity VR interface for RL training
├── training_utils.py         # Training utilities and logging
├── train_rl_example.py       # Example training script
└── README.md                 # This file
```

## Quick Start

### 1. Prerequisites

- Pre-trained ACT model checkpoint (`policy_best.ckpt`)
- Dataset statistics file (`dataset_stats.pkl`)
- Unity VR environment configured for RL training

### 2. Basic Usage

```bash
python rl/train_rl_example.py \
    --pretrained_ckpt_path /path/to/policy_best.ckpt \
    --ckpt_dir /path/to/checkpoint/directory \
    --experiment_name my_rl_experiment \
    --max_episodes 1000 \
    --kl_weight 0.05
```

### 3. Unity Client Integration

The Unity VR client should send messages in the following format:

#### State Message (for action inference)
```json
{
    "type": "state",
    "data": {
        "qpos": [0.1, 0.2, ...],
        "image": "base64_encoded_image_data",
        "timestep": 123
    }
}
```

#### Episode Data Message (after episode completion)
```json
{
    "type": "episode_data",
    "data": {
        "timesteps": [120, 121, 122, ...],
        "rewards": [0.1, 0.2, 0.5, ...],
        "done": true
    }
}
```

## Configuration

### RL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kl_weight` | 0.05 | Initial KL constraint weight |
| `min_kl_weight` | 0.001 | Minimum KL weight during decay |
| `kl_decay_factor` | 0.95 | Factor for KL weight decay |
| `kl_decay_steps` | 100 | Steps between KL weight updates |
| `clip_param` | 0.2 | PPO clipping parameter |
| `value_coef` | 0.5 | Value loss coefficient |
| `entropy_coef` | 0.01 | Entropy regularization coefficient |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda parameter |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_episodes` | 1000 | Maximum training episodes |
| `update_frequency` | 2048 | Update policy every N steps |
| `batch_size` | 64 | Batch size for PPO updates |
| `n_epochs` | 10 | PPO epochs per update |
| `save_frequency` | 100 | Save model every N episodes |

## Advanced Usage

### Custom RL Configuration

```python
from rl.unity_vr_rl_interface import UnityVRRLInterface

# Custom RL configuration
rl_config = {
    'kl_weight': 0.1,
    'min_kl_weight': 0.005,
    'clip_param': 0.15,
    'update_frequency': 1024,
    'max_episodes': 2000
}

# Create interface with custom config
interface = UnityVRRLInterface(
    pretrained_ckpt_path='path/to/model.ckpt',
    rl_config=rl_config,
    # ... other parameters
)
```

### Manual Training Loop

```python
from rl.ppo_agent import PPOACTPolicy, PPOMemory
from rl.training_utils import TrainingLogger

# Load pretrained policy
pretrained_policy = load_pretrained_policy('path/to/model.ckpt')

# Create PPO policy
ppo_policy = PPOACTPolicy(pretrained_policy, kl_weight=0.05)

# Create memory buffer
memory = PPOMemory(batch_size=64)

# Training loop
for episode in range(max_episodes):
    # Collect episode data
    # ... (interact with environment)
    
    # Update policy
    if len(memory.states) >= update_frequency:
        stats = ppo_policy.update(memory)
        print(f"Episode {episode}: {stats}")
```

### Monitoring and Logging

The system provides comprehensive logging and monitoring:

```python
from rl.training_utils import TrainingLogger, MemoryMonitor

# Initialize logger
logger = TrainingLogger('./logs', 'my_experiment')

# Log episode
logger.log_episode(episode=10, reward=1.5, length=200, timestep=2000)

# Log training step
logger.log_training_step(episode=10, stats={'policy_loss': 0.1, 'kl_div': 0.05})

# Generate plots
logger.plot_training_curves(save_plot=True)

# Monitor memory usage
memory_monitor = MemoryMonitor(max_memory_mb=8000)
if not memory_monitor.check_memory():
    memory_monitor.cleanup_memory()
```

## KL Constraint Scheduling

The KL constraint weight can be scheduled during training to gradually reduce the constraint:

```python
from rl.ppo_agent import KLScheduler

scheduler = KLScheduler(
    initial_kl_weight=0.05,
    min_kl_weight=0.001,
    decay_factor=0.95,
    decay_steps=100
)

# Update KL weight
new_kl_weight = scheduler.step()
ppo_policy.kl_weight = new_kl_weight
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: If you encounter GPU memory issues, try:
   - Reducing `batch_size`
   - Reducing `update_frequency`
   - Using `memory_monitor.cleanup_memory()`

2. **KL Divergence Too High**: If KL divergence becomes too high:
   - Increase `kl_weight`
   - Decrease `kl_decay_factor`
   - Increase `kl_decay_steps`

3. **Training Instability**: If training is unstable:
   - Reduce learning rate
   - Increase `clip_param`
   - Reduce `entropy_coef`

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ALOHA/act
export CUDA_VISIBLE_DEVICES=0
python rl/train_rl_example.py --debug
```

## Performance Tips

1. **Memory Efficiency**:
   - Process episodes in chunks
   - Clear GPU cache regularly
   - Use appropriate batch sizes

2. **Training Speed**:
   - Use appropriate `update_frequency`
   - Optimize Unity VR environment performance
   - Use GPU acceleration

3. **Stability**:
   - Start with higher KL weight and decay gradually
   - Monitor KL divergence and adjust if needed
   - Use early stopping to prevent overfitting

## Integration with Unity VR

The RL interface extends the existing Unity VR inference interface. Key differences:

1. **Port**: Uses port 5557 by default (different from inference port 5556)
2. **Message Types**: Supports additional message types for episode data
3. **Training Mode**: Can switch between training and inference modes
4. **Reward Handling**: Processes rewards sent from Unity client

### Unity Client Changes Required

Your Unity VR client needs to:

1. Send episode completion messages with rewards
2. Handle training/inference mode switching
3. Connect to the RL training port (5557)

## Citation

If you use this RL fine-tuning system in your research, please cite:

```bibtex
@article{act_rl_finetuning,
    title={Reinforcement Learning Fine-tuning for Action Chunking with Transformers},
    author={Your Name},
    year={2024}
}
```

## License

This project is licensed under the same license as the original ACT implementation.
