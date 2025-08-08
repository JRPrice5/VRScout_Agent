import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
import psutil
import GPUtil

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from my_utils import load_data # data functions
from my_utils import sample_box_pose, sample_insertion_pose # robot functions
from my_utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 32  # Updated from 14 to match actual dimension
    lr_backbone = args.get('lr_backbone', 1e-6)
    # Use backbone from args if provided, otherwise default to resnet18
    backbone = args.get('backbone', 'resnet18')
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    forward_start_detailed = time.time()
    
    image_data, qpos_data, action_data, is_pad = data
    
    # Time GPU transfer
    gpu_transfer_start = time.time()
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    gpu_transfer_time = time.time() - gpu_transfer_start
    
    # Time policy forward pass
    policy_forward_start = time.time()
    result = policy(qpos_data, image_data, action_data, is_pad)
    policy_forward_time = time.time() - policy_forward_start
    
    total_forward_time = time.time() - forward_start_detailed
    
    # Print detailed timing for first few batches of first epoch
    if hasattr(forward_pass, 'call_count'):
        forward_pass.call_count += 1
    else:
        forward_pass.call_count = 1
        
    if forward_pass.call_count <= 5:
        print(f"[ForwardPass] Call {forward_pass.call_count} detailed timing:")
        print(f"  GPU transfer: {gpu_transfer_time:.4f}s")
        print(f"  Policy forward: {policy_forward_time:.4f}s")
        print(f"  Total forward: {total_forward_time:.4f}s")
        print(f"  Batch shape: {image_data.shape}")
    
    return result


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # Profiling setup
    profile_dir = os.path.join(ckpt_dir, 'profiling')
    os.makedirs(profile_dir, exist_ok=True)
    
    # Initialize profiling metrics
    profiling_data = {
        'epoch_times': [],
        'train_times': [],
        'val_times': [],
        'forward_times': [],
        'backward_times': [],
        'optimizer_times': [],
        'data_loading_times': [],
        'dataloader_overhead': [],
        'gpu_memory_usage': [],
        'gpu_utilization': [],
        'cpu_usage': []
    }
    
    print(f"Starting profiling - results will be saved to {profile_dir}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    # Profile first 3 epochs in detail
    profile_epochs = min(3, num_epochs)
    
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch}')
        
        # GPU memory and utilization tracking
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
            try:
                gpus = GPUtil.getGPUs()
                gpu_util = gpus[0].load * 100 if gpus else 0
            except:
                gpu_util = 0
        else:
            gpu_memory_before = 0
            gpu_util = 0
            
        cpu_percent = psutil.cpu_percent()
        
        # Validation with timing
        val_start_time = time.time()
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            val_data_loading_time = 0
            val_forward_time = 0
            
            for batch_idx, data in enumerate(val_dataloader):
                data_start = time.time()
                # Data is already loaded, measure forward pass
                data_loading_time = time.time() - data_start
                val_data_loading_time += data_loading_time
                
                forward_start = time.time()
                forward_dict = forward_pass(data, policy)
                forward_time = time.time() - forward_start
                val_forward_time += forward_time
                
                epoch_dicts.append(forward_dict)
                
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                
        val_time = time.time() - val_start_time
        print(f'Val loss:   {epoch_val_loss:.5f} (Time: {val_time:.2f}s)')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Training with detailed profiling
        train_start_time = time.time()
        policy.train()
        optimizer.zero_grad()
        
        train_data_loading_time = 0
        train_forward_time = 0
        train_backward_time = 0
        train_optimizer_time = 0
        
        # Use torch profiler for detailed analysis on first few epochs
        if epoch < profile_epochs:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(profile_dir, f'epoch_{epoch}')
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()
        else:
            profiler = None
            
        batch_iter_start = time.time()
        dataloader_overhead = 0
        
        for batch_idx, data in enumerate(train_dataloader):
            # Measure time between batches (dataloader overhead)
            if batch_idx > 0:
                batch_ready_time = time.time()
                dataloader_overhead += batch_ready_time - batch_iter_start
            
            # Data loading timing (data transfer to GPU)
            data_transfer_start = time.time()
            # Data is already loaded by dataloader, measure GPU transfer in forward_pass
            data_loading_time = time.time() - data_transfer_start
            train_data_loading_time += data_loading_time
            
            # Forward pass timing (includes GPU transfer)
            forward_start = time.time()
            forward_dict = forward_pass(data, policy)
            forward_time = time.time() - forward_start
            train_forward_time += forward_time
            
            # Backward pass timing
            backward_start = time.time()
            loss = forward_dict['loss']
            loss.backward()
            backward_time = time.time() - backward_start
            train_backward_time += backward_time
            
            # Optimizer step timing
            optimizer_start = time.time()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_time = time.time() - optimizer_start
            train_optimizer_time += optimizer_time
            
            train_history.append(detach_dict(forward_dict))
            
            if profiler:
                profiler.step()
                
            # Print detailed timing for first epoch, first few batches
            if epoch == 0 and batch_idx < 3:
                print(f"  Batch {batch_idx}: Forward={forward_time:.3f}s, "
                      f"Backward={backward_time:.3f}s, Optimizer={optimizer_time:.3f}s")
                      
            # Prepare for next iteration timing
            batch_iter_start = time.time()
                      
        if profiler:
            profiler.stop()
            
        train_time = time.time() - train_start_time
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f} (Time: {train_time:.2f}s)')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
        # GPU memory after training
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
            try:
                gpus = GPUtil.getGPUs()
                gpu_util_after = gpus[0].load * 100 if gpus else 0
            except:
                gpu_util_after = 0
        else:
            gpu_memory_after = 0
            gpu_util_after = 0
            
        epoch_time = time.time() - epoch_start_time
        
        # Store profiling data
        profiling_data['epoch_times'].append(epoch_time)
        profiling_data['train_times'].append(train_time)
        profiling_data['val_times'].append(val_time)
        profiling_data['forward_times'].append(train_forward_time)
        profiling_data['backward_times'].append(train_backward_time)
        profiling_data['optimizer_times'].append(train_optimizer_time)
        profiling_data['data_loading_times'].append(train_data_loading_time)
        profiling_data['dataloader_overhead'].append(dataloader_overhead)
        profiling_data['gpu_memory_usage'].append(gpu_memory_after)
        profiling_data['gpu_utilization'].append(gpu_util_after)
        profiling_data['cpu_usage'].append(cpu_percent)
        
        # Print detailed timing breakdown
        print(f"Timing breakdown - Total: {epoch_time:.2f}s, "
              f"Train: {train_time:.2f}s, Val: {val_time:.2f}s")
        print(f"  Forward: {train_forward_time:.2f}s, "
              f"Backward: {train_backward_time:.2f}s, "
              f"Optimizer: {train_optimizer_time:.2f}s")
        print(f"  DataLoader overhead: {dataloader_overhead:.2f}s "
              f"({dataloader_overhead/train_time*100:.1f}% of train time)")
        print(f"GPU Memory: {gpu_memory_after:.2f}GB, "
              f"GPU Util: {gpu_util_after:.1f}%, CPU: {cpu_percent:.1f}%")

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
            
            # Save profiling data
            profiling_path = os.path.join(profile_dir, f'profiling_data_epoch_{epoch}.pkl')
            with open(profiling_path, 'wb') as f:
                pickle.dump(profiling_data, f)

    # Final profiling summary
    print("\n" + "="*50)
    print("PROFILING SUMMARY")
    print("="*50)
    avg_epoch_time = np.mean(profiling_data['epoch_times'])
    avg_train_time = np.mean(profiling_data['train_times'])
    avg_val_time = np.mean(profiling_data['val_times'])
    avg_forward_time = np.mean(profiling_data['forward_times'])
    avg_backward_time = np.mean(profiling_data['backward_times'])
    avg_optimizer_time = np.mean(profiling_data['optimizer_times'])
    avg_dataloader_overhead = np.mean(profiling_data['dataloader_overhead'])
    avg_gpu_memory = np.mean(profiling_data['gpu_memory_usage'])
    avg_gpu_util = np.mean(profiling_data['gpu_utilization'])
    avg_cpu_util = np.mean(profiling_data['cpu_usage'])
    
    # Calculate unaccounted time
    accounted_time = avg_forward_time + avg_backward_time + avg_optimizer_time + avg_dataloader_overhead
    unaccounted_time = avg_train_time - accounted_time
    
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    print(f"Average train time: {avg_train_time:.2f}s ({avg_train_time/avg_epoch_time*100:.1f}%)")
    print(f"Average val time: {avg_val_time:.2f}s ({avg_val_time/avg_epoch_time*100:.1f}%)")
    print(f"Training breakdown:")
    print(f"  Forward pass: {avg_forward_time:.2f}s ({avg_forward_time/avg_train_time*100:.1f}%)")
    print(f"  Backward pass: {avg_backward_time:.2f}s ({avg_backward_time/avg_train_time*100:.1f}%)")
    print(f"  Optimizer step: {avg_optimizer_time:.2f}s ({avg_optimizer_time/avg_train_time*100:.1f}%)")
    print(f"  DataLoader overhead: {avg_dataloader_overhead:.2f}s ({avg_dataloader_overhead/avg_train_time*100:.1f}% of train time)")
    print(f"  Unaccounted time: {unaccounted_time:.2f}s ({unaccounted_time/avg_train_time*100:.1f}%)")
    print(f"Average GPU memory usage: {avg_gpu_memory:.2f}GB")
    print(f"Average GPU utilization: {avg_gpu_util:.1f}%")
    print(f"Average CPU utilization: {avg_cpu_util:.1f}%")
    
    # Identify bottlenecks
    print("\nBOTTLENECK ANALYSIS:")
    if avg_gpu_util < 50:
        print("⚠️  LOW GPU UTILIZATION - GPU is underutilized")
    if avg_forward_time / avg_train_time > 0.6:
        print("⚠️  FORWARD PASS BOTTLENECK - Model inference is slow")
    if avg_backward_time / avg_train_time > 0.3:
        print("⚠️  BACKWARD PASS BOTTLENECK - Gradient computation is slow")
    if avg_optimizer_time / avg_train_time > 0.2:
        print("⚠️  OPTIMIZER BOTTLENECK - Parameter updates are slow")
    if avg_dataloader_overhead / avg_train_time > 0.4:
        print("⚠️  DATALOADER BOTTLENECK - Data loading/preprocessing is slow")
    if unaccounted_time / avg_train_time > 0.3:
        print("⚠️  UNACCOUNTED TIME - Significant time spent in unmeasured operations")
    
    # Save final profiling data
    final_profiling_path = os.path.join(profile_dir, 'final_profiling_summary.pkl')
    with open(final_profiling_path, 'wb') as f:
        pickle.dump(profiling_data, f)
    
    print(f"\nDetailed profiling data saved to: {profile_dir}")
    print(f"TensorBoard traces available for first {profile_epochs} epochs")
    print("="*50)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--backbone', action='store', type=str, help='Backbone model (resnet18, resnet34, resnet50, dino_vits16, dino_vitb16)', default='resnet18')
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', required=False)
    
    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
