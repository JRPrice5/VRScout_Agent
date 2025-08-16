import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        # Import action indices for splitting actions by type
        from action_utils import ACTION_INDICES

        # Get boolean indices for clamping after normalization
        boolean_indices = ACTION_INDICES['boolean']

        # Convert numpy arrays to torch tensors and ensure float32 dtype
        action_mean = torch.tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        action_std = torch.tensor(self.norm_stats["action_std"], dtype=torch.float32)

        # Normalize all actions as in the original implementation
        action_data = (action_data - action_mean) / action_std

        # After normalization, clamp boolean actions to [0,1] range
        action_data[:, boolean_indices] = torch.clamp(action_data[:, boolean_indices], 0.0, 1.0)

        # Normalize qpos as before with explicit dtype control
        qpos_mean = torch.tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
        qpos_std = torch.tensor(self.norm_stats["qpos_std"], dtype=torch.float32)
        qpos_data = (qpos_data - qpos_mean) / qpos_std

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Concatenate instead of stack to handle different episode lengths
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    # Get example qpos from first episode
    with h5py.File(os.path.join(dataset_dir, 'episode_0.hdf5'), 'r') as root:
        example_qpos = root['/observations/qpos'][()]

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": example_qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # import custom collate function
    from custom_collate import custom_collate_fn

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    
    # Optimized DataLoader configuration for better performance
    # Increase num_workers and prefetch_factor to eliminate batch iteration bottleneck
    num_workers = min(4, os.cpu_count())  # Use multiple workers but not too many
    prefetch_factor = 4  # Prefetch more batches to keep GPU busy
    
    print(f"DataLoader config: num_workers={num_workers}, prefetch_factor={prefetch_factor}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, 
                                 num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=custom_collate_fn,
                                 persistent_workers=True)  # Keep workers alive between epochs
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, 
                               num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=custom_collate_fn,
                               persistent_workers=True)  # Keep workers alive between epochs

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class StepDataset(torch.utils.data.Dataset):
    """Per-step dataset over all timesteps across episodes.

    Yields individual (image_t, qpos_t, action_t, is_pad_t) at a single timestep t from an episode.
    This is designed for large-batch expert sampling in GAIL where batch size should
    depend on total timesteps, not number of episodes.
    """
    def __init__(self, dataset_dir, camera_names, norm_stats, num_episodes):
        super(StepDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.is_sim = None

        # Precompute episode lengths and cumulative index for O(log E) lookup
        self.episode_lengths = []
        for eid in range(num_episodes):
            ep_path = os.path.join(dataset_dir, f'episode_{eid}.hdf5')
            with h5py.File(ep_path, 'r') as root:
                if self.is_sim is None:
                    self.is_sim = bool(root.attrs['sim'])
                ep_len = int(root['/action'].shape[0])
                self.episode_lengths.append(ep_len)
        self.cum_lengths = np.cumsum([0] + self.episode_lengths)  # length num_episodes+1

        # Precompute normalization tensors
        from action_utils import ACTION_INDICES
        self.boolean_indices = ACTION_INDICES['boolean']
        self.action_mean = torch.tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        self.action_std = torch.tensor(self.norm_stats["action_std"], dtype=torch.float32)
        self.qpos_mean = torch.tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
        self.qpos_std = torch.tensor(self.norm_stats["qpos_std"], dtype=torch.float32)

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _index_to_episode(self, idx):
        # Find episode e s.t. cum[e] <= idx < cum[e+1]
        e = int(np.searchsorted(self.cum_lengths, idx, side='right') - 1)
        t = int(idx - self.cum_lengths[e])
        return e, t

    def __getitem__(self, index):
        episode_id, t = self._index_to_episode(index)
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = bool(root.attrs['sim'])
            # Observations at t
            qpos = root['/observations/qpos'][t]
            # Images at t for all cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(root[f'/observations/images/{cam_name}'][t])
            all_cam_images = np.stack(all_cam_images, axis=0)  # [K, H, W, C]

            # Action alignment: mimic EpisodicDataset alignment policy
            if is_sim:
                action = root['/action'][t]
            else:
                # hack to better align for real data
                action = root['/action'][max(0, t - 1)]

        # torch tensors
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action).float()

        # Channel-last -> channel-first: [K, C, H, W]
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        # Normalize
        action_data = (action_data - self.action_mean.squeeze(0)) / self.action_std.squeeze(0)
        # Clamp boolean action dims post-normalization
        action_data[self.boolean_indices] = torch.clamp(action_data[self.boolean_indices], 0.0, 1.0)

        qpos_data = (qpos_data - self.qpos_mean.squeeze(0)) / self.qpos_std.squeeze(0)

        # Match EpisodicDataset API shape expectations used downstream in GAIL code:
        # action -> [T=1, A]; is_pad -> [T=1] all zeros (no padding for a single step)
        action_data = action_data.unsqueeze(0)
        is_pad = torch.zeros((1,), dtype=torch.bool)

        return image_data, qpos_data, action_data, is_pad


def load_expert_step_loader(dataset_dir, num_episodes, camera_names, batch_size,
                            num_workers=None, prefetch_factor=4, shuffle=True):
    """Create a DataLoader over per-step expert samples for GAIL.

    - Allows large batch sizes (e.g., 256/512/1024) since sampling is over all timesteps.
    - Uses same normalization stats as episodic loader.
    - Returns batches of:
        image:  [B, K, C, H, W]
        qpos:   [B, Dq]
        action: [B, 1, A]
        is_pad: [B, 1]
    """
    # Reuse normalization computation over the specified episodes
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # Sensible defaults for workers
    if num_workers is None:
        num_workers = min(4, os.cpu_count()) if os.cpu_count() is not None else 2

    print(f"Expert Step DataLoader config: batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}, shuffle={shuffle}")

    dataset = StepDataset(dataset_dir=dataset_dir,
                          camera_names=camera_names,
                          norm_stats=norm_stats,
                          num_episodes=num_episodes)

    # Default collate works (fixed-size tensors per item)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=True)

    return loader, norm_stats, dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
