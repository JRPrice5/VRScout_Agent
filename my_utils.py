import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, num_queries, tail_clip: Optional[int] = None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = int(num_queries)
        # Optional: exclude last `tail_clip` timesteps of each episode from sampling
        self.tail_clip = int(tail_clip) if tail_clip is not None else 0
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
            # Effective episode length after tail clipping for action/valid start indices
            effective_len = max(0, episode_len - self.tail_clip)
            if sample_full_episode:
                # When sampling the whole episode, align start to 0 but ensure it is within the
                # non-clipped portion when possible
                start_ts = 0
            else:
                # Choose a start within the non-clipped region. If effective_len == 0, fallback to 0
                start_pool = max(1, effective_len)
                start_ts = int(np.random.choice(start_pool))
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # compute minimal action slice bounds and read only needed rows
            window_len = self.num_queries
            if is_sim:
                start = start_ts
            else:
                start = max(0, start_ts - 1)  # hack, to make timesteps more aligned
            # Do not cross into the clipped tail
            end = min(start + window_len, effective_len)
            action = root['/action'][start:end]
            available_len = end - start

        self.is_sim = is_sim
        # Restrict to fixed window length = num_queries to match policy needs
        action_dim = original_action_shape[1]
        # Slice available part and pad to window_len if needed
        slice_len = min(available_len, window_len)
        action_window = np.zeros((window_len, action_dim), dtype=np.float32)
        if slice_len > 0:
            action_window[:slice_len] = action[:slice_len]
        # is_pad marks padded timesteps within this fixed window
        is_pad = np.ones((window_len,), dtype=np.bool_)
        is_pad[:slice_len] = 0

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action_window).float()
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


def get_norm_stats(dataset_dir, num_episodes=None, episode_ids=None):
    """Compute normalization stats over specified episodes.

    Backward compatible: if episode_ids is None, falls back to range(num_episodes).
    """
    if episode_ids is None:
        assert num_episodes is not None, "Either num_episodes or episode_ids must be provided"
        ids = list(range(num_episodes))
    else:
        ids = list(episode_ids)

    all_qpos_data = []
    all_action_data = []
    for episode_idx in ids:
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

    # Get example qpos from the first selected episode
    example_ep = ids[0]
    with h5py.File(os.path.join(dataset_dir, f'episode_{example_ep}.hdf5'), 'r') as root:
        example_qpos = root['/observations/qpos'][()]

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": example_qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, num_queries,
              dataset_type: str = 'episodic', val_dataset_type: str = 'episodic', tail_clip: Optional[int] = None):
    """Create train/val DataLoaders.

    - Default: EpisodicDataset for both splits (backward compatible).
    - Optionally: use StepDataset for training and/or validation via dataset_type args.
    - Normalization stats are computed on train episodes only and reused for val.
    - tail_clip: if provided, excludes the last N timesteps of each episode in both datasets.
    """
    print(f'\nData from: {dataset_dir}\n')
    # obtain train/val split at episode level
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats from TRAIN EPISODES ONLY
    norm_stats = get_norm_stats(dataset_dir, episode_ids=train_indices.tolist())

    # import custom collate function
    from custom_collate import custom_collate_fn

    # construct datasets (fixed window length = num_queries)
    if dataset_type == 'step':
        train_dataset = StepDataset(dataset_dir=dataset_dir,
                                    camera_names=camera_names,
                                    norm_stats=norm_stats,
                                    num_episodes=len(train_indices),
                                    num_queries=num_queries,
                                    episode_ids=train_indices.tolist(),
                                    tail_clip=tail_clip)
    else:
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, num_queries,
                                        tail_clip=tail_clip)

    if val_dataset_type == 'step':
        val_dataset = StepDataset(dataset_dir=dataset_dir,
                                  camera_names=camera_names,
                                  norm_stats=norm_stats,
                                  num_episodes=len(val_indices),
                                  num_queries=num_queries,
                                  episode_ids=val_indices.tolist(),
                                  tail_clip=tail_clip)
    else:
        val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, num_queries,
                                      tail_clip=tail_clip)

    # DataLoader configuration
    num_workers = min(4, os.cpu_count())  # Use multiple workers but not too many
    prefetch_factor = 4  # Prefetch more batches to keep GPU busy
    print(f"DataLoader config: num_workers={num_workers}, prefetch_factor={prefetch_factor}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=custom_collate_fn,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True,
                                num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=custom_collate_fn,
                                persistent_workers=True)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class StepDataset(torch.utils.data.Dataset):
    """Per-step dataset over all timesteps across episodes.

    Yields (image_t, qpos_t, action_window, is_pad_window) at a timestep t from an episode.
    action_window is a chunk of length T=num_queries (default 1), starting at t (or t-1 on real
    data to align), padded as needed at the end if the episode runs out.
    This is designed for large-batch expert sampling in GAIL where batch size should depend on
    total timesteps, not number of episodes.
    """
    def __init__(self, dataset_dir, camera_names, norm_stats, num_episodes, num_queries: int = 1, episode_ids=None, tail_clip: Optional[int] = None):
        super(StepDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.num_queries = int(num_queries) if num_queries is not None else 1
        # Optional: exclude last `tail_clip` timesteps of each episode from sampling
        self.tail_clip = int(tail_clip) if tail_clip is not None else 0
        self.is_sim = None

        # Precompute episode lengths and cumulative index for O(log E) lookup
        self.episode_ids = list(range(num_episodes)) if episode_ids is None else list(episode_ids)
        self.episode_lengths = []
        for eid in self.episode_ids:
            ep_path = os.path.join(dataset_dir, f'episode_{eid}.hdf5')
            with h5py.File(ep_path, 'r') as root:
                if self.is_sim is None:
                    self.is_sim = bool(root.attrs['sim'])
                ep_len = int(root['/action'].shape[0])
                ep_eff = max(0, ep_len - self.tail_clip)
                self.episode_lengths.append(ep_eff)
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
        episode_pos, t = self._index_to_episode(index)
        episode_id = self.episode_ids[episode_pos]
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

            # Action alignment and chunking to mimic EpisodicDataset policy
            episode_len = self.episode_lengths[episode_pos]
            window_len = self.num_queries
            if is_sim:
                start = t
            else:
                # hack to better align for real data
                start = max(0, t - 1)
            # Do not cross into the clipped tail
            end = min(start + window_len, episode_len)
            action_slice = root['/action'][start:end]
            available_len = end - start

        # torch tensors
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        # Build fixed-size window [T, A] with padding if needed
        action_dim = int(self.action_mean.numel())
        action_window = np.zeros((self.num_queries, action_dim), dtype=np.float32)
        if available_len > 0:
            action_window[:available_len] = action_slice[:available_len]
        action_data = torch.from_numpy(action_window).float()

        # Channel-last -> channel-first: [K, C, H, W]
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        # Normalize
        action_data = (action_data - self.action_mean.squeeze(0)) / self.action_std.squeeze(0)
        # Clamp boolean action dims post-normalization over time dimension
        action_data[:, self.boolean_indices] = torch.clamp(action_data[:, self.boolean_indices], 0.0, 1.0)

        qpos_data = (qpos_data - self.qpos_mean.squeeze(0)) / self.qpos_std.squeeze(0)

        # Match EpisodicDataset API shape expectations used downstream in GAIL code:
        # action -> [T, A]; is_pad -> [T]
        is_pad = torch.zeros((self.num_queries,), dtype=torch.bool)
        if available_len < self.num_queries:
            is_pad[available_len:] = True

        return image_data, qpos_data, action_data, is_pad


def load_expert_step_loader(dataset_dir, num_episodes, camera_names, batch_size,
                            num_workers=None, prefetch_factor=4, shuffle=True):
    """Create a DataLoader over per-step expert samples for GAIL.

    - Allows large batch sizes (e.g., 256/512/1024) since sampling is over all timesteps.
    - Uses same normalization stats as episodic loader.
    - Returns batches of:
        image:  [B, K, C, H, W]
        qpos:   [B, Dq]
        action: [B, T, A]
        is_pad: [B, T]
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
