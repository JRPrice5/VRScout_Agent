import os
import random
import re

import h5py
import numpy as np
import torch
from act_bs.config.root import RootConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

cfg = OmegaConf.structured(RootConfig)


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        norm_stats,
        num_queries,
        tail_clip: int | None = None,
        audio_history_len: int = 1,
    ):
        super(EpisodicDataset).__init__()
        self.cfg = OmegaConf.structured(RootConfig)
        self.episode_ids = episode_ids
        pattern = re.compile(r"\.hdf5$")
        self.dataset_dir = dataset_dir
        self.episodes = [
            file for file in os.listdir(self.dataset_dir) if pattern.search(file)
        ]
        self.norm_stats = norm_stats
        self.num_queries = int(num_queries)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, self.episodes[episode_id])
        metadata, obstacles, qpos, actions = read_replay(dataset_path)
        njs = metadata[0] * 30
        jd = metadata[1] * 30

        window_len = self.cfg.model.chunk_size
        frame_count = actions.shape[0]

        frame_id = 0
        action_window = np.zeros(
            (window_len, self.cfg.data.action_length), dtype=np.float32
        )

        while not self._is_contiguous_actions(action_window) or np.all(
            action_window == 0
        ):
            # Pick random frame within replay
            frame_id = random.randint(0, frame_count - window_len)

            # Populate action window
            for i in range(window_len):
                action_window[i] = actions[frame_id + i]

        anchor_t = action_window[0, self.cfg.data.action_t_idx]
        obstacle_window, obstacle_mask = self._get_obs_window(
            obstacles, jd, njs, anchor_t
        )

        metadata = torch.from_numpy(metadata).float()
        qpos_data = torch.from_numpy(qpos[frame_id]).float()
        action_data = torch.from_numpy(action_window).float()
        obstacle_data = torch.from_numpy(obstacle_window).float()
        obstacle_data = torch.from_numpy(obstacle_mask).float()

        # Convert numpy arrays to torch tensors and ensure float32 dtype
        action_mean = torch.tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        action_std = torch.tensor(self.norm_stats["action_std"], dtype=torch.float32)

        # Normalize all actions as in the original implementation
        action_data = (action_data - action_mean) / action_std

        # Normalize qpos as before with explicit dtype control
        qpos_mean = torch.tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
        qpos_std = torch.tensor(self.norm_stats["qpos_std"], dtype=torch.float32)
        qpos_data = (qpos_data - qpos_mean) / qpos_std

        return metadata, obstacle_data, qpos_data, action_data, obstacle_mask

    def _is_contiguous_actions(self, actions):
        """Checks whether each action timestamp are within x of each other"""
        return np.all(np.diff(actions[:, 0]) > 0) and np.all(
            np.diff(actions[:, 0]) < (2 / self.cfg.model.sample_rate)
        )

    def _get_obs_window(self, obstacles, jd, njs, anchor_t):
        window = []
        spawn_t = anchor_t + jd / (2 * njs)
        max_obs = self.cfg.data.max_obstacles
        padded_window = np.zeros((max_obs, self.cfg.data.obstacle_length))

        for i in range(0, len(obstacles), 1):
            obstacle = obstacles[i]
            start_t = obstacle[self.cfg.data.obs_t_idx]
            end_t = obstacle[27]

            if start_t > spawn_t:
                return padded_window

            if end_t >= anchor_t or start_t >= anchor_t:
                window.append(obstacle)

        if len(window) > max_obs:
            raise ValueError(f"number of obstacles in window exceeded {max_obs}")

        padded_window[: len(window), :] = np.array(window)
        mask = np.ones(len(padded_window))
        mask[len(window) :] = 0
        return padded_window, mask


def read_replay(replay_path):
    print(f"reading from {replay_path}")
    with h5py.File(replay_path, "r") as root:
        replay_dset = dict(sorted(root["/actions"].items(), key=lambda x: int(x[0])))
        action = np.zeros(
            (len(replay_dset), cfg.data.episode_length, cfg.data.action_length)
        )
        for i, action_episode in enumerate(replay_dset.values()):
            action[i, :, :] = action_episode[:, :]

        obstacles = root["/obstacles"][:]
        metadata = root["/metadata"][:]

    action = np.reshape(action, (-1, cfg.data.action_length))
    qpos = np.zeros((action.shape[0], action.shape[1] - 1))
    qpos[1:, :] = action[:-1, 1:]
    qpos[0, :] = qpos[1, :]  # WARN: fills missing 1st qpos w/ 2nd
    return metadata, obstacles, qpos, action


def get_norm_stats(dataset_dir, num_episodes=None, episode_ids=None):
    """Compute normalization stats over specified episodes.

    Backward compatible: if episode_ids is None, falls back to range(num_episodes).
    """
    if episode_ids is None:
        assert num_episodes is not None, (
            "Either num_episodes or episode_ids must be provided"
        )
        ids = list(range(num_episodes))
    else:
        ids = list(episode_ids)

    all_qpos_data = []
    all_action_data = []
    for episode_idx in ids:
        replay_name = [
            filename
            for filename in os.listdir(dataset_dir)
            if filename.endswith(".hdf5")
        ][episode_idx]
        replay_path = os.path.join(dataset_dir, replay_name)
        _, _, qpos, action = read_replay(replay_path)
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

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
    }

    return stats


def load_data(
    dataset_dir,
    num_episodes,
    batch_size_train,
    batch_size_val,
    num_queries,
):
    """Create train/val DataLoaders.

    - Default: EpisodicDataset for both splits (backward compatible).
    - Normalization stats are computed on train episodes only and reused for val.
    - tail_clip: if provided, excludes the last N timesteps of each episode in both datasets.
    """
    print(f"\nData from: {dataset_dir}\n")
    # obtain train/val split at episode level
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats from TRAIN EPISODES ONLY
    norm_stats = get_norm_stats(dataset_dir, episode_ids=train_indices.tolist())

    # construct datasets (fixed window length = num_queries)
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        norm_stats,
        num_queries,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        norm_stats,
        num_queries,
    )

    # DataLoader configuration
    num_workers = min(4, os.cpu_count())  # Use multiple workers but not too many
    prefetch_factor = 4  # Prefetch more batches to keep GPU busy
    print(
        f"DataLoader config: num_workers={num_workers}, prefetch_factor={prefetch_factor}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader, norm_stats


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
