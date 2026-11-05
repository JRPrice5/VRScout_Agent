import numpy as np
import re
import torch
import os
import h5py
from torch.utils.data import DataLoader
from typing import Optional
from omegaconf import OmegaConf
from act_bs.config.root import RootConfig


# TODO: Convert to game state
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        norm_stats,
        num_queries,
        tail_clip: Optional[int] = None,
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
        qpos, action, state = read_replay(dataset_path)

        # Restrict to fixed window length = num_queries to match policy needs
        # Slice available part and pad to window_len if needed
        slice_len = min(self.cfg.model.chunk_size, self.num_queries)
        action_window = np.zeros(
            (self.num_queries, self.cfg.data.action_length), dtype=np.float32
        )
        if slice_len > 0:
            action_window[:slice_len] = action[:slice_len]
        # is_pad marks padded timesteps within this fixed window
        is_pad = np.ones((self.num_queries,), dtype=np.bool_)
        is_pad[:slice_len] = 0

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action_window).float()
        state_data = torch.from_numpy(state).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Convert numpy arrays to torch tensors and ensure float32 dtype
        action_mean = torch.tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        action_std = torch.tensor(self.norm_stats["action_std"], dtype=torch.float32)

        # Normalize all actions as in the original implementation
        action_data = (action_data - action_mean) / action_std

        # Normalize qpos as before with explicit dtype control
        qpos_mean = torch.tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
        qpos_std = torch.tensor(self.norm_stats["qpos_std"], dtype=torch.float32)
        qpos_data = (qpos_data - qpos_mean) / qpos_std

        # TODO: Don't normalise one hot dims:
        state_mean = torch.tensor(self.norm_stats["state_mean"], dtype=torch.float32)
        state_std = torch.tensor(self.norm_stats["state_std"], dtype=torch.float32)
        state_data = (state_data - state_mean) / state_std

        return state_data, qpos_data, action_data, is_pad


# TODO: Ensure state reflects metadata + obstacles
def read_replay(cfg, replay_path):
    with h5py.File(replay_path, "r") as root:
        num_frames = cfg.data.episode_length * len(root["/action"])
        qpos = np.zeros((num_frames, cfg.data.action_length))
        action = np.zeros((num_frames, cfg.data.action_length))
        for i, episode in enumerate(root["/action"].values()):
            if i != len(root) - 1:
                qpos[i + 1] = episode[:][1:]
            action[i] = episode[:]
        obstacles = root["/observations/obstacles"][:]
    state = obstacles
    return qpos, action, state


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
    all_state_data = []
    for episode_idx in ids:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        qpos, action, state = read_replay(dataset_path)
        all_state_data.append(torch.from_numpy(state))
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Concatenate instead of stack to handle different episode lengths
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    all_state_data = torch.cat(all_state_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    # normalize state data
    state_mean = all_state_data.mean(dim=0, keepdim=True)
    state_std = all_state_data.std(dim=0, keepdim=True)
    state_std = torch.clip(state_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "state_mean": state_mean.numpy().squeeze(),
        "state_std": state_std.numpy().squeeze(),
    }

    return stats


# TODO: convert data loaders to game state shards
## Determine the size of data point to train/eval on
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

    # import custom collate function
    from custom_collate import custom_collate_fn

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
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=custom_collate_fn,
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
