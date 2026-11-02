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
        self.episodes = [
            file for file in os.listdir(self.dataset_dir) if pattern.search(file)
        ]
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.num_queries = int(num_queries)
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, self.episodes[episode_id])
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/action"][0:]
            window_len = self.num_queries
            action = root["/action"][1:]

        # Restrict to fixed window length = num_queries to match policy needs
        action_dim = root["/action"].shape[1]
        # Slice available part and pad to window_len if needed
        slice_len = min(self.cfg.episode_len, window_len)
        action_window = np.zeros((window_len, action_dim), dtype=np.float32)
        if slice_len > 0:
            action_window[:slice_len] = action[:slice_len]
        # is_pad marks padded timesteps within this fixed window
        is_pad = np.ones((window_len,), dtype=np.bool_)
        is_pad[:slice_len] = 0

        # TODO: fetch game state data from folder
        # construct observations
        # game_data = torch.from_numpy(load from folder).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action_window).float()
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

        return state_data, qpos_data, action_data, is_pad


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
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            # TODO: get qpos from action
            action_shape = root["/action"].shape[1:]
            qpos = root["/action"][1:].append(np.zeros((action_shape)))
            action = root["/action"][0:]
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
    with h5py.File(
        os.path.join(dataset_dir, f"episode_{example_ep}.hdf5"), "r"
    ) as root:
        example_qpos = root["/observations/qpos"][()]

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": example_qpos,
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

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


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
