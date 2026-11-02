import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from my_utils import load_data  # data functions
from my_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy
from visualize_episodes import save_videos


def main(cfg):
    set_seed(cfg.evaluation.seed)
    ckpt_dir = cfg.evaluation.checkpoint_dir
    batch_size_train = cfg.evaluation.batch_size
    batch_size_val = cfg.evaluation.batch_size
    dataset_dir = cfg.data.hdf5.dataset_dir
    num_episodes = (
        cfg.evaluation.num_episodes
        if cfg.evaluation.num_episodes != -1
        else len(os.listdir(dataset_dir))
    )

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        batch_size_train,
        batch_size_val,
        cfg.model.chunk_size,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # Define checkpoint path for pretrained policy
    pretrained_ckpt_path = cfg.evaluation.pretrained_ckpt_path

    # Evaluate loss on train dataset
    print("\nEvaluating on training dataset:")
    train_loss_summary = evaluate_bc_loss(train_dataloader, cfg, pretrained_ckpt_path)

    # Evaluate loss on validation dataset
    print("\nEvaluating on validation dataset:")
    val_loss_summary = evaluate_bc_loss(val_dataloader, cfg, pretrained_ckpt_path)

    print(f"\nEvaluation complete.")
    print(f"Train loss: {train_loss_summary['loss']:.6f}")
    print(f"Validation loss: {val_loss_summary['loss']:.6f}")


def make_policy(policy_class, cfg):
    if policy_class == "ACT":
        policy = ACTPolicy(cfg)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    state_data, qpos_data, action_data, is_pad = data
    state_data, qpos_data, action_data, is_pad = (
        state_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, state_data, action_data, is_pad)


def evaluate_bc_loss(dataloader, config, ckpt_path=None):
    """
    Load a pretrained policy and compute the loss on the dataset without training.

    Args:
        dataloader: DataLoader for the dataset
        config: Configuration dictionary
        ckpt_path: Path to the checkpoint file to load

    Returns:
        loss_summary: Dictionary with mean loss values
    """
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    # Create policy
    policy = make_policy(policy_class, policy_config)
    policy.cuda()

    # Load pretrained weights if provided
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
        policy.load_state_dict(torch.load(ckpt_path))
    else:
        print("No checkpoint provided, using randomly initialized policy")

    # Evaluate on dataset
    policy.eval()
    all_dicts = []

    print("Computing loss on dataset...")
    with torch.inference_mode():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            forward_dict = forward_pass(data, policy)
            all_dicts.append(detach_dict(forward_dict))

    # Compute mean loss
    loss_summary = compute_dict_mean(all_dicts)

    # Print results
    print(f"\nEvaluation Results:")
    print(f"Total loss: {loss_summary['loss']:.5f}")
    summary_string = ""
    for k, v in loss_summary.items():
        summary_string += f"{k}: {v.item():.3f} "
    print(summary_string)

    return loss_summary


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)
    parser.add_argument(
        "--backbone",
        action="store",
        type=str,
        help="Backbone model (resnet18, resnet34, resnet50, dino_vits16, dino_vitb16)",
        default="resnet18",
    )
    parser.add_argument(
        "--lr_backbone", action="store", type=float, help="lr_backbone", required=False
    )
    parser.add_argument(
        "--pretrained_ckpt_path",
        action="store",
        type=str,
        help="Path to pretrained policy checkpoint",
        required=False,
    )

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument(
        "--activation",
        action="store",
        type=str,
        help="activation for transformer",
        required=False,
    )

    main(vars(parser.parse_args()))
