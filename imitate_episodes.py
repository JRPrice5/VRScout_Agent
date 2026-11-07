import torch
import numpy as np
import os
import pickle
import re
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
from act_bs.config.root import RootConfig

from my_utils import load_data  # data functions
from my_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy


def main(cfg):
    set_seed(1)
    # command line parameters
    # is_eval = args["eval"]
    ckpt_dir = cfg.training.checkpoint_dir
    batch_size_train = cfg.training.batch_size
    batch_size_val = cfg.validation.batch_size

    # task_config = TASK_CONFIGS[task_name]
    dataset_dir = cfg.data.hdf5_dir
    pattern = re.compile(r"\.hdf5$")
    num_episodes = (
        cfg.training.num_episodes
        if cfg.training.num_episodes != -1
        else len([file for file in os.listdir(dataset_dir) if pattern.search(file)])
    )

    train_dataloader, val_dataloader, stats = load_data(
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

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, cfg)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
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
    return policy(qpos_data, state_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, cfg):
    num_epochs = cfg.training.epochs
    ckpt_dir = cfg.training.checkpoint_dir
    seed = cfg.training.seed
    policy_class = cfg.model.policy_class
    resume_path = cfg.training.resume_path
    if not resume_path:
        resume_path = None

    set_seed(seed)

    policy = make_policy(policy_class, cfg)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # Initialize tracking and best state
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # Resume support
    start_epoch = 0
    # If a directory is provided, try common filenames
    if resume_path is not None and os.path.isdir(resume_path):
        dir_path = resume_path
        candidate_files = [
            os.path.join(dir_path, "training_state.ckpt"),
            os.path.join(dir_path, "training_state.pt"),
            os.path.join(dir_path, "policy_last.ckpt"),
            os.path.join(dir_path, "policy_best.ckpt"),
        ]
        resume_path = next((p for p in candidate_files if os.path.isfile(p)), None)

    if resume_path is not None and os.path.isfile(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location="cpu")
            # Determine format
            if isinstance(ckpt, dict) and any(
                k in ckpt for k in ["model_state_dict", "state_dict", "model"]
            ):
                state_dict = ckpt.get(
                    "model_state_dict", ckpt.get("state_dict", ckpt.get("model", None))
                )
                if state_dict is not None:
                    missing, unexpected = policy.load_state_dict(
                        state_dict, strict=False
                    )
                else:
                    missing = unexpected = []
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        pass
                # Epoch info
                if "epoch" in ckpt:
                    start_epoch = int(ckpt["epoch"]) + 1
                # Restore min_val_loss and initialize best_ckpt_info from resume if available
                if "min_val_loss" in ckpt:
                    try:
                        min_val_loss = float(ckpt["min_val_loss"])
                    except Exception:
                        pass
                # Initialize best_ckpt_info to current policy state so end-of-training doesn't fail
                best_ckpt_info = (
                    start_epoch - 1,
                    min_val_loss if "min_val_loss" in locals() else np.inf,
                    deepcopy(policy.state_dict()),
                )
            else:
                # Assume it's a raw state_dict
                loading_status = policy.load_state_dict(ckpt, strict=False)
                # Try to parse epoch from filename like policy_epoch_{E}_seed_*.ckpt
                try:
                    fname = os.path.basename(resume_path)
                    m = re.search(r"policy_epoch_(\d+)_seed_", fname)
                    if m:
                        start_epoch = int(m.group(1)) + 1
                except Exception:
                    pass
            print(f"Resumed model from {resume_path}; starting at epoch {start_epoch}")
        except Exception as e:
            print(f"Warning: failed to resume from {resume_path}: {e}")

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                metadata, obstacle_data, qpos, action, obstacle_mask = data
                metadata, obstacle_data, qpos, action, obstacle_mask = (
                    metadata.cuda(),
                    obstacle_data.cuda(),
                    qpos.cuda(),
                    action.cuda(),
                    obstacle_mask.cuda(),
                )

                loss_dict = policy(metadata, obstacle_data, qpos, action, obstacle_mask)
                epoch_dicts.append(loss_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        current_epoch_train = []
        for batch_idx, data in enumerate(train_dataloader):
            metadata, obstacle_data, qpos, action, obstacle_mask = data
            metadata, obstacle_data, qpos, action, obstacle_mask = (
                metadata.cuda(),
                obstacle_data.cuda(),
                qpos.cuda(),
                action.cuda(),
                obstacle_mask.cuda(),
            )
            if batch_idx == 0:
                print(
                    f"[Train] Batch shapes - metadata: {metadata.shape}, obstacles: {obstacle_data}, Qpos: {qpos.shape}, Action: {action.shape}"
                )

            optimizer.zero_grad()
            loss_dict = policy(metadata, obstacle_data, qpos, action, obstacle_mask)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            det = detach_dict(loss_dict)
            train_history.append(det)
            current_epoch_train.append(det)
        epoch_summary = compute_dict_mean(current_epoch_train)
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)
        # if train_perm is not None and unique_count == expected:
        #     # Keep minimal log of visitation (first and last 5 indices)
        #     head = train_perm[:5]
        #     tail = train_perm[-5:]
        #     print(f"Visited sample indices (head): {head} ... (tail): {tail}")

        # Save checkpoint: every epoch when training with StepDataset; otherwise every 500 epochs
        if epoch % 500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
            # Save a full training state for resume on the same cadence
            full_state = {
                "epoch": epoch,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": seed,
                "min_val_loss": float(min_val_loss),
                "policy_class": cfg.model.policy_class,
                "policy_config": cfg,
            }
            torch.save(full_state, os.path.join(ckpt_dir, "training_state.ckpt"))

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


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
    cfg = OmegaConf.structured(RootConfig)
    main(cfg)
