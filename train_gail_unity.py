import argparse
import os
import pickle
import torch
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from policy import ACTPolicy
from trainers.trainer_gail import TrainerGAIL
from unity_vr_gail_interface import UnityVRGAILInterface
from my_utils import load_data
from action_utils import ACTION_INDICES
from gail.discriminator import reward_from_prob


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def split_actions(action_tensor):
    # action_tensor: [B, T, A]
    cont_idx = ACTION_INDICES['continuous']
    bin_idx = ACTION_INDICES['boolean']
    a_cont = action_tensor[..., cont_idx]
    a_bin = action_tensor[..., bin_idx] if len(bin_idx) > 0 else None
    return a_cont, a_bin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5557)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--update_every', type=int, default=1024)
    # Imitation/ACT-style args
    parser.add_argument('--policy_class', type=str, default='ACT')
    parser.add_argument('--task_name', type=str, default='gail_rl')
    parser.add_argument('--batch_size', type=int, default=128) # This batch size (dataloader batch size) is for discriminator update
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--kl_weight', type=float, default=0.0)
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--query_frequency', type=int, default=1)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--activation', type=str, default='relu')
    # PPO/GAIL hyperparams
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--ckpt_interval', type=int, default=5, help='save checkpoint every N updates')
    parser.add_argument('--bc_weight', type=float, default=0.0, help='weight for BC anchor term')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    task_name = args.task_name
    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        raise ValueError(f'Unknown task name: {task_name}')

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset statistics used by Unity interface and policy I/O
    stats_path = os.path.join(args.ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Build policy
    args_override = {
        'kl_weight': args.kl_weight,
        'num_queries': args.chunk_size,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr': args.lr,
        'lr_backbone': args.lr_backbone,
        'backbone': args.backbone,
        'enc_layers': args.enc_layers,
        'dec_layers': args.dec_layers,
        'nheads': args.nheads,
        'activation': args.activation,
        'camera_names': camera_names,
        'ckpt_dir': args.ckpt_dir,
        'policy_class': args.policy_class,
        'task_name': args.task_name,
        'seed': args.seed,
        'num_epochs': args.num_epochs,
    }
    policy = ACTPolicy(args_override).to(device)

    # Load pretrained ACT weights if present (robust to head/key differences)
    def _try_load(sd_dict) -> bool:
        """Attempt to load a state dict into policy with several fallbacks.
        Returns True if any attempt succeeds."""
        def _report(msg_prefix, res):
            try:
                # PyTorch returns IncompatibleKeys with attributes
                mk = getattr(res, 'missing_keys', [])
                uk = getattr(res, 'unexpected_keys', [])
            except Exception:
                mk, uk = [], []
            print(f"{msg_prefix} (missing={len(mk)}, unexpected={len(uk)})")

        # 1) Direct load to full policy (allow missing heads)
        try:
            res = policy.load_state_dict(sd_dict, strict=False)
            _report("Loaded into policy (strict=False)", res)
            return True
        except Exception:
            pass

        # 2) Try removing 'module.' prefix
        try:
            sd2 = { (k[7:] if k.startswith('module.') else k): v for k, v in sd_dict.items() }
            res = policy.load_state_dict(sd2, strict=False)
            _report("Loaded into policy after stripping 'module.' (strict=False)", res)
            return True
        except Exception:
            pass

        # 3) Try loading into base model (checkpoint may only have ACT model weights)
        try:
            # If keys look like 'model.xxx', strip that when loading into policy.model
            if any(k.startswith('model.') for k in sd_dict.keys()):
                base_sd = { k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd_dict.items() }
            else:
                base_sd = sd_dict
            res = policy.model.load_state_dict(base_sd, strict=False)
            _report("Loaded into policy.model (strict=False)", res)
            print("Note: Action/value heads initialized randomly (first-time head init).")
            return True
        except Exception:
            pass

        return False

    ckpt_candidates = [
        os.path.join(args.ckpt_dir, 'policy_best.ckpt'),
        os.path.join(args.ckpt_dir, 'policy.ckpt'),
        os.path.join(args.ckpt_dir, 'gail_ckpt_final.pth'),
    ]
    for ckpt_path in ckpt_candidates:
        if os.path.isfile(ckpt_path):
            try:
                raw = torch.load(ckpt_path, map_location='cpu')
                # Normalize to a plain state-dict first
                if isinstance(raw, dict) and 'policy_state' in raw:
                    state = raw['policy_state']
                elif isinstance(raw, dict) and 'state_dict' in raw:
                    state = raw['state_dict']
                else:
                    state = raw if isinstance(raw, dict) else {}

                ok = _try_load(state)
                if ok:
                    policy.eval()
                    print(f"Loaded pretrained weights from {ckpt_path}")
                    break
                else:
                    print(f"Warning: no compatible keys found in {ckpt_path}; skipping.")
            except Exception as e:
                print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")

    # Initialize trainer
    trainer = TrainerGAIL(policy,
                          act_args=args_override,
                          cont_dim=policy.cont_dim,
                          bin_dim=policy.bin_dim,
                          h_dim=policy.hidden_dim,
                          device=device,
                          cfg={
                              'lr_policy': 3e-4,
                              'lr_disc': 1e-4,
                              'grad_clip': 1.0,
                          })

    batch_size_train = args.batch_size
    batch_size_val = args.batch_size

    # Expert data loader
    train_loader, _, _, _ = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
    )
    expert_iter = cycle(train_loader)

    # Metrics accumulators
    hist = { 'loss_D': [], 'acc_exp': [], 'acc_pol': [], 'loss_pi': [], 'loss_v': [], 'ent': [], 'loss_bc': [] }
    update_count = 0

    def save_plots():
        try:
            fig, axs = plt.subplots(2, 3, figsize=(10, 6))
            keys = ['loss_D','acc_exp','acc_pol','loss_pi','loss_v','ent','loss_bc']
            for ax, k in zip(axs.flatten(), keys):
                ax.plot(hist[k])
                ax.set_title(k)
            plt.tight_layout()
            plot_path = os.path.join(args.ckpt_dir, 'gail_training_plots.png')
            plt.savefig(plot_path)
            plt.close(fig)
        except Exception as e:
            print(f'Plot save failed: {e}')

    def save_checkpoint(tag: str):
        try:
            ckpt = {
                'policy_state': policy.state_dict(),
                'disc_state': trainer.disc.state_dict(),
                'opt_policy': trainer.opt_policy.state_dict(),
                'opt_value': trainer.opt_value.state_dict(),
                'opt_disc': trainer.opt_disc.state_dict(),
                'args': vars(args),
            }
            path = os.path.join(args.ckpt_dir, f'gail_ckpt_{tag}.pth')
            torch.save(ckpt, path)
            print(f'Saved checkpoint: {path}')
        except Exception as e:
            print(f'Checkpoint save failed: {e}')

    def on_batch(batch):
        # batch comes from UnityVRGAILInterface with tensors on CPU; move to device
        for k, v in list(batch.items()):
            if isinstance(v, dict):
                batch[k] = {kk: vv.to(device) for kk, vv in v.items()}
            elif torch.is_tensor(v):
                batch[k] = v.to(device)

        qpos_pol = batch['obs']['qpos']
        image_pol = batch['obs']['image']
        a_cont_pol = batch['a_cont']
        a_bin_pol = batch.get('a_bin', None)
        mask_pol = batch['mask']

        # Policy forward to get h and value consistency (no grad during rollout)
        with torch.no_grad():
            out_pol = policy.forward(qpos_pol, image_pol, out_mode='dist')
            h_pol = out_pol['h']  # [B, Q, H]
            value_pol = out_pol['value']  # can be used to compare with stored

        # Fetch an expert batch
        image_exp, qpos_exp, action_exp, is_pad_exp = next(expert_iter)
        image_exp = image_exp.to(device).float()
        qpos_exp = qpos_exp.to(device).float()
        action_exp = action_exp.to(device).float()
        is_pad_exp = is_pad_exp.to(device)

        B_exp, T_exp, _ = action_exp.shape
        # To match policy data, sample a single random timestep from each expert trajectory
        t_idx = torch.randint(0, T_exp, (B_exp,), device=device)
        batch_idx = torch.arange(B_exp, device=device)
        # Sample expert actions at random timestep per episode
        action_exp_t = action_exp[batch_idx, t_idx]

        a_cont_exp, a_bin_exp = split_actions(action_exp_t.unsqueeze(1)) # -> [B,1,A]
        # Expert observations are single-step; encode directly (no grad for D input features)
        with torch.no_grad():
            h_exp_full = policy.encode_to_hidden(qpos_exp, image_exp)  # [B, Q, H]
        # Use current timestep (query 0) for discriminator
        h_exp = h_exp_full[:, 0:1, :]  # [B,1,H]
        mask_exp = torch.ones(B_exp, 1, device=device)

        # Reshape policy data to have a temporal dimension of 1
        B_pol = h_pol.shape[0]
        # Use current timestep (query 0) then add temporal dim
        h_pol = h_pol[:, 0:1, :]  # [B,1,H]
        a_cont_pol = a_cont_pol.unsqueeze(1) # [B,1,A]
        if a_bin_pol is not None:
            a_bin_pol = a_bin_pol.unsqueeze(1) # [B,1,A]
        mask_pol = mask_pol.unsqueeze(1) # [B,1]

        # To match batch sizes, sample from the larger policy batch
        if B_pol > B_exp:
            sample_indices = torch.randperm(B_pol, device=h_pol.device)[:B_exp]
            h_pol = h_pol[sample_indices]
            a_cont_pol = a_cont_pol[sample_indices]
            if a_bin_pol is not None:
                a_bin_pol = a_bin_pol[sample_indices]
            mask_pol = mask_pol[sample_indices]

        # Update discriminator
        loss_D, acc_exp, acc_pol = trainer.update_discriminator(
            h_exp, a_cont_exp, a_bin_exp, mask_exp,
            h_pol, a_cont_pol, a_bin_pol, mask_pol
        )

        # Compute GAIL rewards for policy batch
        with torch.no_grad():
            logits_pol, prob_pol = trainer.disc(h_pol, a_cont_pol, a_bin_pol, mask_pol)
            r = reward_from_prob(prob_pol)

        # Add rewards to batch and run PPO update (with optional BC anchor)
        batch['reward'] = r
        ppo_logs = trainer.ppo_update(
            batch,
            clip_eps=args.clip_eps,
            gamma=args.gamma,
            lam=args.lam,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            bc_data=(qpos_exp, image_exp, a_cont_exp, a_bin_exp, (~is_pad_exp).float()) if args.bc_weight > 0 else None,
            bc_weight=args.bc_weight,
        )

        # Log & checkpoint
        hist['loss_D'].append(loss_D)
        hist['acc_exp'].append(acc_exp)
        hist['acc_pol'].append(acc_pol)
        hist['loss_pi'].append(ppo_logs['loss_pi'])
        hist['loss_v'].append(ppo_logs['loss_v'])
        hist['ent'].append(ppo_logs['ent'])
        hist['loss_bc'].append(ppo_logs.get('loss_bc', 0.0))
        print({'loss_D': loss_D, 'acc_exp': acc_exp, 'acc_pol': acc_pol, **ppo_logs})

        nonlocal update_count
        update_count += 1
        if update_count % args.ckpt_interval == 0:
            save_plots()
            save_checkpoint(f'update_{update_count}')

    # Start Unity GAIL Interface
    interface = UnityVRGAILInterface(
        policy=policy,
        stats=stats,
        host=args.host,
        port=args.port,
        update_every=args.update_every,
        on_batch=on_batch,
        query_frequency=args.query_frequency,
    )

    try:
        interface.start_server()
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        save_plots()
        save_checkpoint('final')


if __name__ == '__main__':
    main()
