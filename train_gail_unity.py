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
from my_utils import load_expert_step_loader
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
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='run training or evaluation only')
    parser.add_argument('--resume', action='store_true', help='resume discriminator/optimizers from latest GAIL ckpt (train mode only)')
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
    parser.add_argument('--action_mode', type=str, choices=['sample','mean','ahat','mean_low_noise'], default='mean_low_noise',
                        help="How to generate env actions: 'sample' from dist, 'mean' of dist, 'mean_low_noise' (low-temp sampling), or 'ahat' from ACT decoder")
    parser.add_argument('--mean_temperature', type=float, default=0.18,
                        help='Temperature for mean_low_noise mode (effective std scaling)')
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--activation', type=str, default='relu')
    # PPO/GAIL hyperparams
    parser.add_argument('--lr_policy', type=float, default=5e-5)
    parser.add_argument('--lr_disc', type=float, default=5e-5)
    parser.add_argument('--clip_eps', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--ent_coef', type=float, default=0.001)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--ppo_epochs', type=int, default=4, help='number of PPO epochs per update batch')
    # BC warmup for stochastic heads
    parser.add_argument('--bc_warmup_iters', type=int, default=5,
                        help='Number of initial update steps to run BC warmup on PPO heads')
    parser.add_argument('--bc_warmup_weight', type=float, default=1.0,
                        help='Weight for BC warmup NLL loss')
    # Auxiliary alignment of RL head mean(mu) to ACT decoder a_hat (pre-tanh space)
    parser.add_argument('--align_mu_weight', type=float, default=0.3,
                        help='Weight of auxiliary loss aligning RL mu to atanh(a_hat)')
    parser.add_argument('--ckpt_interval', type=int, default=5, help='save checkpoint every N updates')
    parser.add_argument('--bc_weight', type=float, default=0.05, help='weight for BC anchor term')
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

    # Initialize trainer and, if training, data loaders and callbacks
    trainer = None
    on_batch = None
    save_plots = lambda: None
    save_checkpoint = lambda tag: None
    update_count = 0

    if args.mode == 'train':
        trainer = TrainerGAIL(policy,
                              act_args=args_override,
                              cont_dim=policy.cont_dim,
                              bin_dim=policy.bin_dim,
                              h_dim=policy.hidden_dim,
                              device=device,
                              cfg={
                                  'lr_policy': args.lr_policy,
                                  'lr_disc': args.lr_disc,
                                  'grad_clip': 1.0,
                                  'ppo_epochs': args.ppo_epochs,
                                  'align_mu_weight': args.align_mu_weight,
                              })

        # Optionally resume discriminator/optimizers from the latest GAIL ckpt
        if args.resume:
            try:
                all_files = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.startswith('gail_ckpt_') and f.endswith('.pth')]
                if len(all_files) > 0:
                    latest = max(all_files, key=os.path.getmtime)
                    raw = torch.load(latest, map_location='cpu')
                    if isinstance(raw, dict):
                        if 'disc_state' in raw:
                            trainer.disc.load_state_dict(raw['disc_state'], strict=False)
                        if 'opt_policy' in raw:
                            trainer.opt_policy.load_state_dict(raw['opt_policy'])
                        if 'opt_value' in raw:
                            trainer.opt_value.load_state_dict(raw['opt_value'])
                        if 'opt_disc' in raw:
                            trainer.opt_disc.load_state_dict(raw['opt_disc'])
                        update_count = int(raw.get('update_count', 0))
                        print(f"Resumed trainer/optimizers from {latest} (update_count={update_count})")
            except Exception as e:
                print(f"Warning: failed to resume trainer from ckpt: {e}")

        expert_batch_size = args.update_every
        expert_loader, _, _ = load_expert_step_loader(
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            camera_names=camera_names,
            batch_size=expert_batch_size,
            shuffle=True,              # good for GAIL
            prefetch_factor=2,         # keep GPU fed
        )
        expert_iter = cycle(expert_loader)

        # Metrics accumulators
        hist = { 'loss_D': [], 'acc_exp': [], 'acc_pol': [], 'loss_pi': [], 'loss_v': [], 'ent': [], 'ent_raw': [], 'loss_bc': [] }

        def save_plots():
            try:
                keys = ['loss_D','acc_exp','acc_pol','loss_pi','loss_v','ent','ent_raw','loss_bc']
                rows = 3
                cols = 3
                fig, axs = plt.subplots(rows, cols, figsize=(12, 9))
                axes = axs.flatten()
                for i, k in enumerate(keys):
                    ax = axes[i]
                    ax.plot(hist[k])
                    ax.set_title(k)
                # Hide any unused subplots
                for j in range(len(keys), len(axes)):
                    fig.delaxes(axes[j])
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
                    'update_count': update_count,
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
            # Debug shapes for policy batch
            print('[DBG][POL] qpos_pol', tuple(qpos_pol.shape))
            print('[DBG][POL] image_pol', tuple(image_pol.shape))
            with torch.no_grad():
                out_pol = policy.forward(qpos_pol, image_pol, out_mode='dist')
                h_pol = out_pol['h']  # [B, Q, H]
                value_pol = out_pol['value']  # can be used to compare with stored
            print('[DBG][POL] h_pol', tuple(h_pol.shape))

            # Fetch an expert batch
            image_exp, qpos_exp, action_exp, is_pad_exp = next(expert_iter)
            image_exp = image_exp.to(device).float()
            qpos_exp = qpos_exp.to(device).float()
            action_exp = action_exp.to(device).float()
            is_pad_exp = is_pad_exp.to(device)

            print('[DBG][EXP] image_exp', tuple(image_exp.shape))
            print('[DBG][EXP] qpos_exp', tuple(qpos_exp.shape))
            print('[DBG][EXP] action_exp', tuple(action_exp.shape))
            # Two cases: expert has time dim (T) or is single-step per episode
            if qpos_exp.ndim == 3 and image_exp.ndim == 6:
                B_exp, T_exp, _ = action_exp.shape
                # Sample a single random timestep from each expert trajectory
                t_idx = torch.randint(0, T_exp, (B_exp,), device=device)
                batch_idx = torch.arange(B_exp, device=device)
                action_exp_t = action_exp[batch_idx, t_idx]          # [B, A]
                qpos_exp_t = qpos_exp[batch_idx, t_idx]              # [B, Dq]
                image_exp_t = image_exp[batch_idx, t_idx]            # [B, Cams, C, H, W]
                print('[DBG][EXP] (seq) t_idx', tuple(t_idx.shape))
            else:
                # Single-step expert batch already
                B_exp = action_exp.shape[0]
                action_exp_t = action_exp.squeeze(1) if action_exp.ndim == 3 and action_exp.shape[1] == 1 else action_exp  # [B, A]
                qpos_exp_t = qpos_exp                                   # [B, Dq]
                image_exp_t = image_exp                                 # [B, Cams, C, H, W]
                print('[DBG][EXP] (single-step)')

            print('[DBG][EXP] action_exp_t', tuple(action_exp_t.shape))
            print('[DBG][EXP] qpos_exp_t', tuple(qpos_exp_t.shape))
            print('[DBG][EXP] image_exp_t', tuple(image_exp_t.shape))

            # Split actions and encode expert hidden
            a_cont_exp, a_bin_exp = split_actions(action_exp_t.unsqueeze(1)) # -> [B,1,A]
            # Diagnostics: check expert continuous action scale vs policy expectation (TanhDiagNormal in [-1,1])
            try:
                cont_chk = a_cont_exp.squeeze(1)  # [B, C]
                print(f"[CHK][EXP] a_cont_exp range per-batch: min={cont_chk.min().item():.3f} max={cont_chk.max().item():.3f}")
                # print small head for a quick glance
                print(f"[CHK][EXP] a_cont_exp head (BxC up to 4x4):\n{cont_chk[:4, :4]}")
            except Exception as e:
                print(f"[CHK][EXP] a_cont_exp diagnostics failed: {e}")
            with torch.no_grad():
                h_exp_full = policy.encode_to_hidden(qpos_exp_t, image_exp_t)  # [B, Tq, H]
                h_exp = h_exp_full[:, 0:1, :]  # [B,1,H]
            print('[DBG][EXP] h_exp', tuple(h_exp.shape))
            mask_exp = torch.ones(B_exp, 1, device=device)

            # Diagnostics: sample policy-continuous actions to verify they live in [-1,1]
            try:
                with torch.no_grad():
                    out_dbg = policy.forward(qpos_exp_t, image_exp_t, out_mode='dist')
                    dist_dbg = out_dbg['cont_dist']
                    a_samp = dist_dbg.rsample()
                    print(f"[CHK][POL] sampled cont action range: min={a_samp.min().item():.3f} max={a_samp.max().item():.3f}")
                print("[CHK] TanhDiagNormal expects actions in [-1,1]; if |a_cont_exp|>1 often, there is a scale mismatch.")
            except Exception as e:
                print(f"[CHK][POL] sampling diagnostics failed: {e}")

            # Ensure policy data has a temporal dimension of 1 (already present from buffer)
            B_pol = h_pol.shape[0]
            # Use current timestep (query 0) for hidden; actions/mask already [B,1,...]
            h_pol = h_pol[:, 0:1, :]  # [B,1,H]
            print('[DBG][POL] h_pol (q0)', tuple(h_pol.shape))

            # To match batch sizes, sample from the larger policy batch
            if B_pol > B_exp:
                sample_indices = torch.randperm(B_pol, device=h_pol.device)[:B_exp]
                h_pol = h_pol[sample_indices]
                a_cont_pol = a_cont_pol[sample_indices]
                if a_bin_pol is not None:
                    a_bin_pol = a_bin_pol[sample_indices]
                mask_pol = mask_pol[sample_indices]
            # Final tensors to D
            print('[DBG][D IN] h_exp', tuple(h_exp.shape), 'a_cont_exp', tuple(a_cont_exp.shape), 'a_bin_exp', None if a_bin_exp is None else tuple(a_bin_exp.shape), 'mask_exp', tuple(mask_exp.shape))
            print('[DBG][D IN] h_pol', tuple(h_pol.shape), 'a_cont_pol', tuple(a_cont_pol.shape), 'a_bin_pol', None if a_bin_pol is None else tuple(a_bin_pol.shape), 'mask_pol', tuple(mask_pol.shape))

            # Update discriminator
            loss_D, acc_exp, acc_pol = trainer.update_discriminator(
                h_exp, a_cont_exp, a_bin_exp, mask_exp,
                h_pol, a_cont_pol, a_bin_pol, mask_pol
            )

            # Optional BC warmup to initialize stochastic heads
            nonlocal update_count
            if args.bc_warmup_iters > 0 and update_count < args.bc_warmup_iters:
                warmup_nll = trainer.bc_warmup_step(
                    qpos_exp, image_exp, a_cont_exp, a_bin_exp, mask_exp,
                    iters=1, weight=args.bc_warmup_weight
                )
                print(f"[Warmup] BC NLL: {warmup_nll:.4f}")

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
            hist['ent_raw'].append(ppo_logs.get('ent_raw', 0.0))
            hist['loss_bc'].append(ppo_logs.get('loss_bc', 0.0))
            print({'loss_D': loss_D, 'acc_exp': acc_exp, 'acc_pol': acc_pol, **ppo_logs})

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
        on_batch=on_batch,  # None in eval mode disables training updates
        query_frequency=args.query_frequency,
        action_mode=args.action_mode,
        mean_temperature=args.mean_temperature,
    )

    try:
        interface.start_server()
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        if args.mode == 'train':
            save_plots()
            save_checkpoint('final')


if __name__ == '__main__':
    main()
