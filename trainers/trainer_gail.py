from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict

from gail.discriminator import GAILDiscriminator, gail_loss_D, reward_from_prob, masked_mean
from rl.ppo import compute_gae, ppo_clip_loss, value_loss
from distributions import BernoulliMulti


class TrainerGAIL:
    def __init__(self, policy, act_args, cont_dim, bin_dim, h_dim, device='cuda', cfg=None):
        self.policy = policy.to(device)
        self.device = device
        self.cfg = cfg or {}
        # Safer defaults for squashed Gaussian stability
        self.cfg.setdefault('min_log_std', -1.5)
        self.cfg.setdefault('max_log_std', 0.5)
        self.disc = GAILDiscriminator(h_dim=h_dim, cont_dim=cont_dim, bin_dim=bin_dim).to(device)
        # policy optimizer: policy trunk + action heads (NOT value)
        # build param groups so value_head is excluded from opt_policy
        value_params = list(self.policy.value_head.parameters())
        value_param_ids = {id(p) for p in value_params}
        policy_params = [p for p in self.policy.parameters() if id(p) not in value_param_ids]
        self.opt_policy = optim.Adam(policy_params, lr=self.cfg.get('lr_policy', 3e-4))
        # allow separate LR for value head, fallback to lr_policy for backward compat
        self.opt_value = optim.Adam(value_params, lr=self.cfg.get('lr_value', self.cfg.get('lr_policy', 3e-4)))
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.cfg.get('lr_disc', 1e-4))

    def compute_logp_and_entropy(self, out_dist, a_cont, a_bin, mask):
        """Compute log-prob and entropy for ALL timesteps.
        - Supports optional query dimension by averaging across queries.
        - Supports squashed continuous policy via tanh correction when cfg['squashed_cont'] is True.
        Returns: [B, T] tensors (logp, ent), masked by `mask`.
        """
        from distributions import TanhDiagNormal, BernoulliMulti

        cont_dist = out_dist['cont_dist']
        bin_dist = out_dist['bin_dist']

        # Prepare batch/time dims
        assert mask.dim() == 2, f"mask must be [B,T], got {tuple(mask.shape)}"
        B_mask, T_mask = mask.shape

        def reshape_BT(x):
            if x is None:
                return None
            n = x.shape[0]
            if n == B_mask * T_mask:
                return x.view(B_mask, T_mask, *x.shape[1:])
            if n == B_mask:
                x_bt = x.view(B_mask, 1, *x.shape[1:])
                return x_bt.expand(-1, T_mask, *([-1] * (x.dim() - 1)))
            raise ValueError(f"Unexpected batch dimension {n} for tensor with shape {tuple(x.shape)}, expected {B_mask} or {B_mask*T_mask}")

        # Continuous params to [B,T,(Tq),C]
        mu_bt = reshape_BT(cont_dist.mu)
        logstd_bt = reshape_BT(cont_dist.log_std)

        # Actions to [B,T,(Tq),C]
        if a_cont.dim() == 4:  # [B, T?, Tq, C]
            # Only expand time dimension if it's 1
            if a_cont.shape[0] == B_mask and a_cont.shape[1] == 1 and T_mask > 1:
                a_cont_bt = a_cont.expand(B_mask, T_mask, a_cont.shape[2], a_cont.shape[3])
            elif a_cont.shape[0] == B_mask and a_cont.shape[1] == T_mask:
                a_cont_bt = a_cont
            else:
                raise ValueError(f"a_cont 4D incompatible with (B,T)=({B_mask},{T_mask}), got {tuple(a_cont.shape)}")
        elif a_cont.dim() == 3:  # [B, T?, C]
            if a_cont.shape[0] == B_mask and a_cont.shape[1] == 1 and T_mask > 1:
                a_cont_bt = a_cont.expand(B_mask, T_mask, a_cont.shape[2])
            elif a_cont.shape[0] == B_mask and a_cont.shape[1] == T_mask:
                a_cont_bt = a_cont
            else:
                raise ValueError(f"a_cont 3D incompatible with (B,T)=({B_mask},{T_mask}), got {tuple(a_cont.shape)}")
        else:
            raise ValueError(f"Unexpected a_cont shape {tuple(a_cont.shape)}")

        # Build distribution and compute per-step logp/entropy using squashed Gaussian consistently
        DistClass = TanhDiagNormal
        min_ls = float(self.cfg.get('min_log_std', -5.0))
        max_ls = float(self.cfg.get('max_log_std', 2.0))

        # If query dim exists, align with actions; reduce over queries by mean
        if mu_bt.dim() == 4:  # [B,T,Tq,C]
            # If actions lack query dim or have singleton query dim, use first query of distribution
            if a_cont_bt.dim() == 3 or (a_cont_bt.dim() == 4 and a_cont_bt.shape[2] == 1):
                mu_use = mu_bt[:, :, 0, :]
                logstd_use = logstd_bt[:, :, 0, :]
                dist = DistClass(mu_use, logstd_use, min_log_std=min_ls, max_log_std=max_ls)  # batch [B,T]
                a_use = a_cont_bt if a_cont_bt.dim() == 3 else a_cont_bt[:, :, 0, :]
                logp_cont = dist.log_prob(a_use)  # [B,T]
                ent_cont = dist.entropy()         # [B,T]
            else:
                # Both have query dim -> compute per-query then average over queries
                dist = DistClass(mu_bt, logstd_bt, min_log_std=min_ls, max_log_std=max_ls)          # batch [B,T,Tq]
                logp_q = dist.log_prob(a_cont_bt)           # [B,T,Tq]
                ent_q = dist.entropy()                      # [B,T,Tq]
                logp_cont = logp_q.mean(dim=2)              # [B,T]
                ent_cont = ent_q.mean(dim=2)                # [B,T]
        else:  # [B,T,C]
            dist = DistClass(mu_bt, logstd_bt, min_log_std=min_ls, max_log_std=max_ls)              # batch [B,T]
            # If actions provide query dim, use first query of actions
            if a_cont_bt.dim() == 4:
                a_use = a_cont_bt[:, :, 0, :]
            else:
                a_use = a_cont_bt
            logp_cont = dist.log_prob(a_use)                # [B,T]
            ent_cont = dist.entropy()                       # [B,T]

        logp_BT = logp_cont
        ent_BT = ent_cont

        # Optional binary part over all timesteps; handle optional query dim similarly
        if bin_dist is not None and a_bin is not None:
            if hasattr(bin_dist, 'logits') and bin_dist.logits is not None:
                logits = reshape_BT(bin_dist.logits)  # [B,T,(Tq),Cb]
                make_bin = lambda t: BernoulliMulti(logits=t)
            else:
                probs = reshape_BT(bin_dist.probs)
                make_bin = lambda t: BernoulliMulti(probs=t)

            # Align a_bin
            if a_bin.dim() == 4:  # [B, T?, Tq, Cb]
                if a_bin.shape[0] == B_mask and a_bin.shape[1] == 1 and T_mask > 1:
                    a_bin_bt = a_bin.expand(B_mask, T_mask, a_bin.shape[2], a_bin.shape[3])
                elif a_bin.shape[0] == B_mask and a_bin.shape[1] == T_mask:
                    a_bin_bt = a_bin
                else:
                    raise ValueError(f"a_bin 4D incompatible with (B,T)=({B_mask},{T_mask}), got {tuple(a_bin.shape)}")
            elif a_bin.dim() == 3:  # [B, T?, Cb]
                if a_bin.shape[0] == B_mask and a_bin.shape[1] == 1 and T_mask > 1:
                    a_bin_bt = a_bin.expand(B_mask, T_mask, a_bin.shape[2])
                elif a_bin.shape[0] == B_mask and a_bin.shape[1] == T_mask:
                    a_bin_bt = a_bin
                else:
                    raise ValueError(f"a_bin 3D incompatible with (B,T)=({B_mask},{T_mask}), got {tuple(a_bin.shape)}")
            else:
                raise ValueError(f"Unexpected a_bin shape {tuple(a_bin.shape)}")

            # Determine parameter tensor (logits or probs) to use
            param_bt = logits if hasattr(bin_dist, 'logits') and bin_dist.logits is not None else probs

            if param_bt.dim() == 4:  # [B,T,Tq,Cb]
                if a_bin_bt.dim() == 3 or (a_bin_bt.dim() == 4 and a_bin_bt.shape[2] == 1):
                    # Actions have no query dim or singleton query dim -> use first query of distribution
                    a_bin_use = a_bin_bt if a_bin_bt.dim() == 3 else a_bin_bt[:, :, 0, :]
                    bin_dist_use = make_bin(param_bt[:, :, 0, :])
                    logp_bin = bin_dist_use.log_prob(a_bin_use)  # [B,T]
                    ent_bin = bin_dist_use.entropy()             # [B,T]
                else:
                    bin_dist_q = make_bin(param_bt)             # [B,T,Tq]
                    logp_q = bin_dist_q.log_prob(a_bin_bt)      # [B,T,Tq]
                    ent_q = bin_dist_q.entropy()                # [B,T,Tq]
                    logp_bin = logp_q.mean(dim=2)
                    ent_bin = ent_q.mean(dim=2)
            else:  # [B,T,Cb]
                bin_dist_use = make_bin(param_bt)
                if a_bin_bt.dim() == 4:
                    a_bin_use = a_bin_bt[:, :, 0, :]
                else:
                    a_bin_use = a_bin_bt
                logp_bin = bin_dist_use.log_prob(a_bin_use)
                ent_bin = bin_dist_use.entropy()

            logp_BT = logp_BT + logp_bin
            ent_BT = ent_BT + ent_bin

        # Mask and return [B,T]
        logp_BT = logp_BT * mask
        ent_BT = ent_BT * mask
        return logp_BT, ent_BT

    def update_discriminator(self, h_exp, a_cont_exp, a_bin_exp, mask_exp, h_pol, a_cont_pol, a_bin_pol, mask_pol):
        self.disc.train()
        self.opt_disc.zero_grad(set_to_none=True)
        print(f'{h_exp.shape=}')
        print(f'{a_cont_exp.shape=}')
        if a_bin_exp is not None:
            print(f'{a_bin_exp.shape=}')
        print(f'{h_pol.shape=}')
        print(f'{a_cont_pol.shape=}')
        if a_bin_pol is not None:
            print(f'{a_bin_pol.shape=}')

        _, prob_exp = self.disc(h_exp.detach(), a_cont_exp.detach(), a_bin_exp.detach() if a_bin_exp is not None else None, mask_exp)
        _, prob_pol = self.disc(h_pol.detach(), a_cont_pol.detach(), a_bin_pol.detach() if a_bin_pol is not None else None, mask_pol)
        
        print(f'prob_exp mean={prob_exp.mean().item():.4f} min={prob_exp.min().item():.4f} max={prob_exp.max().item():.4f}')
        print(f'prob_pol mean={prob_pol.mean().item():.4f} min={prob_pol.min().item():.4f} max={prob_pol.max().item():.4f}')
        print(f'mask_exp sum={mask_exp.sum().item()} mask_pol sum={mask_pol.sum().item()}')
        # BCE component diagnostics
        eps = 1e-6
        pos_term = -torch.log(torch.clamp(prob_exp, eps, 1.0 - eps))
        neg_term = -torch.log(torch.clamp(1.0 - prob_pol, eps, 1.0 - eps))
        print(f'pos_loss(mean)={masked_mean(pos_term, mask_exp).item():.6f}  neg_loss(mean)={masked_mean(neg_term, mask_pol).item():.6f}')
        smooth = getattr(self.disc, 'label_smooth', 0.0)
        print(f'y_pos(smooth)={smooth:.3f}')
        loss_D = gail_loss_D(prob_exp, prob_pol, mask_exp, mask_pol, smooth=smooth)
        loss_D.backward()
        nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=5.0)
        self.opt_disc.step()
        with torch.no_grad():
            acc_exp = masked_mean((prob_exp > 0.5).float(), mask_exp)
            acc_pol = masked_mean((prob_pol <= 0.5).float(), mask_pol)
        self.disc.eval()
        return loss_D.item(), acc_exp.item(), acc_pol.item()

    def bc_warmup_step(self, qpos_exp, image_exp, a_cont_exp, a_bin_exp, mask_exp, iters: int = 1, weight: float = 1.0):
        """Pretrain stochastic heads by maximizing log-prob of expert actions.
        Only updates policy (excluding value head). Returns avg NLL over valid steps.
        """
        if weight <= 0.0 or iters <= 0:
            return 0.0
        self.policy.train()
        value_params = list(self.policy.value_head.parameters())
        policy_params = [p for p in self.policy.parameters() if all(p is not vp for vp in value_params)]
        total_loss = 0.0
        for _ in range(iters):
            out = self.policy.forward(qpos_exp, image_exp, out_mode='dist')
            logp_exp, _ = self.compute_logp_and_entropy(out, a_cont_exp, a_bin_exp, mask_exp)
            # Negative log-likelihood averaged over valid steps
            denom = mask_exp.sum().clamp_min(1.0)
            nll = -(logp_exp.sum() / denom)
            loss = weight * nll
            self.opt_policy.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy_params, max_norm=self.cfg.get('grad_clip', 1.0))
            self.opt_policy.step()
            total_loss += nll.item()
        self.policy.eval()
        return total_loss / max(1, iters)

    def ppo_update(self, batch, clip_eps=0.2, gamma=0.99, lam=0.95, ent_coef=1e-3, vf_coef=0.5,
                   bc_data=None, bc_weight: float = 0.0):
        self.policy.train()

        # Inputs should not carry graphs from rollout storage
        qpos = batch['obs']['qpos'].to(self.device).detach()
        image = batch['obs']['image'].to(self.device).detach()
        a_cont = batch['a_cont'].to(self.device).detach()
        a_bin = batch.get('a_bin')
        a_bin = a_bin.to(self.device).detach() if a_bin is not None else None
        mask = batch['mask'].to(self.device).detach()
        logp_old = batch['logp'].to(self.device).detach()
        v_old = batch['value'].to(self.device).detach()
        r = batch['reward'].to(self.device).detach()
        dones = batch['dones'].to(self.device).detach()

        # --- Reconcile shapes to [B, T] ---
        # Infer B from rewards or actions
        B = r.shape[0] if r.dim() >= 1 else mask.shape[0]
        # Infer T from any 1D tensors (ppo update step size = B*T)
        T = 1
        for x in (v_old, dones, mask):
            if x.dim() == 1 and x.numel() % B == 0:
                T = max(T, x.numel() // B)
        # Reshape/expand r to [B, T]
        if r.dim() == 2 and r.shape[1] == 1 and T > 1:
            r = r.expand(B, T)
        elif r.dim() == 1 and r.numel() == B * T:
            r = r.view(B, T)
        elif r.dim() == 2 and r.shape == (B, T):
            pass
        elif r.dim() == 1 and r.numel() == B and T > 1:
            # Broadcast per-trajectory reward across timesteps as a fallback
            r = r.view(B, 1).expand(B, T)
        else:
            raise ValueError(f"reward 'r' has incompatible shape {tuple(r.shape)} with inferred (B,T)=({B},{T})")
        # Reshape v_old/dones/mask to [B, T]
        if v_old.dim() == 1 and v_old.numel() == B * T:
            v_old = v_old.view(B, T)
        if dones.dim() == 1 and dones.numel() == B * T:
            dones = dones.view(B, T)
        if mask.dim() == 1 and mask.numel() == B * T:
            mask = mask.view(B, T)
        # Reshape logp_old to [B, T]
        if logp_old.dim() == 1 and logp_old.numel() == B * T:
            logp_old = logp_old.view(B, T)
        elif logp_old.dim() == 2 and logp_old.shape == (B, T):
            pass
        else:
            raise ValueError(f"logp_old has incompatible shape {tuple(logp_old.shape)} with inferred (B,T)=({B},{T})")
        # Final shape assertions
        assert r.shape == (B, T), f"r shape {tuple(r.shape)} != ({B},{T})"
        assert v_old.shape == (B, T), f"v_old shape {tuple(v_old.shape)} != ({B},{T})"
        assert dones.shape == (B, T), f"dones shape {tuple(dones.shape)} != ({B},{T})"
        assert mask.shape == (B, T), f"mask shape {tuple(mask.shape)} != ({B},{T})"
        assert logp_old.shape == (B, T), f"logp_old shape {tuple(logp_old.shape)} != ({B},{T})"

        # Compute advantages/returns
        adv, ret = compute_gae(r, v_old, dones, mask, gamma=gamma, lam=lam)
        adv = torch.clamp(adv, -10.0, 10.0)
        # Normalize advantages using only valid (mask>0) timesteps
        valid = mask > 0
        if valid.any():
            adv_mean = adv[valid].mean()
            adv_std = adv[valid].std(unbiased=False)
            adv_norm = torch.zeros_like(adv)
            adv_norm[valid] = (adv[valid] - adv_mean) / (adv_std + 1e-8)
            adv = adv_norm
        else:
            adv = torch.zeros_like(adv)

        # Keep rollout-time logp_old as the behavior policy reference
        # Perform multiple PPO epochs per batch with fixed logp_old
        ppo_epochs = int(self.cfg.get('ppo_epochs', 4))
        cont_dim = None
        for epoch in range(max(1, ppo_epochs)):
            # Forward policy
            out = self.policy.forward(qpos, image, out_mode='dist')
            logp, ent = self.compute_logp_and_entropy(out, a_cont, a_bin, mask)
            v = out['value']
            # Log mean entropy per continuous action dim for exploration diagnostics
            try:
                cont_dim = out['cont_dist'].mu.shape[-1]
            except Exception:
                cont_dim = None
            # Log std diagnostics to catch min/max floors
            try:
                ls = out['cont_dist'].log_std
                print(f"[PPO DBG] log_std min/mean/max: {ls.min().item():.3f} {ls.mean().item():.3f} {ls.max().item():.3f}")
            except Exception:
                pass

            # --- Align value tensor to [B, T] ---
            if v.dim() == 2:
                if v.shape == (B, T):
                    pass
                elif v.shape[0] == B * T and v.shape[1] == 1:
                    # Flattened sequence of per-step values -> reshape to [B, T]
                    v = v.view(B, T)
                elif v.shape[0] == B * T and v.shape[1] != 1:
                    # Per-step values with an extra feature/query dim -> take first feature/query then reshape to [B, T]
                    v = v[:, 0].view(B, T)
                elif v.shape[0] == B and v.shape[1] == 1 and T > 1:
                    v = v.expand(B, T)
                elif v.shape[0] == B and v.shape[1] != T:
                    # Reduce any extra dimension (e.g., queries) to a scalar per batch, then broadcast over T
                    v = v.mean(dim=1, keepdim=True).expand(B, T)
                else:
                    # Try best-effort reshape if total elements match
                    if v.numel() == B * T:
                        v = v.view(B, T)
                    else:
                        raise ValueError(f"value 'v' has incompatible shape {tuple(v.shape)} with inferred (B,T)=({B},{T})")
            elif v.dim() == 1:
                if v.numel() == B * T:
                    v = v.view(B, T)
                elif v.numel() == B and T > 1:
                    v = v.view(B, 1).expand(B, T)
                else:
                    raise ValueError(f"value 'v' has incompatible shape {tuple(v.shape)} with inferred (B,T)=({B},{T})")
            else:
                raise ValueError(f"Unexpected value 'v' shape {tuple(v.shape)}; expected 1D or 2D")

            # Losses
            L_pi = ppo_clip_loss(logp, logp_old, adv, clip_eps, mask)
            L_v = value_loss(v, ret, mask)
            L_ent = -(ent.sum() / mask.sum().clamp_min(1.0)) if ent_coef != 0.0 else v.new_tensor(0.0)
            if cont_dim is not None and cont_dim > 0:
                with torch.no_grad():
                    mean_ent_per_dim = (ent.sum() / mask.sum().clamp_min(1.0)) / cont_dim
                    print(f"[PPO DBG] mean entropy per dim: {mean_ent_per_dim.item():.4f} (cont_dim={cont_dim})")
            # We do two separate optimization steps to avoid value gradients updating the shared body
            # Policy step: optimize all params EXCEPT value head on (pi + ent + optional BC)
            value_params = list(self.policy.value_head.parameters())
            policy_params = [
                p for p in self.policy.parameters()
                if all(p is not vp for vp in value_params)
            ]

            # Optional BC anchor: encourage current policy to imitate expert actions on demo states
            L_bc = v.new_tensor(0.0)
            if bc_data is not None and bc_weight > 0.0:
                qpos_exp, image_exp, a_cont_exp, a_bin_exp, mask_exp = bc_data
                qpos_exp = qpos_exp.to(self.device)
                image_exp = image_exp.to(self.device)
                a_cont_exp = a_cont_exp.to(self.device)
                a_bin_exp = a_bin_exp.to(self.device) if a_bin_exp is not None else None
                mask_exp = mask_exp.to(self.device)

                out_exp = self.policy.forward(qpos_exp, image_exp, out_mode='dist')
                logp_exp, _ = self.compute_logp_and_entropy(out_exp, a_cont_exp, a_bin_exp, mask_exp)
                # Negative log-likelihood averaged over valid steps
                L_bc = -(logp_exp.sum() / mask_exp.sum().clamp_min(1.0))

            loss_policy = L_pi + ent_coef * L_ent + (bc_weight * L_bc if bc_weight > 0.0 else v.new_tensor(0.0))
            self.opt_policy.zero_grad(set_to_none=True)
            loss_policy.backward()
            nn.utils.clip_grad_norm_(policy_params, max_norm=self.cfg.get('grad_clip', 1.0))
            self.opt_policy.step()

            # Value step: optimize only value_head on (vf_coef * L_v)
            # Temporarily freeze all non-value params to avoid value gradients updating them
            non_value_params = [p for p in self.policy.parameters() if all(p is not vp for vp in value_params)]
            trunk_requires = []
            for p in non_value_params:
                trunk_requires.append(p.requires_grad)
                p.requires_grad = False
            try:
                # Recompute value to ensure correct graph under current requires_grad settings
                out_val = self.policy.forward(qpos, image, out_mode='dist')
                v_val = out_val['value']
                # Align to [B,T] similarly as above
                if v_val.dim() == 2:
                    if v_val.shape == (B, T):
                        pass
                    elif v_val.shape[0] == B * T and v_val.shape[1] == 1:
                        v_val = v_val.view(B, T)
                    elif v_val.shape[0] == B * T and v_val.shape[1] != 1:
                        v_val = v_val[:, 0].view(B, T)
                    elif v_val.shape[0] == B and v_val.shape[1] == 1 and T > 1:
                        v_val = v_val.expand(B, T)
                    elif v_val.shape[0] == B and v_val.shape[1] != T:
                        v_val = v_val.mean(dim=1, keepdim=True).expand(B, T)
                    else:
                        if v_val.numel() == B * T:
                            v_val = v_val.view(B, T)
                        else:
                            raise ValueError(f"value 'v' has incompatible shape {tuple(v_val.shape)} with inferred (B,T)=({B},{T})")
                elif v_val.dim() == 1:
                    if v_val.numel() == B * T:
                        v_val = v_val.view(B, T)
                    elif v_val.numel() == B and T > 1:
                        v_val = v_val.view(B, 1).expand(B, T)
                    else:
                        raise ValueError(f"value 'v' has incompatible shape {tuple(v_val.shape)} with inferred (B,T)=({B},{T})")
                else:
                    raise ValueError(f"Unexpected value 'v' shape {tuple(v_val.shape)}; expected 1D or 2D")

                L_v_val = value_loss(v_val, ret, mask)
                loss_value = vf_coef * L_v_val
                self.opt_value.zero_grad(set_to_none=True)
                loss_value.backward()
                nn.utils.clip_grad_norm_(self.policy.value_head.parameters(), max_norm=self.cfg.get('grad_clip', 1.0))
                self.opt_value.step()
            finally:
                # Restore requires_grad flags for frozen params
                for p, req in zip(non_value_params, trunk_requires):
                    p.requires_grad = req
        self.policy.eval()

        # Debug stats after PPO update: compare new logp vs frozen logp_old
        with torch.no_grad():
            out_post = self.policy.forward(qpos, image, out_mode='dist')
            logp_post, _ = self.compute_logp_and_entropy(out_post, a_cont, a_bin, mask)
            valid = mask > 0
            num_valid = valid.sum().item()
            print(f"[PPO DBG] (POST) mask valid sum={num_valid} shape={tuple(mask.shape)}")
            if num_valid > 0:
                lp = logp_post[valid]
                lpo = logp_old[valid]
                dv = (lp - lpo)
                dv_clamped = torch.clamp(dv, -20.0, 20.0)
                ratio = torch.exp(dv_clamped)
                ad = adv[valid]
                print(f"[PPO DBG] (POST) logp   min/mean/max: {lp.min().item():.3f} {lp.mean().item():.3f} {lp.max().item():.3f}")
                print(f"[PPO DBG] (POST) logp_o min/mean/max: {lpo.min().item():.3f} {lpo.mean().item():.3f} {lpo.max().item():.3f}")
                print(f"[PPO DBG] (POST) dlogp  min/mean/max: {dv.min().item():.3f} {dv.mean().item():.3f} {dv.max().item():.3f}")
                print(f"[PPO DBG] (POST) ratio  min/mean/max: {ratio.min().item():.3e} {ratio.mean().item():.3e} {ratio.max().item():.3e}")
                print(f"[PPO DBG] (POST) adv    min/mean/max/std: {ad.min().item():.3f} {ad.mean().item():.3f} {ad.max().item():.3f} {ad.std().item():.3f}")
        # Compute raw entropy metric irrespective of ent_coef (mean over valid steps)
        with torch.no_grad():
            denom = mask.sum().clamp_min(1.0)
            ent_raw = (ent.sum() / denom).item()
        return {
            'loss_pi': L_pi.item(),
            'loss_v': L_v.item(),
            'ent': (-L_ent).item() if ent_coef != 0.0 else 0.0,
            'ent_raw': ent_raw,
            'loss_bc': L_bc.item() if bc_weight > 0.0 else 0.0,
        }

    #TODO additional anchors or regularizers can be added here
