from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict

from gail.discriminator import GAILDiscriminator, gail_loss_D, reward_from_prob, masked_mean
from rl.ppo import compute_gae, ppo_clip_loss, value_loss
from distributions import DiagNormal, BernoulliMulti


class TrainerGAIL:
    def __init__(self, policy, act_args, cont_dim, bin_dim, h_dim, device='cuda', cfg=None):
        self.policy = policy.to(device)
        self.device = device
        self.cfg = cfg or {}
        self.disc = GAILDiscriminator(h_dim=h_dim, cont_dim=cont_dim, bin_dim=bin_dim).to(device)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=self.cfg.get('lr_policy', 3e-4))
        self.opt_value = optim.Adam(list(self.policy.value_head.parameters()) +
                                    list(self.policy.cont_mu_head.parameters()) +
                                    list(self.policy.cont_log_std_head.parameters()) +
                                    list(self.policy.bin_logits_head.parameters()),
                                    lr=self.cfg.get('lr_policy', 3e-4))
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.cfg.get('lr_disc', 1e-4))

    def compute_logp_and_entropy(self, out_dist, a_cont, a_bin, mask):
        """Compute log-prob and entropy using ONLY the first query (index 0).
        Returns [B,T]-shaped tensors with values in the first column to match downstream masking.
        """
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
                # add time dim and broadcast along T
                x_bt = x.view(B_mask, 1, *x.shape[1:])
                return x_bt.expand(-1, T_mask, *([-1] * (x.dim() - 1)))
            # Fallback: if n != B or B*T, raise for clarity
            raise ValueError(f"Unexpected batch dimension {n} for tensor with shape {tuple(x.shape)}, expected {B_mask} or {B_mask*T_mask}")

        # Align continuous params to [B,T,Tq,C]
        if hasattr(cont_dist, 'mu') and cont_dist.mu.dim() == 3:
            mu_bt = reshape_BT(cont_dist.mu)          # [B,T,Tq,C]
            logstd_bt = reshape_BT(cont_dist.log_std) # [B,T,Tq,C]
            # Select first timestep and first query
            mu_first = mu_bt[:, 0, 0, :]              # [B,C]
            log_std_first = logstd_bt[:, 0, 0, :]     # [B,C]
        else:
            # No query dim; just [B,T,C]
            mu_bt = reshape_BT(cont_dist.mu)          # [B,T,C]
            logstd_bt = reshape_BT(cont_dist.log_std) # [B,T,C]
            mu_first = mu_bt[:, 0, :]
            log_std_first = logstd_bt[:, 0, :]

        cont_first = DiagNormal(mu_first, log_std_first)
        # Align a_cont to [B, Cc]
        if a_cont.dim() == 3:
            a_cont_bt = reshape_BT(a_cont)            # [B,T,Tq,C]
            a_cont_first = a_cont_bt[:, 0, 0, :]      # [B,C]
        elif a_cont.dim() == 2:
            a_cont_bt = reshape_BT(a_cont)            # [B,T,C]
            a_cont_first = a_cont_bt[:, 0, :]         # [B,C]
        else:
            raise ValueError(f"Unexpected a_cont shape {tuple(a_cont.shape)}")

        cont_logp_first = cont_first.log_prob(a_cont_first)  # [B]
        ent_first = cont_first.entropy()  # [B]

        # Optional binary part on first query
        if bin_dist is not None and a_bin is not None:
            # Build a first-query Bernoulli distribution
            if hasattr(bin_dist, 'logits') and bin_dist.logits is not None:
                logits = bin_dist.logits
                if logits.dim() == 3:
                    logits_bt = reshape_BT(logits)      # [B,T,Tq,Cb]
                    logits_first = logits_bt[:, 0, 0, :] # [B,Cb]
                elif logits.dim() == 2:
                    logits_bt = reshape_BT(logits)      # [B,T,Cb]
                    logits_first = logits_bt[:, 0, :]
                else:
                    raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")
                bin_first = BernoulliMulti(logits=logits_first)
            else:
                probs = bin_dist.probs
                if probs.dim() == 3:
                    probs_bt = reshape_BT(probs)        # [B,T,Tq,Cb]
                    probs_first = probs_bt[:, 0, 0, :]
                elif probs.dim() == 2:
                    probs_bt = reshape_BT(probs)        # [B,T,Cb]
                    probs_first = probs_bt[:, 0, :]
                else:
                    raise ValueError(f"Unexpected probs shape {tuple(probs.shape)}")
                bin_first = BernoulliMulti(probs=probs_first)

            # Align a_bin to [B, Cb]
            if a_bin.dim() == 3:
                a_bin_bt = reshape_BT(a_bin)           # [B,T,Tq,Cb]
                a_bin_first = a_bin_bt[:, 0, 0, :]
            elif a_bin.dim() == 2:
                a_bin_bt = reshape_BT(a_bin)           # [B,T,Cb]
                a_bin_first = a_bin_bt[:, 0, :]
            else:
                raise ValueError(f"Unexpected a_bin shape {tuple(a_bin.shape)}")
            cont_logp_first = cont_logp_first + bin_first.log_prob(a_bin_first)
            ent_first = ent_first + bin_first.entropy()

        # Place values only in the first time column to match "first-query" semantics
        T_out = T_mask
        logp_BT = mask.new_zeros((B_mask, T_out), dtype=cont_logp_first.dtype)
        ent_BT = mask.new_zeros((B_mask, T_out), dtype=ent_first.dtype)
        logp_BT[:, 0] = cont_logp_first
        ent_BT[:, 0] = ent_first

        return logp_BT * mask, ent_BT * mask

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
        loss_D = gail_loss_D(prob_exp, prob_pol, mask_exp)
        loss_D.backward()
        nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=5.0)
        self.opt_disc.step()
        with torch.no_grad():
            acc_exp = masked_mean((prob_exp > 0.5).float(), mask_exp)
            acc_pol = masked_mean((prob_pol <= 0.5).float(), mask_pol)
        self.disc.eval()
        return loss_D.item(), acc_exp.item(), acc_pol.item()

    def ppo_update(self, batch, clip_eps=0.2, gamma=0.99, lam=0.95, ent_coef=0.0, vf_coef=0.5,
                   bc_data=None, bc_weight: float = 0.0):
        self.policy.train()

        qpos = batch['obs']['qpos'].to(self.device)
        image = batch['obs']['image'].to(self.device)
        a_cont = batch['a_cont'].to(self.device)
        a_bin = batch.get('a_bin')
        a_bin = a_bin.to(self.device) if a_bin is not None else None
        mask = batch['mask'].to(self.device)
        logp_old = batch['logp'].to(self.device)
        v_old = batch['value'].to(self.device)
        r = batch['reward'].to(self.device)
        dones = batch['dones'].to(self.device)

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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Forward policy
        out = self.policy.forward(qpos, image, out_mode='dist')
        logp, ent = self.compute_logp_and_entropy(out, a_cont, a_bin, mask)
        v = out['value']

        # Losses
        L_pi = ppo_clip_loss(logp, logp_old, adv, clip_eps, mask)
        L_v = value_loss(v, ret, mask)
        L_ent = -(ent.sum() / mask.sum().clamp_min(1.0)) if ent_coef != 0.0 else v.new_tensor(0.0)
        loss = L_pi + vf_coef * L_v + ent_coef * L_ent

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
            loss = loss + bc_weight * L_bc

        self.opt_policy.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.cfg.get('grad_clip', 1.0))
        self.opt_policy.step()
        self.policy.eval()
        return {
            'loss_pi': L_pi.item(),
            'loss_v': L_v.item(),
            'ent': (-L_ent).item() if ent_coef != 0.0 else 0.0,
            'loss_bc': L_bc.item() if bc_weight > 0.0 else 0.0,
        }

    #TODO additional anchors or regularizers can be added here
