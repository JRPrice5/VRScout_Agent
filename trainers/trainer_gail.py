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

        # Slice first query for continuous distribution
        if hasattr(cont_dist, 'mu') and cont_dist.mu.dim() == 3:
            B, T, Cc = cont_dist.mu.shape
            mu_first = cont_dist.mu[:, 0, :]
            log_std_first = cont_dist.log_std[:, 0, :]
        else:
            # No time/query dimension; infer T from mask
            B = cont_dist.mu.shape[0]
            T = mask.shape[1] if mask.dim() == 2 else 1
            mu_first = cont_dist.mu
            log_std_first = cont_dist.log_std

        cont_first = DiagNormal(mu_first, log_std_first)
        # Align a_cont to [B, Cc]
        if a_cont.dim() == 3:
            a_cont_first = a_cont[:, 0, :]
        else:
            a_cont_first = a_cont

        cont_logp_first = cont_first.log_prob(a_cont_first)  # [B]
        ent_first = cont_first.entropy()  # [B]

        # Optional binary part on first query
        if bin_dist is not None and a_bin is not None:
            # Build a first-query Bernoulli distribution
            if hasattr(bin_dist, 'logits') and bin_dist.logits is not None:
                logits = bin_dist.logits
                logits_first = logits[:, 0, :] if logits.dim() == 3 else logits
                bin_first = BernoulliMulti(logits=logits_first)
            else:
                probs = bin_dist.probs
                probs_first = probs[:, 0, :] if probs.dim() == 3 else probs
                bin_first = BernoulliMulti(probs=probs_first)

            a_bin_first = a_bin[:, 0, :] if a_bin.dim() == 3 else a_bin
            cont_logp_first = cont_logp_first + bin_first.log_prob(a_bin_first)
            ent_first = ent_first + bin_first.entropy()

        # Place into [B, T] outputs with only first column set
        logp_BT = mask.new_zeros((B, T), dtype=cont_logp_first.dtype)
        ent_BT = mask.new_zeros((B, T), dtype=ent_first.dtype)
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
        # Final shape assertions
        assert r.shape == (B, T), f"r shape {tuple(r.shape)} != ({B},{T})"
        assert v_old.shape == (B, T), f"v_old shape {tuple(v_old.shape)} != ({B},{T})"
        assert dones.shape == (B, T), f"dones shape {tuple(dones.shape)} != ({B},{T})"
        assert mask.shape == (B, T), f"mask shape {tuple(mask.shape)} != ({B},{T})"

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
