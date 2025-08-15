from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict

from gail.discriminator import GAILDiscriminator, gail_loss_D, reward_from_prob, masked_mean
from rl.ppo import compute_gae, ppo_clip_loss, value_loss


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
        cont_logp = out_dist['cont_dist'].log_prob(a_cont)
        ent = out_dist['cont_dist'].entropy()
        if out_dist['bin_dist'] is not None and a_bin is not None:
            cont_logp = cont_logp + out_dist['bin_dist'].log_prob(a_bin)
            ent = ent + out_dist['bin_dist'].entropy()
        # sum over dims
        logp = cont_logp  # already summed over dims in distributions
        # mask and sum over time
        return logp * mask, ent * mask

    def update_discriminator(self, h_exp, a_cont_exp, a_bin_exp, mask_exp, h_pol, a_cont_pol, a_bin_pol, mask_pol):
        self.disc.train()
        self.opt_disc.zero_grad(set_to_none=True)
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

    def ppo_update(self, batch, clip_eps=0.2, gamma=0.99, lam=0.95, ent_coef=0.0, vf_coef=0.5):
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

        self.opt_policy.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.cfg.get('grad_clip', 1.0))
        self.opt_policy.step()
        self.policy.eval()
        return {
            'loss_pi': L_pi.item(),
            'loss_v': L_v.item(),
            'ent': (-L_ent).item() if ent_coef != 0.0 else 0.0,
        }
    
    #TODO BC anchor hook
