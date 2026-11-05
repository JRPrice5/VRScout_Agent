import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional

from detr.main import build_ACT_model_and_optimizer
from action_utils import compute_continuous_loss
from distributions import TanhDiagNormal


class ACTPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(cfg)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = cfg.training.kl_weight
        print(f"KL Weight {self.kl_weight}")

        # Feature dims and heads (do not modify backbone/decoder/CVAE)
        hidden_dim = self.model.decoder.d_model
        self.hidden_dim = hidden_dim
        # Infer action splits
        self.action_dim = int(getattr(self.model, "action_dim", 22))
        # Stochastic heads
        self.mu_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    # TODO: wire environment state into the model
    def __call__(self, qpos, game_state, actions=None, is_pad=None, audio=None):
        # Convert inputs to float32 to match model weights
        qpos = qpos.float()
        if audio is not None:
            audio = audio.float()

        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries].float()
            is_pad = is_pad[:, : self.model.num_queries]

            # TODO: Remove controller inputs and xr origin position from model (indices 21:31)
            # TODO: Adjust for image->game_state
            a_hat, is_pad_hat, (mu, logvar, prior_mu, prior_logvar) = self.model(
                qpos, game_state, actions, is_pad, audio=audio
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(
                mu, logvar, prior_mu, prior_logvar
            )

            continuous_loss = compute_continuous_loss(a_hat, actions, is_pad)

            loss_dict = dict()
            loss_dict["continuous_l1"] = continuous_loss
            loss_dict["l1"] = continuous_loss
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = continuous_loss + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, stats = self.model(
                qpos, game_state, audio=audio
            )  # no action, sample from prior
            return a_hat

    def forward(
        self,
        qpos,
        game_state=None,
        audio=None,
        out_mode: str = "dist",
    ):
        """
        Enhanced forward that exposes ACT features and adds stochastic/value heads.
        - out_mode="dist": return distributions for continuous + value + hidden.
        - out_mode="mean" or "det": return means for actions + value + hidden.
        Shapes:
          h: [B, T, H], mean/log_std: [B, T, Cc], value: [B, T]
        """
        qpos = qpos.float()
        if audio is not None:
            audio = audio.float()

        # Compute hidden states h using the backbone/transformer
        # TODO: Adjust for image->game_state
        h = self.encode_to_hidden(qpos, game_state, audio=audio)  # [B, T, H]
        try:
            print("[DBG][FWD] h", tuple(h.shape))
        except Exception:
            pass

        # Heads
        try:
            print("[DBG][FWD] pre mu_head")
        except Exception:
            pass
        # Run heads in FP32 for numerical stability
        h_f32 = h.float()
        mu = self.mu_head(h_f32)
        try:
            print("[DBG][FWD] post mu_head")
        except Exception:
            pass
        # Clamp log_std tightly to keep std in a reasonable range: exp([-2.5, -0.3]) ≈ [0.082, 0.74]
        try:
            print("[DBG][FWD] pre log_std_head")
        except Exception:
            pass
        log_std = self.log_std_head(h_f32).clamp(min=-2.5, max=-0.3)
        try:
            print("[DBG][FWD] post log_std_head")
        except Exception:
            pass
        value = self.value_head(h_f32).squeeze(-1)
        try:
            print("[DBG][FWD] post value_head")
        except Exception:
            pass
        try:
            print("[DBG][FWD] pre action_head")
        except Exception:
            pass
        a_hat = self.model.action_head(
            h
        )  # [B, Tq, A]  (keep model's native dtype for decoder head)
        try:
            print("[DBG][FWD] post action_head")
        except Exception:
            pass
        try:
            print(
                "[DBG][FWD] mu",
                tuple(mu.shape),
                "log_std",
                tuple(log_std.shape),
                "value",
                tuple(value.shape),
                "ahat",
                tuple(a_hat.shape),
            )
        except Exception:
            pass

        if out_mode in ("dist", "distribution"):
            # Use squashed Gaussian to keep samples bounded in [-1,1]
            dist = TanhDiagNormal(mu, log_std, min_log_std=-2.5, max_log_std=-0.3)
            return {
                "a_hat": a_hat,
                "h": h,
                "dist": dist,
                "value": value,
            }
        else:
            # For unsquashed Gaussian, the mean is raw mu
            mean = mu
            return {
                "a_hat": a_hat,
                "h": h,
                "mean": mean,
                "value": value,
            }

    def encode_to_hidden(
        self,
        qpos,
        game_state=None,
        audio=None,
        microbatch: Optional[int] = None,
    ):
        """
        Expose per-step hidden features h_t without refactoring the backbone.
        Returns h of shape [B, T, H].
        """
        self.model.eval()  # feature extraction is inference-like
        device = "cuda"
        B = qpos.shape[0]
        mb = int(microbatch) if (microbatch is not None and microbatch > 0) else B
        outs = []
        with torch.no_grad():
            for s in range(0, B, mb):
                e = min(B, s + mb)
                q_mb = qpos[s:e]
                audio_mb = audio[s:e] if audio is not None else None

                # Move to device lazily per chunk
                q_mb = q_mb.to(device, non_blocking=True).float()
                if audio_mb is not None:
                    audio_mb = audio_mb.to(device, non_blocking=True).float()

                with autocast(enabled=True):
                    qpos_proj = self.model.input_proj_robot_state(q_mb)
                    env_proj = self.model.input_proj_game_state(game_state)

                    transformer_input = torch.cat([qpos_proj, env_proj], dim=1)

                    hs_mb = self.model.decoder(
                        transformer_input,
                        None,
                        self.model.query_embed.weight,
                        self.model.pos.weight,
                    )[0]

                outs.append(hs_mb)
        hidden_state = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        # hs is [B, T, H]
        return hidden_state

    def heads_from_h(self, h, out_mode: str = "dist"):
        """
        Run only stochastic/action/value heads from precomputed hidden h.
        h: [B, T, H]
        """
        h_f32 = h.float()
        mu = self.mu_head(h_f32)
        log_std = self.log_std_head(h_f32).clamp(min=-2.5, max=-0.3)
        value = self.value_head(h_f32).squeeze(-1)
        a_hat = self.model.action_head(h)  # keep model native dtype

        if out_mode in ("dist", "distribution"):
            dist = TanhDiagNormal(mu, log_std, min_log_std=-2.5, max_log_std=-0.3)
            return {
                "a_hat": a_hat,
                "h": h,
                "dist": dist,
                "value": value,
            }
        else:
            mean = mu
            return {
                "a_hat": a_hat,
                "h": h,
                "mean": mean,
                "value": value,
            }

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar, prior_mu=None, prior_logvar=None):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    if prior_mu is None or prior_logvar is None:
        # Standard Normal Prior N(0, I)
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    else:
        # General Gaussian Prior N(prior_mu, exp(prior_logvar))
        if prior_mu.data.ndimension() == 4:
            prior_mu = prior_mu.view(prior_mu.size(0), prior_mu.size(1))
        if prior_logvar.data.ndimension() == 4:
            prior_logvar = prior_logvar.view(prior_logvar.size(0), prior_logvar.size(1))

        prior_var = prior_logvar.exp()
        var = logvar.exp()

        # KL = 0.5 * (log(sigma_p^2) - log(sigma_q^2) + (sigma_q^2 + (mu_q - mu_p)^2)/sigma_p^2 - 1)
        klds = 0.5 * (
            prior_logvar - logvar + (var + (mu - prior_mu).pow(2)) / prior_var - 1
        )

    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
