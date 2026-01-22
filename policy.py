import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from typing import Optional

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from action_utils import compute_mixed_loss, split_actions_by_type, ACTION_INDICES
from distributions import TanhDiagNormal, BernoulliMulti
import IPython
e = IPython.embed

class ACTPolicy(nn.Module): #TODO check whether image normalization (transforms.Normalize )is correct for our image shape
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

        # Feature dims and heads (do not modify backbone/transformer/CVAE)
        hidden_dim = self.model.transformer.d_model
        self.hidden_dim = hidden_dim
        # Infer action splits
        action_dim = int(getattr(self.model, 'action_dim', 32))
        if action_dim < 28:
            raise ValueError(f"action_dim too small ({action_dim}); expected at least 28")
        # continuous: 0..23 plus 28..(action_dim-1)
        self.cont_dim = 24 + max(0, action_dim - 28)
        # boolean: 24..27
        self.bin_dim = 4
        # Stochastic heads
        self.cont_mu_head = nn.Linear(hidden_dim, self.cont_dim)
        self.cont_log_std_head = nn.Linear(hidden_dim, self.cont_dim)
        self.bin_logits_head = nn.Linear(hidden_dim, self.bin_dim)
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, qpos, image, actions=None, is_pad=None, audio=None):
        env_state = None
        # Print image shape and info before normalization
        print(f"[Policy] Input image shape: {image.shape}, dtype: {image.dtype}")
        # print(f"[Policy] Image min: {image.min()}, max: {image.max()}, mean: {image.mean()}")
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        try:
            image = normalize(image)
            # print("[Policy] Normalization successful")
        except Exception as e:
            print(f"[Policy] Normalization error: {e}")
            raise
        # Convert inputs to float32 to match model weights
        qpos = qpos.float()
        if audio is not None:
            audio = audio.float()
            
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries].float()
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, audio=audio)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            # Compute mixed loss for continuous and boolean actions
            mixed_losses = compute_mixed_loss(a_hat, actions, is_pad, 
                                             continuous_weight=1.0, 
                                             boolean_weight=0.0)  # Lower weight for boolean actions
            
            loss_dict = dict()
            loss_dict['continuous_l1'] = mixed_losses['continuous_l1']
            loss_dict['boolean_bce'] = mixed_losses['boolean_bce']
            loss_dict['l1'] = mixed_losses['continuous_l1']  # Keep for backward compatibility
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = mixed_losses['total_loss'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, audio=audio) # no action, sample from prior
            return a_hat

    def forward(self, qpos, image, prev_actions=None, is_pad=None, audio=None, out_mode: str = "dist"):
        """
        Enhanced forward that exposes ACT features and adds stochastic/value heads.
        - out_mode="dist": return distributions for continuous and binary actions + value + hidden.
        - out_mode="mean" or "det": return means for actions + value + hidden.
        Shapes:
          h: [B, T, H], cont_mean/log_std: [B, T, Cc], bin_logits: [B, T, Cb], value: [B, T]
        """
        env_state = None
        # Normalize image; support 4D/5D
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if image.dim() == 5:
            B, K, C, H, W = image.shape
            image_flat = image.view(B * K, C, H, W).contiguous()
            image_flat = normalize(image_flat)
            image = image_flat.view(B, K, C, H, W)
        elif image.dim() == 4:
            image = normalize(image)
        else:
            raise ValueError(f"Unsupported image dims {image.dim()}, expected 4D or 5D tensor")
        qpos = qpos.float()
        if audio is not None:
            audio = audio.float()

        # Compute hidden states h using the backbone/transformer
        # forward() already normalized image, so skip extra normalization inside encoder
        h = self.encode_to_hidden(qpos, image, env_state, audio=audio, do_normalize=False)  # [B, T, H]
        try:
            print('[DBG][FWD] h', tuple(h.shape))
        except Exception:
            pass

        # Heads
        try:
            print('[DBG][FWD] pre cont_mu_head')
        except Exception:
            pass
        # Run heads in FP32 for numerical stability
        h_f32 = h.float()
        cont_mu = self.cont_mu_head(h_f32)
        try:
            print('[DBG][FWD] post cont_mu_head')
        except Exception:
            pass
        # Clamp log_std tightly to keep std in a reasonable range: exp([-2.5, -0.3]) ≈ [0.082, 0.74]
        try:
            print('[DBG][FWD] pre cont_log_std_head')
        except Exception:
            pass
        cont_log_std = self.cont_log_std_head(h_f32).clamp(min=-2.5, max=-0.3)
        try:
            print('[DBG][FWD] post cont_log_std_head')
        except Exception:
            pass
        bin_logits = None
        if self.bin_dim > 0:
            try:
                print('[DBG][FWD] pre bin_logits_head')
            except Exception:
                pass
            bin_logits = self.bin_logits_head(h_f32)
            try:
                print('[DBG][FWD] post bin_logits_head')
            except Exception:
                pass
        try:
            print('[DBG][FWD] pre value_head')
        except Exception:
            pass
        value = self.value_head(h_f32).squeeze(-1)
        try:
            print('[DBG][FWD] post value_head')
        except Exception:
            pass
        try:
            print('[DBG][FWD] pre action_head')
        except Exception:
            pass
        a_hat = self.model.action_head(h)  # [B, Tq, A]  (keep model's native dtype for decoder head)
        try:
            print('[DBG][FWD] post action_head')
        except Exception:
            pass
        try:
            print('[DBG][FWD] cont_mu', tuple(cont_mu.shape), 'log_std', tuple(cont_log_std.shape),
                  'bin', None if bin_logits is None else tuple(bin_logits.shape), 'value', tuple(value.shape), 'ahat', tuple(a_hat.shape))
        except Exception:
            pass

        if out_mode in ("dist", "distribution"):
            # Use squashed Gaussian to keep samples bounded in [-1,1]
            cont_dist = TanhDiagNormal(cont_mu, cont_log_std, min_log_std=-2.5, max_log_std=-0.3)
            bin_dist = BernoulliMulti(bin_logits) if bin_logits is not None else None
            return {
                'a_hat': a_hat,
                'h': h,
                'cont_dist': cont_dist,
                'bin_dist': bin_dist,
                'value': value,
            }
        else:
            # For unsquashed Gaussian, the mean is raw mu
            cont_mean = cont_mu
            bin_prob = torch.sigmoid(bin_logits) if bin_logits is not None else None
            return {
                'a_hat': a_hat,
                'h': h,
                'cont_mean': cont_mean,
                'bin_prob': bin_prob,
                'value': value,
            }

    def encode_to_hidden(self, qpos, image, env_state=None, audio=None, do_normalize: bool = True, microbatch: Optional[int] = None):
        """
        Expose per-step hidden features h_t without refactoring the backbone.
        - Optionally normalize images (ImageNet stats) if do_normalize=True.
        - Optionally process in microbatches over the batch dimension to reduce peak VRAM.
        Returns h of shape [B, T, H].
        """
        self.model.eval()  # feature extraction is inference-like
        device = next(self.parameters()).device
        B = qpos.shape[0]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        mb = int(microbatch) if (microbatch is not None and microbatch > 0) else B
        outs = []
        with torch.no_grad():
            for s in range(0, B, mb):
                e = min(B, s + mb)
                q_mb = qpos[s:e]
                img_mb = image[s:e]
                audio_mb = audio[s:e] if audio is not None else None
                
                # Move to device lazily per chunk
                q_mb = q_mb.to(device, non_blocking=True).float()
                if audio_mb is not None:
                    audio_mb = audio_mb.to(device, non_blocking=True).float()
                    
                # Support 5D [B,K,C,H,W] and 4D [B,C,H,W]
                if img_mb.dim() == 5:
                    Bm, K, C, H, W = img_mb.shape
                    img_flat = img_mb.view(Bm * K, C, H, W).contiguous().to(device, non_blocking=True).float()
                    if do_normalize:
                        img_flat = normalize(img_flat)
                    img_mb = img_flat.view(Bm, K, C, H, W)
                elif img_mb.dim() == 4:
                    img_mb = img_mb.contiguous().to(device, non_blocking=True).float()
                    if do_normalize:
                        img_mb = normalize(img_mb)
                else:
                    raise ValueError(f"Unsupported image dims {img_mb.dim()}, expected 4D or 5D tensor")

                if getattr(self.model, 'backbones', None) is not None:
                    all_cam_features, all_cam_pos = [], []
                    # Strict check: batch camera count must match model expectation
                    K = img_mb.shape[1]
                    N = len(self.model.camera_names)
                    assert K == N, f"Camera count mismatch: batch has K={K} images, model expects N={N} (camera_names). Align data or config."
                    with autocast(enabled=True):
                        for cam_id, _ in enumerate(self.model.camera_names):
                            x = img_mb[:, cam_id].contiguous()
                            try:
                                print(f"[DBG][ENC] cam {cam_id} x {tuple(x.shape)} dtype={x.dtype} contiguous={x.is_contiguous()}")
                            except Exception:
                                pass
                            features, pos = self.model.backbones[0](x)  # HARDCODED like model.forward
                            features = features[-1]
                            pos = pos[-1]
                            all_cam_features.append(self.model.input_proj(features))
                            all_cam_pos.append(pos)
                        proprio_input = self.model.input_proj_robot_state(q_mb)
                        
                        # Handle audio input
                        if audio_mb is not None:
                            audio_input = self.model.input_proj_audio(audio_mb)
                        else:
                            audio_input = torch.zeros_like(proprio_input)
                            
                        src = torch.cat(all_cam_features, axis=3)
                        pos = torch.cat(all_cam_pos, axis=3)
                        # Create zero latent and project to match model.forward
                        latent = torch.zeros((q_mb.size(0), getattr(self.model, 'latent_dim', 32)), device=q_mb.device, dtype=q_mb.dtype)
                        latent_input = self.model.latent_out_proj(latent) if hasattr(self.model, 'latent_out_proj') else None
                        try:
                            print("[DBG][ENC] src", tuple(src.shape), "pos", tuple(pos.shape),
                                  "qe", tuple(self.model.query_embed.weight.shape),
                                  "add_pos", tuple(self.model.additional_pos_embed.weight.shape),
                                  "lat", tuple(latent_input.shape), "prop", tuple(proprio_input.shape))
                        except Exception:
                            pass
                        try:
                            print("[DBG][ENC] pre-transformer")
                        except Exception:
                            pass
                        hs_mb = self.model.transformer(
                            src, None, self.model.query_embed.weight, pos,
                            latent_input,
                            proprio_input,
                            self.model.additional_pos_embed.weight,
                            audio_input=audio_input
                        )[0]
                        try:
                            print("[DBG][ENC] post-transformer")
                        except Exception:
                            pass
                else:
                    # Fallback to state-only path
                    with autocast(enabled=True):
                        qpos_proj = self.model.input_proj_robot_state(q_mb)
                        env_proj = self.model.input_proj_env_state(env_state)
                        if audio_mb is not None:
                            audio_input = self.model.input_proj_audio(audio_mb)
                            transformer_input = torch.cat([qpos_proj, env_proj, audio_input], axis=1)
                        else:
                            transformer_input = torch.cat([qpos_proj, env_proj], axis=1)
                        hs_mb = self.model.transformer(transformer_input, None, self.model.query_embed.weight, self.model.pos.weight)[0]

                outs.append(hs_mb)
        hs = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        # hs is [B, T, H]
        return hs

    def heads_from_h(self, h, out_mode: str = "dist"):
        """
        Run only stochastic/action/value heads from precomputed hidden h.
        h: [B, T, H]
        """
        h_f32 = h.float()
        cont_mu = self.cont_mu_head(h_f32)
        cont_log_std = self.cont_log_std_head(h_f32).clamp(min=-2.5, max=-0.3)
        bin_logits = self.bin_logits_head(h_f32) if self.bin_dim > 0 else None
        value = self.value_head(h_f32).squeeze(-1)
        a_hat = self.model.action_head(h)  # keep model native dtype

        if out_mode in ("dist", "distribution"):
            cont_dist = TanhDiagNormal(cont_mu, cont_log_std, min_log_std=-2.5, max_log_std=-0.3)
            bin_dist = BernoulliMulti(bin_logits) if bin_logits is not None else None
            return {
                'a_hat': a_hat,
                'h': h,
                'cont_dist': cont_dist,
                'bin_dist': bin_dist,
                'value': value,
            }
        else:
            cont_mean = cont_mu
            bin_prob = torch.sigmoid(bin_logits) if bin_logits is not None else None
            return {
                'a_hat': a_hat,
                'h': h,
                'cont_mean': cont_mean,
                'bin_prob': bin_prob,
                'value': value,
            }

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        # Convert inputs to float32 to match model weights
        qpos = qpos.float()
        if actions is not None: # training time
            actions = actions[:, 0].float()
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
