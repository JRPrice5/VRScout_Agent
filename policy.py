import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from action_utils import compute_mixed_loss, split_actions_by_type
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
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
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries].float()
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            # Compute mixed loss for continuous and boolean actions
            mixed_losses = compute_mixed_loss(a_hat, actions, is_pad, 
                                             continuous_weight=1.0, 
                                             boolean_weight=0.5)  # Lower weight for boolean actions
            
            loss_dict = dict()
            loss_dict['continuous_l1'] = mixed_losses['continuous_l1']
            loss_dict['boolean_bce'] = mixed_losses['boolean_bce']
            loss_dict['l1'] = mixed_losses['continuous_l1']  # Keep for backward compatibility
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = mixed_losses['total_loss'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

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
