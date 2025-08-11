# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


class DINOJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            if isinstance(x, NestedTensor):
                # Extract tensors for position encoding
                pos.append(self[1](x.tensors).to(x.tensors.dtype))
            else:
                pos.append(self[1](x).to(x.dtype))

        return out, pos


class DINOBackbone(nn.Module):
    """DINO Vision Transformer backbone."""
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        
        # Load pretrained DINO model
        if name == 'dino_vits16':
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
            self.num_channels = 384  # ViT-S embedding dimension
            self.patch_size = 16
        elif name == 'dino_vitb16':
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
            self.num_channels = 768  # ViT-B embedding dimension
            self.patch_size = 16
        else:
            raise ValueError(f"Unknown DINO model: {name}")
        
        # Set training mode for backbone
        if not train_backbone:
            for param in self.model.parameters():
                param.requires_grad_(False)
            # Ensure deterministic eval-time behavior when frozen
            self.model.eval()
            
        self.return_interm_layers = return_interm_layers
        
        # Number of intermediate layers to return when requested
        self._num_interm = 4 if return_interm_layers else 1
    
    def forward(self, tensor_list):
        if isinstance(tensor_list, NestedTensor):
            tensor = tensor_list.tensors
            mask = tensor_list.mask
        else:
            tensor = tensor_list
            mask = None
            
        # Use DINO's API to get intermediate layers robustly
        # We attempt to exclude the CLS token directly if supported
        try:
            features_list = self.model.get_intermediate_layers(
                tensor, n=self._num_interm, reshape=False, return_class_token=False
            )
        except TypeError:
            # Fallback for older API without return_class_token
            features_list = self.model.get_intermediate_layers(
                tensor, n=self._num_interm, reshape=False
            )
        if not isinstance(features_list, (list, tuple)):
            features_list = [features_list]

        # Compute spatial dims from input
        B = tensor.shape[0]
        _, _, orig_H, orig_W = tensor.shape
        H = orig_H // self.patch_size
        W = orig_W // self.patch_size

        out: Dict[str, NestedTensor] = {}

        # If we didn't request intermediates, keep only the last layer
        if not self.return_interm_layers:
            feat = features_list[-1]
            # If CLS token is present, drop it
            if feat.dim() == 3 and feat.shape[1] == 1 + H * W:
                feat = feat[:, 1:, :]
            # Validate token count
            assert feat.shape[1] == H * W, (
                f"Token count ({feat.shape[1]}) does not match expected H*W ({H*W}) "
                f"for input size {orig_H}x{orig_W} and patch size {self.patch_size}"
            )
            # Reshape to [B, D, H, W]
            D = feat.shape[2]
            feat = feat.permute(0, 2, 1).reshape(B, D, H, W)
            # Mask handling
            if mask is not None:
                final_mask = F.interpolate(mask[None].float(), size=(H, W)).to(torch.bool)[0]
            else:
                final_mask = torch.zeros((B, H, W), dtype=torch.bool, device=feat.device)
            out["0"] = NestedTensor(feat, final_mask)
            return out

        # Return intermediates: use all returned features in order
        for i, feat in enumerate(features_list):
            if feat.dim() == 3 and feat.shape[1] == 1 + H * W:
                feat = feat[:, 1:, :]
            assert feat.shape[1] == H * W, (
                f"Intermediate tokens ({feat.shape[1]}) do not match expected H*W ({H*W}) "
                f"for input size {orig_H}x{orig_W} and patch size {self.patch_size}"
            )
            D = feat.shape[2]
            feat_reshaped = feat.permute(0, 2, 1).reshape(B, D, H, W)
            if mask is not None:
                feat_mask = F.interpolate(mask[None].float(), size=(H, W)).to(torch.bool)[0]
            else:
                feat_mask = torch.zeros((B, H, W), dtype=torch.bool, device=feat_reshaped.device)
            out[str(i)] = NestedTensor(feat_reshaped, feat_mask)

        return out

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0 if args.lr_backbone is not None else False
    return_interm_layers = args.masks
    
    if args.backbone.startswith('dino'):
        backbone = DINOBackbone(args.backbone, train_backbone, return_interm_layers)
        model = DINOJoiner(backbone, position_embedding)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
    
    model.num_channels = backbone.num_channels
    return model
