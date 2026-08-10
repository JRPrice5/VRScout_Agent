# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
from torch import nn

from .gamestate_mlp import mlp


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, input: torch.Tensor):
        out = self[0](input)
        pos = self[1](input).to(input.dtype)
        return out, pos


# What are args? Simply model parameters, passed through the command line or via defaults
def build_backbone(args):

    model = Joiner(mlp(), position_embedding)

    return model
