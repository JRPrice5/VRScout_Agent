# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""

import torch
from torch import nn
from act_bs.config.root import RootConfig
from omegaconf import OmegaConf


cfg = OmegaConf.structured(RootConfig)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats):
        super().__init__()
        self.embed = nn.Embedding(cfg.data.chunk_size, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, input: torch.Tensor):
        return self.embed(input)


def build_position_encoding(args):
    return PositionEmbeddingLearned(args.hidden_dim)
