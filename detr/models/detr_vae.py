# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from act_bs.agent.data_utils import load_data

from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    build_transformer,
)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# Determine what needs to be positionally embedded - action sequence
class DETRVAE(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        decoder,
        encoder,
        action_dim,
        obstacle_dim,
        metadata_dim,
        num_queries,
        max_obs,
    ):
        """Initializes the model.
        Parameters:
            game_state_mlp: game_state_encoder.
            decoder: torch module of the decoder architecture. See transformer.py
            action_dim: the size of each prediction
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.decoder = decoder
        self.encoder = encoder
        self.action_dim = action_dim  # Store action dimension for consistent use
        self.obstacle_dim = obstacle_dim
        self.metadata = metadata_dim
        self.hidden_dim = decoder.d_model
        self.action_head = nn.Linear(self.hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        self.input_proj = nn.LazyLinear(self.hidden_dim)
        self.qpos_proj = nn.Linear(action_dim, self.hidden_dim)
        self.obstacle_proj = nn.Linear(obstacle_dim, self.hidden_dim)
        self.metadata_proj = nn.Linear(metadata_dim, self.hidden_dim)
        self.pos = torch.nn.Embedding(max_obs + 3, self.hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z
        self.cls_embed = nn.Embedding(1, self.hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            action_dim, self.hidden_dim
        )  # project action to embedding
        self.latent_proj = nn.Linear(
            self.hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, self.hidden_dim),
        )  # [CLS], qpos, a_seq

        # Print action_dim for debugging purpose
        print(f"[PolicyVAE] State dimension: {action_dim}")

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, self.hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, self.hidden_dim
        )  # learned position embedding for proprio, latent

    def forward(self, metadata, obstacles, qpos, actions=None, obstacle_mask=None):
        """
        metadata: batch, state
        obstacle_data: batch, obs_window, obstacle_dim
        qpos: batch, qpos_dim
        actions: batch, chunk, action_dim
        """
        is_training = actions is not None  # train or val
        batch_size, _ = qpos.shape

        prior_mean = torch.zeros((batch_size, self.latent_dim), device=qpos.device)
        prior_logvar = torch.zeros((batch_size, self.latent_dim), device=qpos.device)

        qpos_embed = self.qpos_proj(qpos)  # (bs, self.hidden_dim)
        qpos_embed = torch.unsqueeze(qpos_embed, dim=1)  # (bs, 1, self.hidden_dim)
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(
                actions
            )  # (bs, seq, self.hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, self.hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(
                batch_size, 1, 1
            )  # (bs, 1, self.hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], dim=1
            )  # (bs, seq+1, self.hidden_dim)
            encoder_input = encoder_input.permute(
                1, 0, 2
            )  # (seq+1, bs, self.hidden_dim)
            # do not mask cls token
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mean = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mean, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mean = logvar = None
            # Use prior mean for inference
            latent_sample = prior_mean
            latent_input = self.latent_out_proj(latent_sample)

        metadata_embed = self.metadata_proj(metadata)
        obstacles_embed = self.obstacle_proj(obstacles)

        decoder_input = torch.cat(
            [
                obstacles_embed,
                torch.unsqueeze(metadata_embed, 1),
                qpos_embed,
                torch.unsqueeze(latent_input, 1),
            ],
            axis=1,
        )
        obstacle_mask = torch.cat(
            [
                obstacle_mask,
                torch.zeros((batch_size, 3), device="cuda", dtype=bool),
            ],
            axis=1,
        )

        hidden_state = self.decoder(
            decoder_input,
            obstacle_mask,
            self.query_embed.weight,
            self.pos.weight,
        )[
            0
        ]  # src,mask,query_embed,pos_embed,latent_input,proprio_input,additional_pos_embed

        a_hat = self.action_head(hidden_state)
        return a_hat, [mean, logvar, prior_mean, prior_logvar]


def build_encoder(cfg):
    d_model = cfg.model.hidden_dim  # 256
    dropout = cfg.model.dropout  # 0.1
    nhead = cfg.model.nheads  # 8
    dim_feedforward = cfg.model.dim_feedforward  # 2048
    num_encoder_layers = cfg.model.enc_layers  # 4
    normalize_before = cfg.model.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(cfg):
    decoder = build_transformer(cfg)

    encoder = build_encoder(cfg)

    model = DETRVAE(
        decoder,
        encoder,
        action_dim=cfg.data.action_length,
        obstacle_dim=cfg.data.obstacle_length,
        metadata_dim=cfg.data.metadata_length,
        num_queries=cfg.model.chunk_size,
        max_obs=cfg.data.max_obstacles,
    )

    train_dataloader, _, _ = load_data(
        cfg.data.hdf5_dir,
        10,
        2,
        2,
        cfg.model.chunk_size,
    )

    dummy = next(iter(train_dataloader))
    metadata, obstacle_data, qpos, action, obstacle_mask = dummy
    metadata, obstacle_data, qpos, action, obstacle_mask = (
        metadata.cuda(),
        obstacle_data.cuda(),
        qpos.cuda(),
        action.cuda(),
        obstacle_mask.cuda(),
    )
    with torch.no_grad():
        model(metadata, obstacle_data, qpos, action, obstacle_mask)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
