# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# Various positional encodings for the transformer.
# """
#
# import torch
# from torch import nn
#
#
# # TODO: migrate learned embeddings to MLP
# # What needs positional embeddings? Action steps need embeddings, Obstacles may need embeddings or none
# class PositionEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """
#
#     def __init__(self, num_pos_feats):
#         super().__init__()
#         # TODO: replace magic number with prediction horizon length
#         self.embed = nn.Embedding(50, num_pos_feats)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # TODO: Why reset parameters? So that past attempts don't bleed into current(?)
#         # What are the parameters? They are the weights of the positional embedding network
#         nn.init.uniform_(self.embed.weight)
#
#     def forward(self, input: torch.Tensor):
#         return self.embed(input)
#
#
# def build_position_encoding(args):
#     return PositionEmbeddingLearned(args.hidden_dim)
