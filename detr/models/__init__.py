# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae


def build_ACT_model(cfg):
    return build_vae(cfg)
