# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .models import build_ACT_model


def build_ACT_model_and_optimizer(cfg):
    model = build_ACT_model(cfg)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            # "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    return model, optimizer
