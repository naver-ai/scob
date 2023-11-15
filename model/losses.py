"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import torch
import torch.nn as nn
from kornia.losses import FocalLoss


def get_loss(loss_cfg):
    if loss_cfg.name == "ce":
        loss_func = nn.CrossEntropyLoss(reduction="none")
    elif loss_cfg.name == "focal":
        loss_func = FocalLoss(
            alpha=loss_cfg.focal_alpha,
            gamma=loss_cfg.focal_gamma,
            reduction="none",
        )
    elif loss_cfg.name == "l1":
        loss_func = nn.L1Loss(reduction="none")
    elif loss_cfg.name == "mse":
        loss_func = nn.MSELoss(reduction="none")
    else:
        raise ValueError(f"Unknown loss_cfg.name={loss_cfg.name}")
    return loss_func


class WeightLoss:
    """Weight loss per sample in each decoder"""

    def __init__(self, dataset_items):
        self.loss_weight_dict = {}
        for dataset_item in dataset_items:
            data_name = dataset_item.name
            for task in dataset_item.tasks:
                self.loss_weight_dict[(data_name, task.name)] = task.loss_weight

    def __call__(self, dec_batch, example_losses):
        weighted_losses = []
        for example_idx, (data_name, task) in enumerate(
            zip(dec_batch["dataset_names"], dec_batch["task_names"])
        ):
            loss_weight = self.loss_weight_dict[(data_name, task)]
            weighted_loss = loss_weight * example_losses[example_idx]
            weighted_losses.append(weighted_loss)
        return torch.stack(weighted_losses)
