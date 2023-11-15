"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import math

import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def cosine_scheduler(optimizer, warmup_steps, training_steps, cycles=0.5, last_step=-1):
    """Cosine LR scheduler with warmup from huggingface

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the lr.
        warmup_steps (int): The number of steps for the warmup phase.
        training_steps (int): The total number of training steps.
        cycles (float): The number of waves in the cosine schedule
            (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_step (int): The index of the last iter when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = current_step - warmup_steps
        progress /= max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_step)


def multistep_scheduler(optimizer, warmup_steps, milestones, gamma=0.1, last_step=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # calculate a warmup ratio
            return current_step / max(1, warmup_steps)
        else:
            # calculate a multistep lr scaling ratio
            idx = np.searchsorted(milestones, current_step)
            return gamma ** idx

    return LambdaLR(optimizer, lr_lambda, last_step)
