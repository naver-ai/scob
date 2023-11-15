"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

from utils.config_manager import ConfigManager


def get_strategy():
    cfg = ConfigManager().cfg

    if cfg.train.strategy.type == "ddp":
        strategy = DDPPlugin(**cfg.train.strategy.ddp_kwargs)
    elif cfg.train.strategy.type == "deepspeed":
        strategy = DeepSpeedPlugin(**(cfg.train.strategy.deepspeed_kwargs))
    else:
        raise ValueError(f"train.strategy is not valid: {cfg.train.strategy}")

    return strategy
