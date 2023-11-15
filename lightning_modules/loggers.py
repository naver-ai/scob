"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from utils.config_manager import ConfigManager
from utils.saver import prepare_tensorboard_log_dir


def get_loggers():
    """Get PL loggers"""
    loggers = []
    tb_logger = _get_tensorboard_logger()
    loggers.append(tb_logger)
    return loggers


def _get_tensorboard_logger():
    """Build a PL TensorBoardLogger"""
    prepare_tensorboard_log_dir()
    cfg = ConfigManager().cfg
    tb_logger = TensorBoardLogger(
        cfg.tensorboard_dir, name="", version="", default_hp_metric=False
    )
    return tb_logger
