"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.config_manager import ConfigManager
from utils.constants import DecoderTypes, Phase

from .gpu_stat_monitor import ModifiedGPUStatsMonitor
from .model_checkpoint import LastestModelCheckpoint
from .save_tokenizer import SaveTokenizer


def get_callbacks():
    cfg = ConfigManager().cfg

    # default callbacks
    callbacks = [
        LearningRateMonitor("step"),
        ModifiedGPUStatsMonitor(),
        SaveTokenizer(),
    ]

    # callback to save the lastest model at every epoch
    #   - Due to the backward compatibility, ModelCheckpoint uses
    #     monitor='val_loss' when monitor=None and 'val_loss' metric exists.
    #     To disable saving a model based on 'val_loss', use save_top_k=0.
    if cfg.train.save_epoch_last:
        cb = LastestModelCheckpoint(
            dirpath=cfg.save_weight_dir, save_top_k=0, save_last=True
        )
        cb.CHECKPOINT_NAME_LAST = "{epoch}-last"
        cb.FILE_EXTENSION = ".pt"
        callbacks.append(cb)

    # callbacks to save metric-based best checkpoints
    decoder_type_set = {decoder_cfg.type for decoder_cfg in cfg.model.decoders.values()}
    if cfg.phase == Phase.FINETUNING:
        if DecoderTypes.TRANSFORMER in decoder_type_set:
            ckpt_fn = "{epoch}_{val_loss:.2f}_{val_norm_ed:.2f}"
            monitor_mode_list = [("val_norm_ed", "min")]
        else:
            ckpt_fn = "{epoch}_{val_loss:.2f}"
            monitor_mode_list = [("val_loss", "min")]
    elif cfg.phase == Phase.PRETRAINING:
        ckpt_fn = "{epoch}_{val_loss:.2f}"
        monitor_mode_list = [("val_loss", "min")]
    else:
        raise ValueError(f"Unknown cfg.phase={cfg.phase}")

    if cfg.train.save_best:
        for monitor, mode in monitor_mode_list:
            filename = f"ckpt_best_{monitor.replace('/', '_')}_" + ckpt_fn

            cb = ModelCheckpoint(
                every_n_epochs=cfg.train.val_interval,
                dirpath=cfg.save_weight_dir,
                filename=filename,
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_weights_only=True,
            )
            cb.FILE_EXTENSION = ".pt"
            callbacks.append(cb)

    return callbacks
