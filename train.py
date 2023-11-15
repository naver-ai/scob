"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import os

import cv2
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.utilities.seed import seed_everything

from lightning_modules.callbacks import get_callbacks
from lightning_modules.data_modules import DataModule
from lightning_modules.loggers import get_loggers
from lightning_modules.strategy import get_strategy
from lightning_modules.w_encoder_decoder import WEncDec
from utils.config_manager import ConfigManager


def main():
    """main train function"""
    cfg = ConfigManager().cfg
    cv2.setNumThreads(0)  # prevent deadlock with pytorch.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
    seed_everything(cfg.reproduce.seed, workers=True)

    strategy = get_strategy()
    callbacks = get_callbacks()

    trainer = Trainer(
        accelerator=cfg.train.accelerator,
        gpus=torch.cuda.device_count(),
        num_nodes=cfg.train.num_nodes,
        max_epochs=cfg.train.max_epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
        callbacks=callbacks,
        strategy=strategy,
        sync_batchnorm=True,
        precision=16 if cfg.train.use_fp16 else 32,
        detect_anomaly=False,
        auto_lr_find=False,
        replace_sampler_ddp=False,
        move_metrics_to_cpu=False,
        enable_progress_bar=False,
        # track_grad_norm=2,
        check_val_every_n_epoch=cfg.train.val_interval,
        logger=get_loggers(),
        log_every_n_steps=cfg.train.log_interval,
        benchmark=cfg.reproduce.cudnn_benchmark,
        deterministic=cfg.reproduce.cudnn_deterministic,
        profiler=get_profiler(cfg),
        limit_val_batches=cfg.val.limit_val_batches,
        limit_test_batches=cfg.test.limit_test_batches,
        num_sanity_val_steps=cfg.val.limit_val_batches,
    )

    # prepare model and data
    pl_module = WEncDec()
    data_module = DataModule(decoder_dict=pl_module.net.decoders.decoder_dict)

    if not cfg.test_only:
        trainer.fit(
            pl_module, datamodule=data_module, ckpt_path=cfg.model.resume_model_path
        )

    # run test
    trainer.test(
        pl_module, datamodule=data_module, ckpt_path=cfg.model.resume_model_path
    )


def get_profiler(cfg):
    """Get a profiler."""
    profiler_cfg = cfg.train.profiler
    if profiler_cfg.method is None:
        profiler = None
    else:
        profiler_class_selector = {
            "simple": SimpleProfiler,
            "advanced": AdvancedProfiler,
            "pytorch": PyTorchProfiler,
        }
        profiler_class = profiler_class_selector[profiler_cfg.method]
        profiler = profiler_class(
            path_to_export_trace=cfg.save_weight_dir, **profiler_cfg.params
        )
    return profiler


if __name__ == "__main__":
    main()
