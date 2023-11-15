"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import abc
import time

import torch
import torch.nn as nn
import torch.utils.data
from deepspeed.ops.adam import DeepSpeedCPUAdam
from overrides import overrides
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from lightning_modules.schedulers import cosine_scheduler, multistep_scheduler
from model import get_model
from model.model_utils import ImageUnnormalizer, get_parameter_names
from utils.config_manager import ConfigManager, cfg_to_hparams


class W(LightningModule, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager().cfg
        self.net = get_model()
        self.scheduler = None
        self.optimizer = None
        self.optimizer_types = {
            "sgd": SGD,
            "adam": Adam,
            "adamw": AdamW,
            "deepspeed_cpu_adam": DeepSpeedCPUAdam,
        }
        self.image_unnormalizer_dict = self._get_image_unnormalizer_dict()

    @rank_zero_only
    @overrides
    def on_fit_start(self):
        print(self.net)

    @torch.no_grad()
    @overrides
    def validation_step(self, batch, batch_idx, *args):
        loader_idx = 0 if len(args) == 0 else args[0]
        return self._val_test_step(batch, batch_idx, loader_idx, mode="val")

    @torch.no_grad()
    @overrides
    def test_step(self, batch, batch_idx, *args):
        loader_idx = 0 if len(args) == 0 else args[0]
        return self._val_test_step(batch, batch_idx, loader_idx, mode="test")

    @abc.abstractmethod
    def _val_test_step(self, batch, batch_idx, loader_idx, mode):
        """For validation and test step"""

    @torch.no_grad()
    @overrides
    def validation_epoch_end(self, validation_step_outputs):
        self._extract_results(validation_step_outputs, mode="val")

    @torch.no_grad()
    @overrides
    def test_epoch_end(self, test_step_outputs):
        self._extract_results(test_step_outputs, mode="test")

    @abc.abstractmethod
    def _extract_results(self, step_outputs, mode):
        """Extract logging results"""

    @overrides
    def configure_optimizers(self):
        self.optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(self.optimizer)
        self.scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
        }
        return [self.optimizer], [self.scheduler]

    def _get_lr_scheduler(self, optimizer):
        cfg_train = self.cfg.train
        lr_schedule_method = cfg_train.optimizer.lr_schedule.method
        lr_schedule_params = cfg_train.optimizer.lr_schedule.params

        if lr_schedule_method is None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule_method == "step":
            scheduler = multistep_scheduler(optimizer, **lr_schedule_params)
        elif lr_schedule_method == "cosine":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = cosine_scheduler(
                optimizer,
                training_steps=max_iter,
                **lr_schedule_params,
            )
        else:
            raise ValueError(
                f"lr schedule method {lr_schedule_method}" " is not supported."
            )

        return scheduler

    def _get_optimizer(self):
        opt_cfg = self.cfg.train.optimizer
        method = opt_cfg.method.lower()

        # check if the specified optimizer can be used
        if method not in self.optimizer_types:
            raise ValueError(f"[Optimizer: {method}] is not supported.")

        weight_decay = 0.0
        if "weight_decay" in opt_cfg.params:
            weight_decay = opt_cfg.params.weight_decay
        elif opt_cfg.method == "adamw" and "weight_decay" not in opt_cfg.params:
            # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            # The default weight_decay value of torch AdamW is 0.01
            weight_decay = 0.01

        if method == "deepspeed_cpu_adam":
            params_keyname = "model_params"
        else:
            params_keyname = "params"

        if weight_decay > 0:
            # Apply weight decay

            # Do not apply weight decay to LayerNorm and bias
            # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L836
            decay_params = get_parameter_names(self.net, [nn.LayerNorm])
            decay_params = [name for name in decay_params if "bias" not in name]

            optimizer_grouped_params = [
                {
                    params_keyname: [
                        p for n, p in self.net.named_parameters() if n in decay_params
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    params_keyname: [
                        p
                        for n, p in self.net.named_parameters()
                        if n not in decay_params
                    ],
                    "weight_decay": 0.0,
                },
            ]

            kwargs = dict(opt_cfg.params)
            kwargs["weight_decay"] = 0.0
            optimizer = self.optimizer_types[method](optimizer_grouped_params, **kwargs)
        else:
            kwargs = dict(opt_cfg.params)
            kwargs[params_keyname] = self.net.parameters()
            optimizer = self.optimizer_types[method](**kwargs)

        return optimizer

    @rank_zero_only
    @overrides
    def on_fit_end(self):
        hparam_dict = cfg_to_hparams(self.cfg, {})
        metric_dict = {"metric/dummy": 0}

        tb_logger = get_specific_pl_logger(self.logger, TensorBoardLogger)
        if tb_logger:
            tb_logger.log_hyperparams(hparam_dict, metric_dict)

    def _log_shell(self, log_info, prefix=""):
        """
        Logging the loss to the shell.
        Args:
            log_info (dict): Target dictionary to log.
                The key value must be string,
                The value value must be torch.Tensor.
            prefix (str): prefix of logging
        """
        steps_per_epoch = int(
            self.cfg.train.num_samples_per_epoch
            / (self.cfg.train.batch_size * self.trainer.world_size)
        )
        current_epoch = self.current_epoch
        current_step = int(self.global_step % steps_per_epoch)
        max_epochs = self.cfg.train.max_epochs

        log_info_shell = {}
        for k, v in log_info.items():
            new_v = v
            if type(new_v) is torch.Tensor:
                new_v = new_v.item()
            log_info_shell[k] = new_v

        out_str = prefix.upper()
        if prefix.upper().strip() in ["TRAIN", "VAL"]:
            out_str += f"[epoch: {current_epoch}/{max_epochs}"
            out_str += f" || iter: {current_step}]"

        if self.training:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_info_shell["lr"] = lr

        for key, value in log_info_shell.items():
            out_str += f" || {key}: {round(value, 5)}"
        out_str += f" || time: {round(time.time() - self.time_tracker, 1)}"
        out_str += " secs."
        self.print(out_str, flush=True)
        self.time_tracker = time.time()

    @staticmethod
    def _get_image_unnormalizer_dict():
        image_unnormalizer_dict = {}
        for image_normalize in [None, "imagenet_default", "imagenet_inception"]:
            image_unnormalizer_dict[image_normalize] = ImageUnnormalizer(
                image_normalize
            )
        return image_unnormalizer_dict

    def log_imgs(self, log_name, imgs, image_normalizes, step):
        unnormalized_img = []
        for img, image_normalize in zip(imgs, image_normalizes):
            unnormalized_img.append(
                self.image_unnormalizer_dict[image_normalize].unnormalize(
                    img.unsqueeze(0)
                )
            )
        unnormalized_img = torch.cat(unnormalized_img, 0)
        self.logger[0].experiment.add_images(
            log_name, unnormalized_img, step, dataformats="NCHW"
        )


def get_specific_pl_logger(pl_loggers, logger_type):
    """
    Selects and returns a specific type of logger from PyTorch Lightning logger.
    Args:
        pl_loggers: PyTorch Lightning Logger instances
        logger_type: PyTorch Lightning Logger class
            (e.g., pytorch_lightning.loggers.tensorboard.TensorBoardLogger)
    Return:
        Specific logger instance or None
    """
    for pl_logger in pl_loggers:
        if isinstance(pl_logger, logger_type):
            return pl_logger
    return None
