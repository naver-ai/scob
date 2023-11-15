"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

# pylint: disable=missing-module-docstring,maybe-no-member
import time
from functools import partial

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lightning_modules.data_modules.collates import W_Collates
from lightning_modules.data_modules.datasets import (
    MetaDataset,
    TransformerDecoderDatasetForFineTuning,
    TransformerDecoderDatasetForPreTraining,
)
from lightning_modules.data_modules.samplers import (
    DistributedWeightedRandomDatasetSampler,
)
from utils.config_manager import ConfigManager
from utils.constants import Phase, Seperators


class DataModule(pl.LightningDataModule):
    def __init__(self, decoder_dict=None):
        super().__init__()
        self.cfg = None
        self.dataset_class_dict = None
        self.train_meta_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.decoder_dict = decoder_dict
        self.collate_fn = None

    @overrides
    def prepare_data(self, *args, **kwargs):
        """Nothing to prepare"""

    @overrides
    def setup(self, stage=None):
        """Initialization codes AFTER the configuration for distributed
        training.
        """
        self.cfg = ConfigManager().cfg

        if self.cfg.phase in [Phase.FINETUNING, Phase.PRETRAINING]:
            self.collate_fn = W_Collates(self.decoder_dict)

        self.dataset_class_dict = self._get_dataset_class_dict()
        self.train_loader = self._get_train_loader()
        self.val_loader = self._get_val_test_loaders(mode="val")
        self.test_loader = self._get_val_test_loaders(mode="test")

    @overrides
    def train_dataloader(self):
        return self.train_loader

    @overrides
    def val_dataloader(self):
        return self.val_loader

    @overrides
    def test_dataloader(self):
        return self.test_loader

    def _get_dataset_class_dict(self):
        dataset_class_dict = {}
        for dtd_key_str, dtd_item in self.cfg.dtd_dict.items():
            dataset_name, task_name, decoder_name = dtd_key_str.split(Seperators.DTD)
            if self.cfg.phase == Phase.PRETRAINING:
                if decoder_name.startswith("transformer_decoder"):
                    dataset_class_dict[dtd_key_str] = partial(
                        TransformerDecoderDatasetForPreTraining,
                        decoder_dict=self.decoder_dict,
                    )
            else:
                if decoder_name.startswith("transformer_decoder"):
                    dataset_class_dict[dtd_key_str] = partial(
                        TransformerDecoderDatasetForFineTuning,
                        decoder_dict=self.decoder_dict,
                    )
        return dataset_class_dict

    def _get_train_loader(self):
        """Get a training data loader."""
        world_size = self.trainer.world_size
        global_rank = self.trainer.global_rank

        start_time = time.time()

        # load datasets and their batch ratios
        datasets = []
        weights = []
        for dtd_key_str, dtd_item in self.cfg.dtd_dict.items():
            data_class = self.dataset_class_dict[dtd_key_str]
            dataset = data_class(
                dtd_key_str=dtd_key_str, dtd_item=dtd_item, mode="train"
            )
            datasets.append(dataset)
            weights.append(dtd_item["batch_ratio"])

        self.train_meta_dataset = MetaDataset(
            datasets, self.trainer.global_rank, self.trainer.world_size
        )

        # generate a per-gpu sampler
        #   - This sampler must use the SAME random seed so that multiple
        #     GPUs can generate the same sample index list and split it
        #     into exclusive per-gpu sample index lists.
        #   - Shuffle is done within each leaf dataset.
        sampler = DistributedWeightedRandomDatasetSampler(
            weights,
            num_datasets=len(datasets),
            num_samples=self.cfg.train.num_samples_per_epoch,
            replacement=True,
            world_size=world_size,
            rank=global_rank,
            seed=self.cfg.reproduce.seed,
            drop_last=True,
        )

        # create dataloader
        # Caution!
        #   - Use persistent_workers=True to make worker_init_fn runs only
        #     once. Otherwise, it will RESET numpy random number generator
        #     and will reuse the same data all the time.
        data_loader = DataLoader(
            self.train_meta_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            sampler=sampler,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=not self.cfg.debug,
        )

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for loading training data: {elapsed_time}", flush=True)

        return data_loader

    def _get_val_test_loaders(self, mode):
        """Get a validation/test data laoder."""
        data_loaders = []

        for dtd_key_str, dtd_item in self.cfg.dtd_dict.items():
            data_class = self.dataset_class_dict[dtd_key_str]
            dataset = data_class(dtd_key_str=dtd_key_str, dtd_item=dtd_item, mode=mode)
            sampler = DistributedSampler(dataset, shuffle=False)
            data_loader = DataLoader(
                dataset,
                batch_size=self.cfg[mode].batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                sampler=sampler,
                num_workers=self.cfg[mode].num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
            )
            data_loaders.append(data_loader)
        return data_loaders

    @overrides
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Do not works well. Every data keys loaded to gpu device"""
        for key in batch.keys():
            if isinstance(batch[key], dict):
                for sub_key in batch[key].keys():
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(device)
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch
