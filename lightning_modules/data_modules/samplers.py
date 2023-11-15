"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

# pylint: disable=missing-module-docstring
import math

import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedWeightedRandomDatasetSampler(Sampler):
    """WeightRandomSampler that samples `dataset` index, not `sample` index.

    This code is based on PyTorch's DistributedSampler.
    """

    # pylint: disable=W0231
    def __init__(
        self,
        weights,
        num_datasets,
        num_samples,
        replacement=True,
        world_size=None,
        rank=None,
        seed=0,
        drop_last=False,
    ):
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        weights = np.asarray(weights)

        self.weights = weights / weights.sum()
        self.num_datasets = num_datasets
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        # calculate the number of per-gpu samples
        if self.drop_last and self.num_samples % self.world_size != 0:
            # self.num_samples_per_gpu * self.world_size  < self.num_samples
            self.num_samples_per_gpu = math.ceil(
                (self.num_samples - self.world_size) / self.world_size
            )
        else:
            # self.num_samples_per_gpu * self.world_size  >= self.num_samples
            self.num_samples_per_gpu = math.ceil(self.num_samples / self.world_size)

        self.total_size = self.num_samples_per_gpu * self.world_size

    def __iter__(self):
        # set the same seed across all GPUs
        rng = np.random.default_rng(self.seed + self.epoch)

        # sample `dataset` indices for num_samples times
        indices = rng.choice(
            a=self.num_datasets,
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples_per_gpu

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_gpu

    def set_epoch(self, epoch):
        self.epoch = epoch
