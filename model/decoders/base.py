"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import abc

import torch.nn as nn

from lightning_modules.downstreams import DownstreamEvaluator


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    downstream_evaluator = DownstreamEvaluator()

    @abc.abstractmethod
    def get_step_out_dict(
        self,
        batch,
        batch_idx,
        loader_idx,
        dec_out_dict,
        enc_kwargs,
        dec_kwargs,
    ):
        """val-test step"""
