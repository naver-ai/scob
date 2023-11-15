"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from collections import defaultdict
from functools import partial

from lightning_modules.data_modules.collates.base import BaseCollate
from lightning_modules.data_modules.collates.transformer_decoder import (
    TransformerDecoderCollate,
)
from utils.constants import DecoderTypes


class W_Collates(BaseCollate):
    def __init__(self, decoder_dict):
        super().__init__()
        type2collate = {
            DecoderTypes.TRANSFORMER: partial(
                TransformerDecoderCollate, decoder_dict=decoder_dict
            ),
        }

        self.collate_dict = {}
        for decoder_name in decoder_dict.keys():
            decoder_type = decoder_name.split("__")[0]
            self.collate_dict[decoder_name] = type2collate[decoder_type]()

    def __call__(self, instances):
        batch = {
            "imgs": [],
            "second_imgs": [],
            "dataset_names": [],
            "task_names": [],
            "decoder_names": [],
        }
        self.stack_batch(batch, instances)
        self.tensorize(batch)

        # per decoder
        indices_per_decoder = defaultdict(list)
        for example_idx, instance in enumerate(instances):
            decoder_name = instance["decoder_names"]
            indices_per_decoder[decoder_name].append(example_idx)
        batch["indices_per_decoder"] = indices_per_decoder

        for decoder_name, decoder_example_indices in batch[
            "indices_per_decoder"
        ].items():
            collate_fn = self.collate_dict[decoder_name]
            batch[decoder_name] = collate_fn(
                instances, decoder_name, decoder_example_indices
            )

        return batch
