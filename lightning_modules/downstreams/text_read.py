"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from typing import Dict, List, Union

from lightning_modules.downstreams.base import BaseDownstream


class TextRead(BaseDownstream):
    def __init__(self, dataset_item, dtd_key):
        super().__init__(dataset_item, dtd_key)

        self.pattern_dict = self._get_pattern_dict()

    def _get_pattern_dict(self) -> Dict:
        pattern_dict = {}
        return pattern_dict

    def _get_class_names(self) -> Union[List, None]:
        return None

    def parse_result_dict(self, decoder_type, pr_dict, gt_dict, kwargs) -> Dict:
        if decoder_type == "transformer_decoder":
            return self.parse_result_dict_transformer_decoder(pr_dict, gt_dict, kwargs)
        else:
            raise ValueError("Unsupported decoder type")

    def parse_result_dict_transformer_decoder(self, pr_dict, gt_dict, kwargs) -> Dict:
        res_dict = {}

        return res_dict

    @staticmethod
    def print_result(res_dict):
        pass
