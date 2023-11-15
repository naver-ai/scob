"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import abc
from typing import Dict, List, Union


class BaseDownstream(metaclass=abc.ABCMeta):
    def __init__(self, dataset_item, dtd_key):
        self.dataset_item = dataset_item
        self.dataset_name, self.task_name, self.decoder_name = dtd_key

    @abc.abstractmethod
    def _get_pattern_dict(self) -> Dict:
        """Get regex pattern dictionary"""

    @abc.abstractmethod
    def _get_class_names(self) -> Union[List, None]:
        """Get class names"""

    @abc.abstractmethod
    # self, decoder_type, pr_seq, gt_seq, pr_loc_output=None, gt_loc_output=None
    def parse_result_dict(self, decoder_type, pr_dict, gt_dict, kwargs) -> Dict:
        """Parse result dict"""

    @staticmethod
    @abc.abstractmethod
    def print_result(res_dict):
        """Print single result"""
