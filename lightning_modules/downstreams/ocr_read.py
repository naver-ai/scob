"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import re
from typing import Dict, List, Union

import numpy as np

from lightning_modules.downstreams.base import BaseDownstream
from cleval import CLEvalMetric


class OCRRead(BaseDownstream):
    def __init__(self, dataset_item, dtd_key):
        super().__init__(dataset_item, dtd_key)
        self.cleval_metric = CLEvalMetric()
        self.pattern_dict = self._get_pattern_dict()

        self.head_type = "base"
        if self.task_name.endswith("2head"):
            self.head_type = "2head"

    def _get_pattern_dict(self) -> Dict:
        pattern_dict = {
            "base": {
                "normal": re.compile(
                    "\[START_BOX\]\s*(\d+) (\d+) (\d+) (\d+) (.*?)\s*\[END_BOX\]"
                ),
                "special": re.compile(
                    "\[START_BOX\]\s*\[POS_(\d+)\] \[POS_(\d+)\] "
                    "\[POS_(\d+)\] \[POS_(\d+)\] (.*?)\s*\[END_BOX\]"
                ),
                "special_char": re.compile(
                    "\[START_BOX\]\s*\[POS_(\d+)\]\[POS_(\d+)\]"
                    "\[POS_(\d+)\]\[POS_(\d+)\](.*?)\s*\[END_BOX\]"
                ),
            },
            "2head": re.compile("\[START_TEXT_BLOCK\]\s*(.*?)\s*\[END_TEXT_BLOCK\]"),
        }
        return pattern_dict

    def _get_class_names(self) -> Union[List, None]:
        return None

    def parse_result_dict(self, decoder_type, pr_dict, gt_dict, kwargs) -> Dict:
        if decoder_type == "transformer_decoder":
            return self.parse_result_dict_transformer_decoder(pr_dict, gt_dict, kwargs)
        else:
            raise ValueError("Unsupported decoder type")

    def parse_result_dict_transformer_decoder(self, pr_dict, gt_dict, kwargs):
        pr_seq = pr_dict["pr_seq"]
        pr_loc_output = pr_dict["pr_loc_output"]
        gt_seq = gt_dict["gt_seq"]
        gt_loc_output = gt_dict["gt_loc_output"]

        # Make it True when you want to see the inference result
        if False:
            print(f"img_path = \"{kwargs['img_path']}\"")
            print(f'pr_seq = "{pr_seq}"')
            print(f'gt_seq = "{gt_seq}"')
            print(f"pr_loc_output = {pr_loc_output}")
            print(f"gt_loc_output = {gt_loc_output}")
            print()

        if self.head_type == "base":
            (
                pr_quads,
                gt_quads,
                pr_letters,
                gt_letters,
                gt_is_dcs,
            ) = self._get_cleval_inp_base(pr_seq, gt_seq, kwargs)
        elif self.head_type == "2head":
            (
                pr_quads,
                gt_quads,
                pr_letters,
                gt_letters,
                gt_is_dcs,
            ) = self._get_cleval_inp_2head(pr_seq, gt_seq, pr_loc_output, gt_loc_output)
        else:
            raise ValueError(f"Unknown self.head_type={self.head_type}")

        # det_r, det_p, det_h, e2e_r, e2e_p, e2e_h 
        result = self.cleval_metric(
            pr_quads, gt_quads, pr_letters, gt_letters, gt_is_dcs
        )
        # print(temp)
        # import ipdb; ipdb.set_trace
        res_dict = {
            "det_r": result["det_r"],
            "det_p": result["det_p"],
            "det_h": result["det_h"],
            "e2e_r": result["e2e_r"],
            "e2e_p": result["e2e_p"],
            "e2e_h": result["e2e_h"],
        }
        return res_dict

    @staticmethod
    def print_result(res_dict):
        pass

    def _get_cleval_inp_base(self, pr_seq, gt_seq, kwargs):
        if "[POS_" in gt_seq:
            if kwargs["tokenizer_name"].startswith("char_"):
                pattern = self.pattern_dict["base"]["special_char"]
                if kwargs["task_name"] == "ocr_read_TextInstancePadding":
                    pr_seq = pr_seq.replace(kwargs["char_pad_token"], "")
                    gt_seq = gt_seq.replace(kwargs["char_pad_token"], "")
            else:
                pattern = self.pattern_dict["base"]["special"]
        else:
            pattern = self.pattern_dict["base"]["normal"]

        pr_objs = pattern.findall(pr_seq)
        gt_objs = pattern.findall(gt_seq)

        pr_letters = [pr_obj[4] for pr_obj in pr_objs]
        gt_letters = [gt_obj[4] for gt_obj in gt_objs]
        gt_is_dcs = [gt_letter == "[DONTCARE]" for gt_letter in gt_letters]

        pr_quads = []
        for pr_obj in pr_objs:
            tl_x, tl_y, br_x, br_y = pr_obj[:4]
            quad = [tl_x, tl_y, br_x, tl_y, br_x, br_y, tl_x, br_y]
            pr_quads.append(quad)
        pr_quads = np.array(pr_quads, dtype=np.float32).reshape((-1, 8))

        gt_quads = []
        for gt_obj in gt_objs:
            tl_x, tl_y, br_x, br_y = gt_obj[:4]
            quad = [tl_x, tl_y, br_x, tl_y, br_x, br_y, tl_x, br_y]
            gt_quads.append(quad)
        gt_quads = np.array(gt_quads, dtype=np.float32).reshape((-1, 8))

        return pr_quads, gt_quads, pr_letters, gt_letters, gt_is_dcs

    def _get_cleval_inp_2head(self, pr_seq, gt_seq, pr_loc_output, gt_loc_output):
        pattern = self.pattern_dict["2head"]

        pr_letters = pattern.findall(pr_seq)
        gt_letters = pattern.findall(gt_seq)

        # TODO: remove this
        scale = 1000
        pr_quads = np.array(pr_loc_output, dtype=np.float32)[: len(pr_letters)]
        pr_quads = pr_quads.reshape((-1, 8)) * scale
        gt_quads = np.array(gt_loc_output, dtype=np.float32)
        gt_quads = gt_quads.reshape((-1, 8)) * scale
        gt_is_dcs = [gt_letter == "[DONTCARE]" for gt_letter in gt_letters]
        return pr_quads, gt_quads, pr_letters, gt_letters, gt_is_dcs
