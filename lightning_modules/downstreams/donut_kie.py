"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import re
from typing import Any, Dict, List, Union

import zss
from nltk import edit_distance
from zss import Node

from lightning_modules.downstreams.base import BaseDownstream


class JSONParseEvaluator:
    # from https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/util.py#L138
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )


class DonutKIE(BaseDownstream):
    def __init__(self, dataset_item, dtd_key):
        super().__init__(dataset_item, dtd_key)
        self.end_prompt_token = "[END_PROMPT]"
        self.evaluator = JSONParseEvaluator()

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

    def parse_result_dict_transformer_decoder(self, pr_dict, gt_dict, kwargs):
        stat_score = {}
        pr_seq = pr_dict["pr_seq"]

        pr_end_prompt_idx = pr_seq.find(self.end_prompt_token)
        without_prompt_pr = pr_seq[
            pr_end_prompt_idx + len(self.end_prompt_token) :
        ].strip()

        pr_json = token2json_from_scratch(without_prompt_pr)
        gt_json = kwargs["gt_json"]

        # stat_score["nted"] = cal_nted(pr_json, gt_json) # donut previous evaluation metric
        stat_score["nted"] = self.evaluator.cal_acc(
            pr_json, gt_json
        )  # donut new evaluation metric arxiv v6

        return stat_score

    @staticmethod
    def print_result(res_dict):
        print(f"nted:\t{res_dict['nted']}", flush=True)


def token2json_from_scratch(pred, is_wyvern_style=False):
    pred = (
        pred.replace("[START_P]", "")
        .replace("[END]", "")
        .replace("[SPACE]", " ")
        .strip()
    )
    pred = re.sub(r"\s+", " ", pred)
    if is_wyvern_style:
        # output = token2json_wyvern(pred)
        raise NotImplementedError
    else:
        output = token2json(pred)
    return output


def token2json(tokens, is_inner_value=False):
    output = dict()

    while tokens:
        start_token = re.search("\[START_(.*?)\]", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(f"\[END_{key}\]", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE
            )
            if content is not None:
                content = content.group(1).strip()
                if "[START_" in content and "[END_" in content:
                    value = token2json(content, is_inner_value=True)
                    if value:
                        output[key] = value
                else:
                    output[key] = [__.strip() for __ in content.split("[DIV]")]
            processed_so_far = tokens.find(end_token) + len(end_token)
            tokens = tokens[processed_so_far:].strip()

            if tokens[:5] == "[DIV]":
                return [output] + token2json(tokens[5:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else output
