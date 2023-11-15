"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Union

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from tqdm import tqdm

from lightning_modules.downstreams.base import BaseDownstream


class TableParsing(BaseDownstream):
    def __init__(self, dataset_item, dtd_key):
        super().__init__(dataset_item, dtd_key)

        pattern_dict = self._get_pattern_dict()
        self.pattern = pattern_dict["normal"]
        self.teds = TEDS()

    def _get_pattern_dict(self) -> Dict:
        pattern_dict = {
            "normal": re.compile("\[START_TP\]\s*(.*?)\s*\[END_TP\]"),
            "special": re.compile("\[END_PROMPT\]\s*(.*?)\s*\[END_TP\]"),
        }
        return pattern_dict

    def _get_class_names(self) -> Union[List, None]:
        return None

    def parse_result_dict(self, decoder_type, pr_dict, gt_dict, kwargs) -> Dict:
        if decoder_type == "transformer_decoder":
            return self.parse_result_dict_transformer_decoder(pr_dict, gt_dict, kwargs)
        else:
            raise ValueError("Unsupported decoder type")

    def parse_result_dict_transformer_decoder(self, pr_dict, gt_dict, kwargs) -> Dict:
        pr_seq = pr_dict["pr_seq"]
        if "[END_TP]" not in pr_seq:
            pr_seq += "[END_TP]"
        gt_seq = gt_dict["gt_seq"]

        pr_seq = self.pattern.findall(pr_seq)
        gt_seq = self.pattern.findall(gt_seq)

        res_dict = {"ted": self.teds.evaluate(pr_seq, gt_seq)} # train
        # res_dict = {
        #     "ted": self.teds.evaluate(pr_seq[0].replace("[UNK]", ""), gt_seq[0])
        # }  # test

        return res_dict

    @staticmethod
    def print_result(res_dict):
        ted = res_dict["ted"]
        print(f"ted:\t{ted:.4f}", flush=True)


# https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py
class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS(object):
    """Tree Edit Distance basead Similarity"""

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
            n_jobs >= 1
        ), "n_jobs must be an integer greather than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        global __tokens__
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            ### exception processed
            if node.attrib.get("colspan", "1") == "":
                num_colspan = 1
            else:
                try:
                    num_colspan = int(float(node.attrib.get("colspan", "1")))
                except:
                    num_colspan = 1

            if node.attrib.get("rowspan", "1") == "":
                num_rowspan = 1
            else:
                try:
                    num_rowspan = int(float(node.attrib.get("rowspan", "1")))
                except:
                    num_rowspan = 1

            new_node = TableTree(
                node.tag,
                num_colspan,
                num_rowspan,
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        if (not pred) or (not true):
            return 0.0

        pred = pred.replace("\\u2028", "\u2028")  # special token recovery
        true = true.replace("\\u2028", "\u2028")  # special token recovery

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        """Computes TEDS score between the prediction and the ground truth of
        a batch of samples
        @params pred_json: {'FILENAME': 'HTML CODE', ...}
        @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
        @output: {'FILENAME': 'TEDS SCORE', ...}
        """
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [
                self.evaluate(pred_json.get(filename, ""), true_json[filename]["html"])
                for filename in tqdm(samples)
            ]
        else:
            inputs = [
                {
                    "pred": pred_json.get(filename, ""),
                    "true": true_json[filename]["html"],
                }
                for filename in samples
            ]
            scores = parallel_process(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1
            )
        scores = dict(zip(samples, scores))
        return scores


# https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/parallel.py
def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
    A parallel version of the map function with a progress bar.
    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of array
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
            keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
            Useful for catching bugs
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [
            function(**a) if use_kwargs else function(a) for a in array[:front_num]
        ]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [
            function(**a) if use_kwargs else function(a)
            for a in tqdm(array[front_num:])
        ]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            "total": len(futures),
            "unit": "it",
            "unit_scale": True,
            "leave": True,
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
