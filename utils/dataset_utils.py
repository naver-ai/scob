"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import os
from collections import OrderedDict

import zss
from editdistance import eval as strdist
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from zss import Node

from utils.misc import get_file


def get_image_normalize_mean_and_std(image_normalize):
    if image_normalize is None:
        mean_and_std = None
    elif image_normalize == "imagenet_default":
        mean_and_std = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    elif image_normalize == "imagenet_inception":
        # In BEiT, "--imagenet_default_mean_and_std: enable this for ImageNet-1k pre-training,
        # i.e., (0.485, 0.456, 0.406) for mean and (0.229, 0.224, 0.225) for std.
        # We use (0.5, 0.5, 0.5) for mean and (0.5, 0.5, 0.5) for std by default
        # on other pre-training data."
        mean_and_std = (IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    else:
        raise ValueError(f"Unknown image_normalize={image_normalize}")

    return mean_and_std


def get_class_names(dataset_root_path, dataset_label):
    class_names_file = get_file(dataset_root_path, "class_names", dataset_label, ".txt")
    if os.path.exists(class_names_file):
        class_names = (
            open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
        )
    else:
        class_names = None

    return class_names


def normalize_dict(data):
    if type(data) == dict:
        data = OrderedDict(data)

    if type(data) == OrderedDict:
        ks = sorted(list(data.keys()))
        new_data = OrderedDict()
        for k in ks:
            v_sorted = normalize_dict(data[k])
            if v_sorted:
                new_data[k] = v_sorted

    elif type(data) == list:
        is_leaf = 0
        not_leaf = 0
        list_in_list = 0
        for item in data:
            if type(item) in [dict, OrderedDict]:
                not_leaf += 1
            elif type(item) == list:
                list_in_list += 1
            else:
                is_leaf += 1

        if (is_leaf > 0 and not_leaf > 0) or list_in_list > 0:
            print(
                f"When something other than a list of strings is mixed. Discard only the other mixed contents. {data}"
            )
            return []

        new_data = []
        if not_leaf >= is_leaf:
            _data = []
            for data1 in data:
                try:
                    if type(data1) == dict:
                        _data.append(OrderedDict(data1))
                    else:
                        assert type(data1) == OrderedDict
                        _data.append(data1)
                except:
                    print(
                        f"Incosistant data!. the type of {data1} is expected to be dict."
                    )
                    print(f"Force fully change to empty dict")
            data = _data

            for data1 in data:
                assert type(data1) == OrderedDict
                normed_data1 = normalize_dict(data1)
                if normed_data1:
                    new_data.append(normed_data1)

            new_data = sorted(
                new_data, key=lambda x: f"{list(x.keys())} {list(x.values())}"
            )
        else:
            # list of string
            str_data = []
            for item in data:
                if type(item) in [str, int, float]:
                    str_data.append(item)
                else:
                    print(
                        f"When something other than a list of strings is mixed. Discard only the other mixed contents. {item}"
                    )
            new_data += sorted(str_data)
    elif type(data) == str:
        new_data = data
    elif type(data) in [int, float]:
        new_data = str(data)
    else:
        raise NotImplementedError

    return new_data


def construct_tree_from_dict(data, node_name=None):
    if node_name is None:
        node_name = "root_node"

    node = Node(node_name)

    if type(data) == OrderedDict:
        for k, v in data.items():
            kid_node = construct_tree_from_dict(v, k)
            node.addkid(kid_node)
    elif type(data) == list:
        if type(data[0]) == OrderedDict:
            for data1 in data:
                kid_node = construct_tree_from_dict(data1, "list")
                node.addkid(kid_node)
        else:
            for data1 in data:
                assert type(data1) in [str, int, float], print(
                    f"{type(data1), data, data1}"
                )  # no dict assumed.
                node.addkid(Node(f"leaf_node_{data1}"))
    else:
        raise Exception(node_name, data)

    return node


def get_tree_size(node):
    kids = node.children
    if len(kids) == 0:
        if node.label == "root_node":
            return 0
        else:
            return len(node.label)
    else:
        n = 0
        for kid in kids:
            n += get_tree_size(kid)
        return n + 1


def update_cost(label1, label2):
    """A function update_cost(a, b) == cost to change a into b >= 0"""
    label1_leaf = "leaf_node" in label1
    label2_leaf = "leaf_node" in label2
    if label1_leaf and label2_leaf:
        return strdist(
            label1.replace("leaf_node_", ""), label2.replace("leaf_node_", "")
        )
    elif not label1_leaf and label2_leaf:
        return 1 + len(label2.replace("leaf_node_", ""))
    elif label1_leaf and not label2_leaf:
        return 1 + len(label1.replace("leaf_node_", ""))
    else:
        return int(label1 != label2)


def insert_and_remove_cost(node):
    label = node.label
    if "leaf_node" in label:
        return len(label.replace("leaf_node_", ""))
    else:
        return 1


def cal_mea(pred, answer, subinfo=None):
    score = 0
    num_of_valid_field = 0
    for k, v in answer.items():
        if type(v) == list:
            assert len(v) == 1, print("debug: ", answer, v)
            v = v[0]
        v = v.strip()
        infered = pred.get(k, [""])
        assert len(infered) == 1
        if type(infered[0]) != str:
            assert type(infered[0]) == dict
            print(
                f"the value of {k} will be drop as the "
                f"structure wrongly inferred!: {infered[0]}"
            )
            print(f"full sample: {pred}")
            print(f"full answer: {answer}")
            if subinfo is not None:
                print(f"subinfo: {subinfo}")
            infered[0] = ""
        infered = infered[0].strip()
        if infered and v and infered == v:
            score += 1
            num_of_valid_field += 1
        else:
            score += 0
            num_of_valid_field += 1
    return score / num_of_valid_field if num_of_valid_field else 0.0


def cal_nted(pred, answer, return_ted_also=False):
    pred = construct_tree_from_dict(normalize_dict(pred))
    answer = construct_tree_from_dict(normalize_dict(answer))
    ted = zss.distance(
        pred,
        answer,
        get_children=zss.Node.get_children,
        insert_cost=insert_and_remove_cost,
        remove_cost=insert_and_remove_cost,
        update_cost=update_cost,
        return_operations=False,
    )
    tree_size = zss.distance(
        construct_tree_from_dict(normalize_dict({})),
        answer,
        get_children=zss.Node.get_children,
        insert_cost=insert_and_remove_cost,
        remove_cost=insert_and_remove_cost,
        update_cost=update_cost,
        return_operations=False,
    )
    nted = ted / tree_size  # get_tree_size(answer)
    if return_ted_also:
        return nted, ted
    else:
        return nted


# --------------------------------------------------------------------------------------
