"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import numpy as np
import torch


class BaseCollate:
    def __init__(self):
        self.numpy_keys = {
            "labels",
        }
        self.tensor_keys = {"imgs", "second_imgs", "multiview_imgs"}
        self.do_not_tensorize_keys = {
            "metas",
            "dataset_names",
            "task_names",
            "decoder_names",
            "gt_jsons",
            "all_answers",
        }

    @staticmethod
    def stack_batch(batch, instances, target_instance_indices=None):
        if target_instance_indices is not None:
            instances = [instances[idx] for idx in target_instance_indices]

        for instance in instances:
            for key in batch.keys():
                if key in instance:
                    batch[key].append(instance[key])

    def tensorize(self, batch):
        remove_keys = []
        for key, values in batch.items():
            if len(values) > 0 and key in self.numpy_keys:
                batch[key] = torch.from_numpy(np.stack(values))
            elif len(values) > 0 and key in self.tensor_keys:
                batch[key] = torch.stack(values)
            elif len(values) > 0 and key in self.do_not_tensorize_keys:
                batch[key] = values
            else:
                assert len(values) == 0, f"Please add key {key} to key_sets attributes!"
                remove_keys.append(key)
        for key in remove_keys:
            del batch[key]
