"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import torch

from lightning_modules.downstreams.donut_kie import DonutKIE
from lightning_modules.downstreams.ocr_read import OCRRead
from lightning_modules.downstreams.table_parsing import TableParsing
from lightning_modules.downstreams.text_read import TextRead
from utils.config_manager import ConfigManager
from utils.constants import Tasks


class DownstreamEvaluator:
    def __init__(self):
        cfg = ConfigManager().cfg
        self.modules = self.get_downstream_modules(cfg.dataset_items)

    @staticmethod
    def get_downstream_modules(dataset_items):
        task_to_class = {
            Tasks.OCR_READ: OCRRead,
            Tasks.OCR_READ_2HEAD: OCRRead,
            Tasks.OCR_READ_TEXTINSTANCEPADDING: OCRRead,
            Tasks.TEXT_READ: TextRead,
            Tasks.DONUT_KIE: DonutKIE,
            Tasks.TABLE_PARSING: TableParsing,
            Tasks.OTOR: OCRRead,
            Tasks.OTOR_ORACLE: OCRRead,
        }

        modules = {}
        for dataset_item in dataset_items:
            for task in dataset_item.tasks:
                task_class = task_to_class[task.name]
                dtd_key = (dataset_item.name, task.name, task.decoder)
                task_instance = task_class(dataset_item, dtd_key)
                modules[(dataset_item.name, task.name, task.decoder)] = task_instance

        return modules

    def downstream_score(
        self, dtd_key, decoder_type, pr_dict, gt_dict, kwargs, verbose=False
    ):
        module = self.modules[dtd_key]
        res_dict = module.parse_result_dict(decoder_type, pr_dict, gt_dict, kwargs)
        if verbose:
            module.print_result(res_dict)

        return res_dict

    def get_cleval_score(self, dtd_key, device):
        module = self.modules[dtd_key]
        assert hasattr(module, "cleval_metric")
        if module.cleval_metric.device != torch.device(device):
            module.cleval_metric = module.cleval_metric.to(device)

        cleval_result = module.cleval_metric.compute()
        module.cleval_metric.reset()
        return cleval_result
