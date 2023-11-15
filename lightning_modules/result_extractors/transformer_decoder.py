"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import numpy as np

from lightning_modules.result_extractors.base import BaseDecResultExtractor


class TransformerDecResultExtractor(BaseDecResultExtractor):
    def __init__(self):
        super().__init__()
        self.decoder_token_losses = []
        self.decoder_loc_losses = []
        self.decoder_losses = []
        self.norm_eds = []

    def extract(self, step_out):
        self.extract_basedec(step_out)
        self.extract_transformerdec(step_out)

    def extract_transformerdec(self, step_out):
        self.decoder_token_losses.append(step_out["decoder_token_loss"])
        self.decoder_loc_losses.append(step_out["decoder_loc_loss"])
        self.decoder_losses.append(step_out["decoder_loss"])
        self.norm_eds.extend(step_out["norm_eds"])

    def get_log_info(self, prefix):
        log_info = self.get_log_info_basedec(prefix)
        log_info.update(self.get_log_info_transformerdec(prefix))
        return log_info

    def get_log_info_transformerdec(self, prefix):
        log_info = {
            f"{prefix}decoder_token_loss": sum(self.decoder_token_losses)
            / len(self.decoder_token_losses),
            f"{prefix}decoder_loc_loss": sum(self.decoder_loc_losses)
            / len(self.decoder_loc_losses),
            f"{prefix}decoder_loss": sum(self.decoder_losses)
            / len(self.decoder_losses),
            f"{prefix}norm_ed": sum(self.norm_eds) / len(self.norm_eds),
        }
        return log_info

    def initialize(self):
        self.initialize_basedec()
        self.initialize_transformerdec()

    def initialize_transformerdec(self):
        self.decoder_losses.clear()
        self.norm_eds.clear()


class TransformerDecOCRReadResultExtractor(TransformerDecResultExtractor):
    def __init__(self, dataset_item):
        super().__init__()

    def extract(self, step_out):
        self.extract_basedec(step_out)
        self.extract_transformerdec(step_out)
        self.extract_task(step_out)

    def extract_task(self, step_out):
        pass

    def get_log_info(self, prefix):
        log_info = self.get_log_info_basedec(prefix)
        log_info.update(self.get_log_info_transformerdec(prefix))
        log_info.update(self.get_log_info_task(prefix))
        return log_info

    def get_log_info_task(self, prefix):
        return {}

    def initialize(self):
        self.initialize_basedec()
        self.initialize_transformerdec()
        self.initialize_task()

    def initialize_task(self):
        pass


class TransformerDecTextReadResultExtractor(TransformerDecResultExtractor):
    def __init__(self, dataset_item):
        super().__init__()

    def extract(self, step_out):
        self.extract_basedec(step_out)
        self.extract_transformerdec(step_out)
        self.extract_task(step_out)

    def extract_task(self, step_out):
        pass

    def get_log_info(self, prefix):
        log_info = self.get_log_info_basedec(prefix)
        log_info.update(self.get_log_info_transformerdec(prefix))
        log_info.update(self.get_log_info_task(prefix))
        return log_info

    def get_log_info_task(self, prefix):
        return {}

    def initialize(self):
        self.initialize_basedec()
        self.initialize_transformerdec()
        self.initialize_task()

    def initialize_task(self):
        pass


class TransformerDecDonutKIEResultExtractor(TransformerDecResultExtractor):
    def __init__(self, dataset_item):
        super().__init__()
        self.nteds = []

    def extract(self, step_out):
        self.extract_basedec(step_out)
        self.extract_transformerdec(step_out)
        self.extract_task(step_out)

    def extract_task(self, step_out):
        for score in step_out["scores"]:
            self.nteds.append(score["nted"])

    def get_log_info(self, prefix):
        log_info = self.get_log_info_basedec(prefix)
        log_info.update(self.get_log_info_transformerdec(prefix))
        log_info.update(self.get_log_info_task(prefix))
        return log_info

    def get_log_info_task(self, prefix):
        log_info = {f"{prefix}nted": np.mean(self.nteds)}
        return log_info

    def initialize(self):
        self.initialize_basedec()
        self.initialize_transformerdec()
        self.initialize_task()

    def initialize_task(self):
        self.nteds.clear()


class TransformerDecTableParsingResultExtractor(TransformerDecResultExtractor):
    def __init__(self, dataset_item):
        super().__init__()
        self.teds = []

    def extract(self, step_out):
        self.extract_basedec(step_out)
        self.extract_transformerdec(step_out)
        self.extract_task(step_out)

    def extract_task(self, step_out):
        for score in step_out["scores"]:
            self.teds.append(score["ted"])

    def get_log_info(self, prefix):
        log_info = self.get_log_info_basedec(prefix)
        log_info.update(self.get_log_info_transformerdec(prefix))
        log_info.update(self.get_log_info_task(prefix))
        return log_info

    def get_log_info_task(self, prefix):
        teds = np.mean(self.teds)
        log_info = {prefix + "teds": teds}
        return log_info

    def initialize(self):
        self.initialize_basedec()
        self.initialize_transformerdec()
        self.initialize_task()

    def initialize_task(self):
        self.teds.clear()
