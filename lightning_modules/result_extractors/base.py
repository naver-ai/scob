"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

class BaseDecResultExtractor:
    def __init__(self):
        super().__init__()
        self.losses = []

    def extract(self, step_out):
        self.extract_basedec(step_out)

    def extract_basedec(self, step_out):
        self.losses.append(step_out["loss"])

    def get_log_info(self, prefix):
        return self.get_log_info_basedec(prefix)

    def get_log_info_basedec(self, prefix):
        log_info = {prefix + "loss": sum(self.losses) / len(self.losses)}
        return log_info

    def initialize(self):
        self.initialize_basedec()

    def initialize_basedec(self):
        self.losses.clear()
