"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license

Modified gpu_monitor callback from PL v1.4.8
"""

from overrides import overrides
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor


class ModifiedGPUStatsMonitor(GPUStatsMonitor):
    """Modified gpu_monitor for naming"""

    @staticmethod
    @overrides
    def _parse_gpu_stats(device_ids, stats, keys):
        """Parse the gpu stats into a loggable dict
        "gpu_util/" prefix added.
        """
        logs = {}
        for i, gpu_id in enumerate(device_ids):
            for j, (x, unit) in enumerate(keys):
                logs[f"gpu_util/gpu_id: {gpu_id}/{x} ({unit})"] = stats[i][j]
        return logs
