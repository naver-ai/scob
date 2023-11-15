"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import os
import subprocess


def get_node_rank():
    """NODE_RANK is set when running multi-node trainig with PyTorch Lightning."""
    return int(os.environ.get("NODE_RANK", 0))


def get_local_rank():
    """Pytorch lightning save local rank to environment variable "LOCAL_RANK"."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank


def is_rank_zero():
    if get_node_rank() == 0 and get_local_rank() == 0:
        return True
    else:
        return False


def cpu_count():
    """Get number of cpu
    os.cpu_count() has a problem with docker container.
    For example, we have 72 cpus. os.cpu_count() always return 72
    even if we allocate only 4 cpus for container.
    """
    return int(subprocess.check_output("nproc").decode().strip())


def get_file(dataset_path, prefix, postfix, ext):
    file_name = f"{prefix}{ext}" if postfix is None else f"{prefix}_{postfix}{ext}"
    file_path = os.path.join(dataset_path, file_name)
    return file_path


def is_otor(task_name, or_oracle=False, oracle=False):
    if or_oracle:
        if task_name in ["otor", "otor_oracle"]:
            return True
    elif oracle:
        if task_name == "otor_oracle":
            return True
    else:
        if task_name == "otor":
            return True
    return False
