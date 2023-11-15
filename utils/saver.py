"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license

This is a util for storing model weight or log.
"""

import os
import warnings
from collections import OrderedDict

import pytorch_lightning as pl
import torch

from utils.config_manager import ConfigManager, FileType


def change_permissions_recursive(path, mode):
    """Change the permissions of the folder or file."""
    if os.path.isfile(path):
        os.chmod(path, mode)

    for root, dirs, files in os.walk(path, topdown=False):
        for directory in dirs:
            os.chmod(os.path.join(root, directory), mode)
        for file in files:
            os.chmod(os.path.join(root, file), mode)


def prepare_checkpoint_dir():
    """Creates and returns a weight storage folder."""
    config_manager = ConfigManager()
    cfg = config_manager.cfg
    weight_dir = cfg.save_weight_dir

    if os.path.exists(weight_dir) and len(os.listdir(weight_dir)) > 0:
        raise ValueError(f"{weight_dir} is not empty. Something is wrong.")
    else:
        _make_dir(weight_dir)
        print(f"Save weights at: {weight_dir}", flush=True)

        setting_yaml = os.path.join(weight_dir, "setting.yaml")
        config_manager.save(setting_yaml, FileType.YAML)
        setting_pickle = os.path.join(weight_dir, "setting.pickle")
        config_manager.save(setting_pickle, FileType.PICKLE)


def prepare_tensorboard_log_dir():
    """Creates and returns a tensorboard log storage folder."""
    cfg = ConfigManager().cfg
    _make_dir(cfg.tensorboard_dir)
    print(f"Logging for tensorboard at: {cfg.tensorboard_dir}", flush=True)


def _make_dir(path):
    """If path does not exist, create a new directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_weights(net, weight_file, device, verbose=False):
    """
    Only the weights are loaded back into the net. (Almost for testing)
    Provides module keyword deletion function for compatibility

    Args:
        net: model
        weight_file: weight_file path
        device: device
        verbose: verbosity
    """

    if weight_file is not None:
        ckpt = torch.load(weight_file, map_location=device)
        if pl.__version__ != ckpt["pytorch-lightning_version"]:
            msg = (
                f"Installed PL version [{pl.__version__}] is different from"
                f" weight PL version [{ckpt['pytorch-lightning_version']}]."
            )
            warnings.warn(msg)

        state_dict = _remove_keyword(ckpt["state_dict"], keyword="net.")
        net.load_state_dict(state_dict)
        if verbose:
            print(f"Weights are loaded from ({weight_file})", flush=True)

        del ckpt
        del state_dict
        torch.cuda.empty_cache()


def _remove_keyword(state_dict, keyword="module."):
    """Make do not start keys with 'module.'. For compatibility."""
    if all([k.startswith(keyword) for k in state_dict.keys()]):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[len(keyword) :]
            new_state_dict[name] = v
        del state_dict
        state_dict = new_state_dict
    return state_dict
