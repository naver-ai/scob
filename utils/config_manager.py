"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license

config parser with omegaconf
"""

import enum
import os
import pickle
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from utils import misc
from utils.constants import AVAILABLE_TASKS, DecoderTypes, HeadTypes, Seperators, Tasks
from utils.misc import cpu_count, is_otor
from utils.singleton import Singleton


class FileType(enum.Enum):  # pylint: disable=missing-class-docstring
    YAML = 1
    PICKLE = 2


class ConfigManager(metaclass=Singleton):
    """Singleton ConfigManager for project

    Notes:
        Do not call ConfigManager.get_instance() inside the model class.
        When creating a model instance, all necessary arguments must be received as constructor arguments.

    SHOULD_NOT_USE_CONFIGS (List[str]): Keys that should not be included in the configuration.
        Because the corresponding configurations are applied at runtime,
        It should not be given through defualt.yaml, user config, or CLI.
    """

    SHOULD_NOT_USE_CONFIGS = [
        "workspace",
        "save_weight_dir",
        "tensorboard_dir",
        "dtd_dict",
    ]

    def __init__(
        self,
        conf_path=None,
        default_conf_path="./configs/default.yaml",
        conf_type=FileType.YAML,
        default_conf_type=FileType.YAML,
        use_cli=True,
    ):
        self.__config = None
        self.__tcp_store_server = None
        self.__tcp_store_cli = None
        self.default_conf_path = default_conf_path
        self.conf_path = conf_path
        self.__load(default_conf_type, conf_type, use_cli)

    @property
    def cfg(self):
        """Return path config"""
        return self.__config

    @property
    def model_cfg(self):
        """Return model config"""
        return self.__config.model

    def __load(self, default_conf_type, conf_type, use_cli):
        """load config from file"""
        # Load configuration parameters
        #   Step 1. Default config
        self.__config = self.__read_conf(self.default_conf_path, default_conf_type)

        #   Step 2. Config specified by __init__()
        if self.conf_path:
            cfg = self.__read_conf(self.conf_path, conf_type)
            self.__merge_config(cfg)

        if use_cli:
            #   Step 3. Config specified CLI's --config parameter
            cfg_cli = self.__get_config_from_cli()
            if "config" in cfg_cli:
                cfg = OmegaConf.load(cfg_cli.config)
                self.__merge_config(cfg)

            #   Step 4. Config specified CLI's --XYZ parameters
            self.__merge_config(cfg_cli)

        self.__init_tcp_store()

        # Validate config content and apply a few changes
        self.__check_config()
        self.__change_config()

        # Finalize config
        OmegaConf.set_readonly(self.__config, True)
        OmegaConf.set_struct(self.__config, True)
        if misc.is_rank_zero():
            print(OmegaConf.to_yaml(self.__config))

    @staticmethod
    def __read_conf(path, file_type=FileType.YAML):
        if file_type is FileType.PICKLE:
            with open(path, "rb") as fp:
                cfg = pickle.load(fp)
        elif file_type is FileType.YAML:
            cfg = OmegaConf.load(path)
        else:
            raise ValueError("[FileType Enum]: Invalid value!")
        return cfg

    def save(self, path, file_type):
        """Save config file to path"""
        with open(path, "wb") as fp:
            if file_type is FileType.PICKLE:
                pickle.dump(self.__config, fp)
                fp.flush()
            elif file_type is FileType.YAML:
                OmegaConf.save(config=self.__config, f=fp.name)
            else:
                raise ValueError("[FileType Enum]: Invalid value!")

    def __merge_config(self, cfg_for_overwrite):
        """omegaconf merge_config

        Notes:
            dataset merge 할 때 dictionary끼리 append 되기 때문에 이를 방지함.
        """
        self.__config = OmegaConf.merge(self.__config, cfg_for_overwrite)

        if "dataset_items" in cfg_for_overwrite:
            self.__config.dataset_items = cfg_for_overwrite.dataset_items
        if (
            "model" in cfg_for_overwrite
            and "decoders" in cfg_for_overwrite.model
            and "type" in list(cfg_for_overwrite.model.decoders.values())[0]
            and "kwargs" in list(cfg_for_overwrite.model.decoders.values())[0]
            and "loss_func" in list(cfg_for_overwrite.model.decoders.values())[0]
        ):
            self.__config.model.decoders = cfg_for_overwrite.model.decoders

    @staticmethod
    def __get_config_from_cli():
        """Get config from cli.
        This function can also cover arguments with '--'
        """
        cfg_cli = OmegaConf.from_cli()
        cli_keys = list(cfg_cli.keys())
        for cli_key in cli_keys:
            if "--" in cli_key:
                cfg_cli[cli_key.replace("--", "")] = cfg_cli[cli_key]
                del cfg_cli[cli_key]

        return cfg_cli

    def __init_tcp_store(self):
        ip = self.__config.tcp_store_ip
        port = self.__config.tcp_store_port
        time_delta = timedelta(seconds=300)
        if misc.is_rank_zero():
            self.__tcp_store_server = dist.TCPStore(ip, port, -1, True, time_delta)
        else:
            self.__tcp_store_cli = dist.TCPStore(ip, port, -1, False, time_delta)

    def __check_config(self):
        """Check config"""
        for key in self.SHOULD_NOT_USE_CONFIGS:
            if key in self.__config.keys():
                raise ValueError(
                    f"Do not use {key} as a configuration. \n"
                    "This configuration should be added"
                    " automatically in runtime."
                )

        for mode in ["train", "val", "test"]:
            num_devices = torch.cuda.device_count() * self.__config.train.num_nodes
            if self.__config[mode].batch_size % num_devices != 0:
                raise ValueError(
                    f"{mode} batch-size should be a multiple"
                    " of the number of gpu devices"
                )

        # check decoder_names
        decoder_name_set_from_dataset_items = set()
        for dataset_item in self.__config.dataset_items:
            for task in dataset_item.tasks:
                decoder_name_set_from_dataset_items.add(task.decoder)

        decoder_name_set_from_decoders = set()
        for decoder_name, decoder_cfg in self.__config.model.decoders.items():
            decoder_name_set_from_decoders.add(decoder_name)
            assert decoder_name.startswith(decoder_cfg.type)

        if decoder_name_set_from_dataset_items != decoder_name_set_from_decoders:
            raise ValueError(
                "Please match decoder-names.\n"
                f"dec-names from dataset_items: {decoder_name_set_from_dataset_items}\n"
                f"dec-names from decoders: {decoder_name_set_from_decoders}"
            )

        # Check available tasks
        for dataset_item in self.__config.dataset_items:
            for task in dataset_item.tasks:
                decoder_cfg = self.__config.model.decoders[task.decoder]
                available_tasks = AVAILABLE_TASKS[decoder_cfg.type]
                if task.name not in available_tasks:
                    raise ValueError(
                        f"Unavailable task {task.name} for decoder {task.decoder}"
                    )

                if decoder_cfg.type == DecoderTypes.TRANSFORMER:
                    assert not (
                        (decoder_cfg.head_type == HeadTypes.TWO_HEAD)
                        ^ task.name.endswith(HeadTypes.TWO_HEAD)
                    ), "Two head model should solve two head task."

                # Check image_normalize type in PATCH_CLS with grayscale label
                if (
                    hasattr(decoder_cfg, "task")
                    and decoder_cfg.task == Tasks.PATCH_CLS
                    and decoder_cfg.kwargs.classification_type == "grayscale"
                ):
                    transforms_dict = task.transforms_dict
                    if isinstance(transforms_dict, str):
                        if transforms_dict not in self.__config.custom_transforms_dict:
                            raise ValueError(
                                f"{transforms_dict} is not in cfg.custom_transforms_dict"
                            )
                        transforms_dict = self.__config.custom_transforms_dict[
                            transforms_dict
                        ]

                    assert transforms_dict["image_normalize"] == "imagenet_default"

    def __change_config(self):
        """Change config like path"""
        cfg = self.__config  # get reference of __config

        # -------------------------------------------
        # for convinience (only used for evaluate.py)
        if cfg.eval.dataset_name is not None and cfg.eval.task_name is not None:
            cfg.dataset_items = [get_eval_dataset_item(cfg.eval)]
        # -------------------------------------------

        workspace = cfg.workspace_name
        if workspace is None:
            workspace = os.getcwd()

        cfg.workspace = workspace
        self.__change_data_paths()
        cfg.model.resume_model_path = self.__change_weight_path(
            workspace, cfg.model.resume_model_path
        )
        self.__change_log_dirs()
        num_devices = torch.cuda.device_count() * cfg.train.num_nodes
        for mode in ["train", "val", "test"]:
            # set per-gpu num_workers
            if cfg.debug:
                cfg[mode].num_workers = 0
            else:
                num_workers = cfg[mode].num_workers
                if num_workers == -1:
                    num_workers = cpu_count() * cfg.train.num_nodes
                cfg[mode].num_workers = max(num_workers // num_devices, 1)

            # set per-gpu batch size
            new_batch_size = cfg[mode].batch_size // num_devices
            cfg[mode].batch_size = new_batch_size

            if mode == "train" and is_otor(cfg.dataset_items[0].tasks[0].name):
                if cfg[mode].batch_size % 2 > 0:
                    assert (
                        cfg[mode].batch_size % 2 == 0
                    ), "when use otor, batch size should be even number."
                cfg[mode].batch_size = cfg[mode].batch_size // 2

        if cfg.reproduce.seed == -1:
            # To avoid each rank have different random_seed, we use TCPStore.
            if misc.is_rank_zero():
                random_seed = int(time.time()) % 10000
                self.__tcp_store_server.set("random_seed", str(random_seed))
            else:
                random_seed = int(self.__tcp_store_cli.get("random_seed").decode())
            cfg.reproduce.seed = random_seed

        # Make DTD (Dataset-name, Task-name, Decoder-name) configs
        dtd_dict = {}
        for dataset_item in cfg.dataset_items:
            dataset_name = dataset_item.name
            for task in dataset_item.tasks:
                dtd_key_str = Seperators.DTD.join(
                    [dataset_name, task.name, task.decoder]
                )

                transforms_dict = task.transforms_dict
                if isinstance(transforms_dict, str):
                    if transforms_dict not in cfg.custom_transforms_dict:
                        raise ValueError(
                            f"{transforms_dict} is not in cfg.custom_transforms_dict"
                        )
                    transforms_dict = cfg.custom_transforms_dict[transforms_dict]

                dtd_dict[dtd_key_str] = {
                    "batch_ratio": task.batch_ratio,
                    "loss_weight": task.loss_weight,
                    "dataset_root_path": dataset_item.dataset_root_path,
                    "dataset_label": dataset_item.dataset_label,
                    "use_anno": task.use_anno,
                    "transforms_dict": transforms_dict,
                }
        cfg.dtd_dict = dtd_dict

        # Set window_size of swin_transformer
        if cfg.model.encoder.encoder_model_name.startswith("swin_transformer"):
            transformer_cfg = cfg.model.encoder[cfg.model.encoder.encoder_model_name]
            if not hasattr(transformer_cfg, "window_size"):
                transformer_cfg.window_size = transformer_cfg.img_size // (
                    transformer_cfg.patch_size * 8
                )

    def __change_data_paths(self):
        cfg = self.__config  # get reference of __config

        for dataset_item in cfg.dataset_items:
            dataset_item.dataset_root_path = os.path.join(
                cfg.data_root_path, dataset_item.dataset_root_path
            )

    @staticmethod
    def __change_weight_path(workspace, weight_path):
        """
        Look for weight_path in the workspace.
        Function to correspond to init_weight and resume_weight.

        Args:
            workspace (str): workspace path
            weight_path (str): weight path

        Returns:
            weight_path (str): updated weight path
        """

        if weight_path is not None:
            resume_path = os.path.join(workspace, weight_path)
            if os.path.exists(resume_path):
                weight_path = resume_path
            elif not os.path.exists(weight_path):
                raise ValueError(
                    "(Resume or init) weight_file is not found."
                    f"\n weight_path: {weight_path}"
                    f"\n resume_path: {resume_path}"
                )
        return weight_path

    def __change_log_dirs(self):
        cfg = self.__config  # get reference of __config

        # To avoid each rank have different date_dir, we use TCPStore.
        if misc.is_rank_zero():
            date_dir = time.strftime("%Y_%m_%d_%H%M", time.localtime(time.time()))
            self.__tcp_store_server.set("date_dir", date_dir)
        else:
            date_dir = self.__tcp_store_cli.get("date_dir").decode()

        cfg.save_weight_dir = os.path.join(
            cfg.workspace, "checkpoints", cfg.exp_name, date_dir
        )
        cfg.tensorboard_dir = os.path.join(
            cfg.workspace, cfg.logging.tensorboard.log_root_dir, cfg.exp_name, date_dir
        )


def get_eval_dataset_item(eval_cfg):
    dataset_item = OmegaConf.create()
    dataset_item.name = eval_cfg.dataset_name
    dataset_item.dataset_root_path = get_eval_dataset_path(eval_cfg.dataset_name)
    dataset_item.dataset_label = eval_cfg.dataset_label
    dataset_item.tasks = [
        {
            "name": eval_cfg.task_name,
            "batch_ratio": 1.0,
            "decoder": eval_cfg.decoder_name,
            "loss_weight": 1.0,
            "use_anno": eval_cfg.use_anno,
            "transforms_dict": eval_cfg.transforms_dict,
        }
    ]
    return dataset_item


def get_eval_dataset_path(eval_dataset_name):
    dataset_path = f"./datasets/{eval_dataset_name}"

    if not os.path.exists(dataset_path):
        raise ValueError(f"{dataset_path} not exists")
    return dataset_path


def cfg_to_hparams(cfg, hparam_dict, parent_str=""):
    """
    Create hyperparameter dictionary using nested dictionary
    Function for logging to tensorboard
    Args:
        cfg (DictConfig): configuration
        hparam_dict (dict): Dictionary representing hyperparameters as non-nested
            You can easily use the function by giving an empty dictionary {} as the initial value.
        parent_str (str): Complete the config key by adding the parent string as a prefix.
            For example, to create a child called train__gt_maker, add the prefix "train__".
    Returns:
        hparam_dict (dict): Dictionary representing hyperparameters as non-nested
    """

    for key, val in cfg.items():
        if isinstance(val, DictConfig):
            hparam_dict = cfg_to_hparams(val, hparam_dict, parent_str + key + "__")
        else:
            hparam_dict[parent_str + key] = str(val)
    return hparam_dict
