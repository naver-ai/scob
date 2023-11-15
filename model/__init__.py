"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from model.encoder_decoder import EncoderAndDecoders, EncoderAndDecodersForPreTraining
from model.initialization import weights_init
from utils.config_manager import ConfigManager
from utils.constants import Phase


def get_model():
    cfg = ConfigManager().cfg

    if cfg.phase == Phase.FINETUNING:
        model = EncoderAndDecoders(cfg)
    elif cfg.phase == Phase.PRETRAINING:
        model = EncoderAndDecodersForPreTraining(cfg)
    else:
        raise ValueError(f"Unknown cfg.task={cfg.task}")

    # Weight initialize
    weights_init_fn = weights_init(cfg.model.weight_init)
    model.init_weights(weights_init_fn)

    return model


if __name__ == "__main__":
    # PYTHONPATH=$PWD python model/__init__.py --config=configs/private/dummy.yaml
    __model = get_model()
