"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import os

from pytorch_lightning.callbacks import Callback

from utils.constants import DecoderTypes


class SaveTokenizer(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, trainer, pl_module):
        for decoder_name, decoder in pl_module.net.decoders.decoder_dict.items():
            if decoder_name.startswith(DecoderTypes.TRANSFORMER):
                cfg = pl_module.cfg
                tokenizer = decoder.tokenizer
                tokenizer_name = cfg.model.decoders[decoder_name].kwargs.tokenizer_name
                tokenizer_dir = os.path.join(
                    cfg.save_weight_dir, "tokenizers", decoder_name, tokenizer_name
                )
                tokenizer.save_pretrained(tokenizer_dir)
