"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from collections import OrderedDict

import torch.nn as nn

from model.decoders.transformer_decoder import TransformerDecoder


class Decoders(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder_dict = get_decoder_dict(cfg)

    def forward(self, batch, connector_out_dict, enc_kwargs):
        decoders_output_dict = {}
        decoders_losses_dict = {}
        decoders_kwargs = {}
        total_loss = 0

        for decoder_name, decoder in self.decoder_dict.items():
            if decoder_name in connector_out_dict:
                connector_out = connector_out_dict[decoder_name]

                output_dict, losses_dict, kwargs = decoder(
                    batch[decoder_name], connector_out, enc_kwargs
                )
                decoders_output_dict[decoder_name] = output_dict
                decoders_losses_dict[decoder_name] = losses_dict
                decoders_kwargs[decoder_name] = kwargs

                total_loss += losses_dict["decoder_loss"] * len(batch[decoder_name])
        total_loss /= len(batch["imgs"])

        return total_loss, decoders_output_dict, decoders_losses_dict, decoders_kwargs

    def init_weights(self, weights_init_fn):
        for decoder in self.decoder_dict.values():
            decoder.init_weights(weights_init_fn)


class DecodersForPreTraining(Decoders):
    """Decoders for pretraining"""


def get_decoder_dict(cfg):
    decoder_dict = OrderedDict()
    for decoder_name, decoder_cfg in cfg.model.decoders.items():
        if decoder_cfg.type == "transformer_decoder":
            decoder = TransformerDecoder(cfg, decoder_cfg, decoder_name)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_cfg.type}")

        decoder_dict[decoder_name] = decoder

    return nn.ModuleDict(decoder_dict)
