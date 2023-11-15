"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from model.decoders import Decoders, DecodersForPreTraining
from model.encoders import Encoder, EncoderForPreTraining
from utils.constants import Tasks


class EncoderAndDecoders(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.train_cfg = cfg.train
        self.model_cfg = cfg.model
        self.enc_model_name = cfg.model.encoder.encoder_model_name
        self.encoder = Encoder(cfg)
        self.decoders = Decoders(cfg)
        self.connector_dict = self.get_connector_dict()

    def get_connector_dict(self):
        enc_model_name = self.model_cfg.encoder.encoder_model_name
        connector_dict = OrderedDict()
        for decoder_name, decoder in self.decoders.decoder_dict.items():
            connector_size = decoder.connector_size
            if enc_model_name in ["vision_transformer", "vision_transformer_hybrid"]:
                connector = nn.Linear(self.encoder.model.embed_dim, connector_size)
            elif enc_model_name == "swin_transformer":
                connector = nn.Linear(self.encoder.model.embed_dim * 8, connector_size)
            elif enc_model_name == "swin_transformer_fpn":
                connector = nn.Linear(
                    self.model_cfg.encoder[enc_model_name].out_channels, connector_size
                )
            else:
                raise ValueError(f"Unknown encoder_model_name={enc_model_name}")

            connector_dict[decoder_name] = connector
        return nn.ModuleDict(connector_dict)

    def forward(self, batch):
        if batch["task_names"][0] in ["otor", "otor_oracle"]:
            batch["imgs"] = torch.cat((batch["imgs"], batch["second_imgs"]), 0)
        # encoder
        enc_out_dict, enc_kwargs = self.encoder(batch)

        # connector
        connector_out_dict = {}
        if len(self.connector_dict) == 1:
            decoder_name = list(self.connector_dict.keys())[0]
            task_name = batch["task_names"][0]
            connector_inp = enc_out_dict["encoder_outputs"]
            connector_inp = self.pool_connector_inp(
                connector_inp, decoder_name, task_name
            )
            connector_out_dict[decoder_name] = self.connector_dict[decoder_name](
                connector_inp
            )
        else:
            # When there are multiple decoders
            connector_out_dict = {}
            for decoder_name, connector in self.connector_dict.items():
                if decoder_name in batch:
                    example_indices = batch["indices_per_decoder"][decoder_name]
                    connector_inp = enc_out_dict["encoder_outputs"][example_indices]
                    task_name = batch["task_names"][example_indices[0]]
                    connector_inp = self.pool_connector_inp(
                        connector_inp, decoder_name, task_name
                    )
                    connector_out_dict[decoder_name] = connector(connector_inp)

        # decoders
        total_loss, decs_out_dict, decs_losses_dict, decs_kwargs = self.decoders(
            batch, connector_out_dict, enc_kwargs
        )

        return total_loss, decs_out_dict, decs_losses_dict, enc_kwargs, decs_kwargs

    def pool_connector_inp(self, connector_inp, decoder_name, task_name):
        if decoder_name.startswith("linear") and task_name == Tasks.CLS:
            if self.enc_model_name in [
                "vision_transformer",
                "vision_transformer_hybrid",
            ]:
                connector_inp = connector_inp[:, 0, :]
            elif self.enc_model_name in ["swin_transformer", "swin_transformer_fpn"]:
                connector_inp = connector_inp.mean(1)
            else:
                raise ValueError(f"Unknown encoder_model_name={self.enc_model_name}")
        return connector_inp

    def init_weights(self, weights_init_fn):
        self.encoder.init_weights(weights_init_fn)
        self.decoders.init_weights(weights_init_fn)
        self.connector_dict.apply(weights_init_fn)


class EncoderAndDecodersForPreTraining(EncoderAndDecoders):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = EncoderForPreTraining(cfg)
        self.decoders = DecodersForPreTraining(cfg)
