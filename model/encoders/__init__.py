"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

from model.model_utils import load_state_dict


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.encoder_name = self.model_cfg.encoder.encoder_model_name
        self.model = get_encoder(self.model_cfg.encoder)
        self.patch_size = self.get_patch_size()

    def get_patch_size(self):
        if self.encoder_name == "vision_transformer":
            patch_size = self.model_cfg.encoder[self.encoder_name].patch_size
        elif self.encoder_name == "vision_transformer_hybrid":
            patch_size = self.model_cfg.encoder[self.encoder_name].raw_patch_size
        elif self.encoder_name in ["swin_transformer", "swin_transformer_fpn"]:
            patch_size = self.model_cfg.encoder[self.encoder_name].patch_size * 8
        else:
            raise ValueError(f"Unknown encoder_model_name={self.encoder_name}")
        return patch_size

    def forward(self, batch):
        encoder_outputs = self.encoder_forward(batch)
        output_dict = {"encoder_outputs": encoder_outputs}
        kwargs = {"patch_size": self.patch_size}

        return output_dict, kwargs

    def encoder_forward(self, batch):
        """Network forward with shuffle option.
        This is different from Encoder's forward since it contains several more
        if-else cases about shufflig.
        """
        if self.encoder_name in ["vision_transformer", "vision_transformer_hybrid"]:
            encoder_outputs = self.vit_forward(batch)
        elif self.encoder_name in ["swin_transformer", "swin_transformer_fpn"]:
            encoder_outputs = self.swin_forward(batch)
        else:
            raise ValueError(f"Unknown self.encoder_name-{self.encoder_name}")

        return encoder_outputs

    def init_weights(self, weights_init_fn):
        for module_name, module in self.__dict__["_modules"].items():
            if module_name != "model":  # pretrained weight from timm
                module.apply(weights_init_fn)

    def vit_forward(self, batch):
        x = self.model.patch_embed(batch["imgs"])
        batch_size, seq_len, _ = x.size()

        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.model.cls_token.expand(batch_size, -1, -1)
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat(
                (
                    cls_token,
                    self.model.dist_token.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)

        x = x[:, 1:, :]  # exclude CLS token

        return x

    def swin_forward(self, batch):
        x = self.model.patch_embed(batch["imgs"])
        batch_size, seq_len, _ = x.size()

        if self.model.absolute_pos_embed is not None:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)

        if self.encoder_name == "swin_transformer":
            x = self.model.layers(x)
            x = self.model.norm(x)  # B L C
        elif self.encoder_name == "swin_transformer_fpn":
            x = self.model.forward_layers(x)

        return x


class EncoderForPreTraining(Encoder):
    """Encoder for pretraining"""


def get_encoder(encoder_cfg):
    enc_model_name = encoder_cfg.encoder_model_name
    if enc_model_name == "swin_transformer":
        transformer_cfg = encoder_cfg[enc_model_name]
        encoder = SwinTransformer(
            img_size=transformer_cfg.img_size,
            patch_size=transformer_cfg.patch_size,
            in_chans=transformer_cfg.in_chans,
            num_classes=0,
            embed_dim=transformer_cfg.embed_dim,
            depths=transformer_cfg.depths,
            num_heads=transformer_cfg.num_heads,
            window_size=transformer_cfg.window_size,
        )
    else:
        raise ValueError(f"Unknown encoder_cfg.encoder_model_name={enc_model_name}")

    if encoder_cfg.encoder_model_weight_path:
        loaded_state_dict = torch.load(
            encoder_cfg.encoder_model_weight_path, map_location="cpu"
        )
        if enc_model_name in ["swin_transformer", "swin_transformer_fpn"]:
            loaded_state_dict = loaded_state_dict["model"]
        kwargs = {}
        if enc_model_name.startswith("swin_transformer"):
            kwargs["window_size"] = transformer_cfg.window_size
        load_state_dict(encoder, loaded_state_dict, "timm_pretrained_model", kwargs)

    return encoder
