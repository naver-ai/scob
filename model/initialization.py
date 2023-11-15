"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import torch.nn as nn
from timm.models.layers import trunc_normal_


def weights_init(init_type="default"):
    """Adopted and modified from FUNIT
    https://github.com/clovaai/dmfont/blob/master/models/modules/modules.py#L14
    """

    def init_weight_(w):
        if w is None or w.ndim < 2:
            return

        if init_type == "gaussian":
            nn.init.normal_(w, 0.0, 0.02)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(w)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(w)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(w)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(w)
        elif init_type == "default":
            pass
        else:
            raise ValueError(f"Unsupported initialization: {init_type}")

    def init_bias_(b):
        if b is not None:
            nn.init.zeros_(b)

    def init_fn(m):
        # pylint: disable=W0212
        if isinstance(
            m, (nn.modules.conv._ConvNd, nn.Linear, nn.RNNBase, nn.RNNCellBase)
        ):
            for n, p in m.named_parameters():
                if "weight" in n:
                    init_weight_(p)
                elif "bias" in n:
                    init_bias_(p)

    return init_fn


def swin_fpn_weights_init(m):
    """
    - https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py#L574
    - https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/necks/fpn.py#L163
    """

    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        # xavier_init(m, distribution='uniform')
        nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
