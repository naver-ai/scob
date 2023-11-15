"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import math
import os
from copy import deepcopy
from itertools import chain

import torch
import torch.nn.functional as F
from asian_bart import AsianBartTokenizer
from tokenizers import pre_tokenizers
from transformers import AutoTokenizer

from utils.dataset_utils import get_image_normalize_mean_and_std

W_PRETRAINED_MODEL = "w_pretrained_model"
TIMM_PRETRAINED_MODEL = "timm_pretrained_model"


@torch.no_grad()
def load_state_dict(model, loaded_state_dict, loaded_model_type, kwargs={}):
    assert loaded_model_type in [TIMM_PRETRAINED_MODEL, W_PRETRAINED_MODEL]
    model_state_dict = model.state_dict()

    loaded_state_dict_keys = list(loaded_state_dict.keys())
    model_state_dict_keys = list(model_state_dict.keys())

    new_model_state_dict = {}

    # Collect weights that have exactly the same shape
    # new_model_state_dict, model_state_dict_keys, loaded_state_dict_keys are updated.
    _base_load_state_dict(
        model_state_dict,
        new_model_state_dict,
        loaded_state_dict,
        model_state_dict_keys,
        loaded_state_dict_keys,
        loaded_model_type,
    )

    # (Encoder) 2D interpolation of position embeddings
    # new_model_state_dict, model_state_dict_keys, loaded_state_dict_keys are updated.
    _encoder_resize_pos_embed(
        model,
        model_state_dict,
        new_model_state_dict,
        loaded_state_dict,
        model_state_dict_keys,
        loaded_state_dict_keys,
        loaded_model_type,
        model_window_size=kwargs.get("window_size", None),
    )

    # Update model weights
    model_state_dict.update(new_model_state_dict)
    model.load_state_dict(model_state_dict)

    # - (Decoder) Partially loads decoder word embeddings
    #   - model_state_dict_keys and loaded_state_dict_keys are updated
    #   - model weights are updated
    if loaded_model_type == W_PRETRAINED_MODEL:
        _decoder_partially_load_word_emb(
            model,
            model_state_dict,
            loaded_state_dict,
            model_state_dict_keys,
            loaded_state_dict_keys,
        )

        _decoder_partially_load_pos_emb(
            model,
            model_state_dict,
            loaded_state_dict,
            model_state_dict_keys,
            loaded_state_dict_keys,
        )

    # - Print the weight transfer result
    _print_weight_transfer_result(
        model_state_dict,
        loaded_state_dict,
        model_state_dict_keys,
        loaded_state_dict_keys,
        loaded_model_type,
    )


def _base_load_state_dict(
    model_state_dict,
    new_model_state_dict,
    loaded_state_dict,
    model_state_dict_keys,
    loaded_state_dict_keys,
    loaded_model_type,
):
    """Collect weights that have exactly the same shape"""
    for k in model_state_dict.keys():
        loaded_k = get_loaded_key(k, loaded_model_type, loaded_state_dict_keys)
        if loaded_k is None:
            continue

        if (
            loaded_k in loaded_state_dict
            and model_state_dict[k].shape == loaded_state_dict[loaded_k].shape
        ):
            new_model_state_dict[k] = loaded_state_dict[loaded_k]
            loaded_state_dict_keys.remove(loaded_k)
            model_state_dict_keys.remove(k)


def _encoder_resize_pos_embed(
    model,
    model_state_dict,
    new_model_state_dict,
    loaded_state_dict,
    model_state_dict_keys,
    loaded_state_dict_keys,
    loaded_model_type,
    model_window_size=None,
):
    """2D interpolation of position embeddings"""

    if (
        loaded_model_type == TIMM_PRETRAINED_MODEL
        and model.__class__.__name__.startswith("SwinTransformer")
    ) or (
        loaded_model_type == W_PRETRAINED_MODEL
        and model.encoder.model.__class__.__name__.startswith("SwinTransformer")
    ):
        for k in deepcopy(model_state_dict_keys):
            if not k.endswith("relative_position_bias_table"):
                continue

            loaded_k = get_loaded_key(k, loaded_model_type, loaded_state_dict_keys)
            if loaded_k is None:
                continue

            loaded_window_size = int(
                (math.sqrt(loaded_state_dict[loaded_k].shape[0]) + 1) / 2
            )
            if model_window_size is None:
                model_window_size = int(
                    (math.sqrt(model_state_dict[k].shape[0]) + 1) / 2
                )
            print(
                f"  - {k}: resized position embedding: "
                f"from {loaded_window_size} to {model_window_size}"
            )
            new_pos_table = _resize_pos_embed_for_swin(
                loaded_state_dict[loaded_k], model_window_size
            )
            if model_state_dict[k].shape[1] == new_pos_table.shape[1]:
                new_model_state_dict[k] = new_pos_table
            else:
                # When num_heads is different
                num_heads = model_state_dict[k].shape[1]
                new_pos_table = (
                    new_pos_table.mean(1).repeat(num_heads).reshape(-1, num_heads)
                )
                new_model_state_dict[k] = new_pos_table

            loaded_state_dict_keys.remove(loaded_k)
            model_state_dict_keys.remove(k)
    elif (
        loaded_model_type == TIMM_PRETRAINED_MODEL
        and model.__class__.__name__ == "VisionTransformer"
    ) or (
        loaded_model_type == W_PRETRAINED_MODEL
        and model.encoder.model.__class__.__name__ == "VisionTransformer"
    ):
        for k in deepcopy(model_state_dict_keys):
            if not k.endswith("pos_embed"):
                continue

            loaded_k = get_loaded_key(k, loaded_model_type, loaded_state_dict_keys)
            if loaded_k is None:
                continue

            gs_old = int(math.sqrt(loaded_state_dict[loaded_k].shape[1] - 1))
            gs_new = int(math.sqrt(model_state_dict[k].shape[1] - 1))

            print(f"  - {k}: resized position embedding: " f"from {gs_old} to {gs_new}")

            new_pos_table = _resize_pos_embed_for_vit(
                loaded_state_dict[loaded_k], gs_old, gs_new
            )
            assert model_state_dict[k].shape[1] == new_pos_table.shape[1]
            new_model_state_dict[k] = new_pos_table

            loaded_state_dict_keys.remove(loaded_k)
            model_state_dict_keys.remove(k)
    else:
        raise NotImplementedError


def _resize_pos_embed_for_vit(posemb, gs_old, gs_new):
    # https://github.com/rwightman/pytorch-image-models/blob/75144395739162a153dae5628320b85b5895634b/timm/models/vision_transformer.py#L508
    # E.g. posemb.shape: [1, 577, 768]
    # E.g. gs_old: 24 (for 384x384)
    # E.g. gs_new: 14 (for 224x224)
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # 1 for [CLS] token
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def _resize_pos_embed_for_swin(posemb, ws):
    # https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/timm/models/vision_transformer.py#L487
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    gs_old = int(math.sqrt(posemb.shape[0]))
    gs_new = 2 * ws - 1
    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new), mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb_grid = posemb_grid[0]
    return posemb_grid


def _decoder_partially_load_word_emb(
    model,
    model_state_dict,
    loaded_state_dict,
    model_state_dict_keys,
    loaded_state_dict_keys,
):
    """(Decoder) Partially loads decoder word embeddings"""

    word_embeddings_keys = {
        "bert": [
            "model.bert.embeddings.word_embeddings.weight",
            "model.cls.predictions.bias",
            "model.cls.predictions.decoder.weight",
            "model.cls.predictions.decoder.bias",
        ],
        "roberta": [
            "model.roberta.embeddings.word_embeddings.weight",
            "model.lm_head.bias",
            "model.lm_head.decoder.weight",
            "model.lm_head.decoder.bias",
        ],
        "facebook/bart": [
            "model.model.decoder.embed_tokens.weight",
            "model.lm_head.weight",
        ],
    }

    for k in deepcopy(model_state_dict_keys):
        loaded_k = get_loaded_key(k, W_PRETRAINED_MODEL, loaded_state_dict_keys)
        if loaded_k is None:
            continue

        for word_embedding_key in chain(*word_embeddings_keys.values()):
            if k.endswith(word_embedding_key):
                target_module = model
                for inner_k in k.split("."):
                    target_module = getattr(target_module, inner_k)

                n_min_tokens = min(
                    target_module.shape[0], loaded_state_dict[loaded_k].shape[0]
                )
                target_module[:n_min_tokens].copy_(
                    loaded_state_dict[loaded_k][:n_min_tokens]
                )

                print(
                    f"  - {k} {tuple(model_state_dict[k].shape)} "
                    "is partially initialized by "
                    f"{loaded_k} {tuple(loaded_state_dict[loaded_k].shape)}"
                )
                loaded_state_dict_keys.remove(loaded_k)
                model_state_dict_keys.remove(k)
    print()


def _decoder_partially_load_pos_emb(
    model,
    model_state_dict,
    loaded_state_dict,
    model_state_dict_keys,
    loaded_state_dict_keys,
):
    """(Decoder) Partially loads decoder position embeddings"""

    pos_embeddings_keys = {
        "bert": [
            "model.bert.embeddings.position_embeddings.weight",
        ],
        "roberta": [
            "model.roberta.embeddings.position_embeddings.weight",
        ],
        "facebook/bart": [
            "model.model.decoder.embed_positions.weight",
        ],
    }

    for k in deepcopy(model_state_dict_keys):
        loaded_k = get_loaded_key(k, W_PRETRAINED_MODEL, loaded_state_dict_keys)
        if loaded_k is None:
            continue

        for pos_embedding_key in chain(*pos_embeddings_keys.values()):
            if k.endswith(pos_embedding_key):
                target_module = model
                for inner_k in k.split("."):
                    target_module = getattr(target_module, inner_k)

                shorter_max_seq_len = min(
                    target_module.shape[0], loaded_state_dict[loaded_k].shape[0]
                )
                target_module[:shorter_max_seq_len].copy_(
                    loaded_state_dict[loaded_k][:shorter_max_seq_len]
                )

                print(
                    f"  - {k} {tuple(model_state_dict[k].shape)} "
                    "is partially initialized by "
                    f"{loaded_k} {tuple(loaded_state_dict[loaded_k].shape)}"
                )
                loaded_state_dict_keys.remove(loaded_k)
                model_state_dict_keys.remove(k)
    print()


def get_loaded_key(key, loaded_model_type, loaded_state_dict_keys):
    if loaded_model_type == TIMM_PRETRAINED_MODEL:
        loaded_key = key
    elif loaded_model_type == W_PRETRAINED_MODEL:
        loaded_key = modify_w_key(key, loaded_state_dict_keys)
    else:
        raise ValueError(f"Unknown loaded_model_type={loaded_model_type}")

    return loaded_key


def modify_w_key(key, loaded_state_dict_keys):
    key_candidates = ["net." + key]
    new_key = None
    for key_candidate in key_candidates:
        if key_candidate in loaded_state_dict_keys:
            new_key = key_candidate
            break

    return new_key


def _print_weight_transfer_result(
    model_state_dict,
    loaded_state_dict,
    model_state_dict_keys,
    loaded_state_dict_keys,
    loaded_model_type,
):
    print(f"Remaining keys of model (not initialized from {loaded_model_type})")
    for k in model_state_dict_keys:
        print(f"  - {k}\t{tuple(model_state_dict[k].shape)}")
    print()

    print(f"Remaining keys of {loaded_model_type} (not used to initialize)")
    for k in loaded_state_dict_keys:
        print(f"  - {k}\t{tuple(loaded_state_dict[k].shape)}")
    print()


def get_tokenizer(
    huggingface_path,
    tokenizer_name,
    decoder_name,
    resume_model_path=None,
    w_pretrained_model_path=None,
    tokenizer_path=None,
):
    """Get Tokenizer for model
    Args:
        huggingface_path (str): huggingface_path
        tokenizer_name (str): tokenizer name
        decoder_name (str): decoder name
        resume_model_path (str): resume_model_path
        w_pretrained_model_path (str): pretrained_model_path
        tokenizer_path (str): tokenizer_path

    Returns:
        tokenizer (Tokenizer): tokenizer
    """
    pretrained_model_path = None
    if resume_model_path:
        pretrained_model_path = resume_model_path
    elif w_pretrained_model_path:
        pretrained_model_path = w_pretrained_model_path
    assert (tokenizer_path is None) or (pretrained_model_path is None)

    if tokenizer_path is None and pretrained_model_path is not None:
        # Check ".../tokenizer/" dir exists,
        # since encoder only pre-trained model doesn't have ".../tokenizer/" dir
        tokenizer_path_parent = os.path.join(
            os.path.dirname(pretrained_model_path), "tokenizers"
        )
        if os.path.exists(tokenizer_path_parent):
            tokenizer_path = os.path.join(
                tokenizer_path_parent, decoder_name, tokenizer_name
            )
        else:
            raise ValueError(f"Tokenizer dir not found: {tokenizer_path_parent}")

    if tokenizer_path is None:
        tokenizer_path = os.path.join(huggingface_path, tokenizer_name)
    else:
        if not os.path.exists(tokenizer_path):
            raise ValueError(
                f"Current tokenizer, {tokenizer_name}, "
                "is different from the one in pre-trained model."
            )

    if tokenizer_name in [
        "bert-base-cased",
        "bert-base-multilingual-cased",
        "roberta-base",
        "xlm-roberta-base",
        "facebook/bart-base",
        "facebook/mbart-large-cc25",
        "char_en__bert-base-cased",
    ]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    elif tokenizer_name == "hyunwoongko/asian-bart-ecjk":
        tokenizer = AsianBartTokenizer.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Invalid tokenizer_name={tokenizer_name}")

    if tokenizer_name == "char_en__bert-base-cased":
        char_pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split("", behavior="removed")]
        )
        tokenizer.backend_tokenizer.pre_tokenizer = char_pre_tokenizer

    print("-" * 80)
    print(f"tokenizer_loaded from {tokenizer_path}")
    print("-" * 80)
    return tokenizer


class ImageUnnormalizer:
    def __init__(self, image_normalize):
        # For unnormalizing image
        mean_and_std = get_image_normalize_mean_and_std(image_normalize)
        if mean_and_std:
            self.normalize_mean = torch.tensor(mean_and_std[0])
            self.normalize_std = torch.tensor(mean_and_std[1])
        else:
            self.normalize_mean = None
            self.normalize_std = None

    def to_device(self, device):
        if self.normalize_mean is not None:
            self.normalize_mean = self.normalize_mean.to(device)
        if self.normalize_std is not None:
            self.normalize_std = self.normalize_std.to(device)

    @staticmethod
    def detach_clone_resize_tocpu(targets):
        """Do [detach, clone, resize, to_cpu] sequentially.
        Args:
            targets (torch.Tensor): NCHW or NHW pytorch tensor
        Returns:
            output (torch.Tensor): NCHW processed output
        """
        targets = targets.detach().to("cpu").float()
        if targets.ndim == 3:
            targets = torch.unsqueeze(targets, 1)
        elif targets.ndim != 4:
            raise ValueError(f"Invalid target shape: {targets.shape}")
        return targets

    def unnormalize(self, imgs):
        """Log image after denormalize"""
        imgs = self.detach_clone_resize_tocpu(imgs)
        imgs *= self.normalize_std.view(1, -1, 1, 1)
        imgs += self.normalize_mean.view(1, -1, 1, 1)
        imgs = torch.clamp(imgs, 0, 1)
        imgs = (imgs * 255.0).to(torch.uint8)
        # type(self.logger[0]) is pytorch_lightning.loggers.tensorboard.TensorBoardLogger
        # self.logger[0].experiment.add_images(log_name, imgs, step, dataformats="NCHW")
        return imgs


def get_parameter_names(model, forbidden_layer_types):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L996
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
