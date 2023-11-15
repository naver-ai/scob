from typing import List, Tuple, Union

import torch
import torchvision.transforms as transforms

from lightning_modules.data_modules.transforms.common import (
    TRANSFORM_NAME_TO_CLASS,
    W_Compose,
)
from utils.dataset_utils import get_image_normalize_mean_and_std


class TransformerDecoderTransformForFineTuning:
    """
    - BEiT:
    https://github.com/microsoft/unilm/blob/master/beit/datasets.py#L27

    - TrOCR:
    https://github.com/microsoft/unilm/blob/53995b4876464146365693396aaaa09e88a4494e/trocr/data_aug.py#L120
    """

    def __init__(
        self,
        size: Union[Tuple, List],
        transforms_list=None,
        image_normalize="imagenet_default",
    ):
        self.common_transform = self.__get_common_transform(size, transforms_list)
        self.patch_transform = self.__get_patch_transform(image_normalize)

    def __call__(self, img, quads):
        for_patches, quads = self.common_transform(img, quads)
        for_patches = self.patch_transform(for_patches)
        return for_patches, quads

    @staticmethod
    def __get_common_transform(size, transforms_list):
        tranforms = []
        for transform_obj in transforms_list:
            transform_class = TRANSFORM_NAME_TO_CLASS[transform_obj.name]
            if transform_obj.params is not None:
                params = dict(transform_obj.params)
            else:
                params = {}

            if transform_obj.name in [
                "Resize",
                "ResizeOD",
                "KeepAspectRatioBilinearResize",
                "ResizeMultiview",
                "MultiScaleResize",
                "KeepAspectRatioBilinearResizeOD",
            ]:
                params["size"] = size
            elif transform_obj.name == "ResizeTwoPic":
                params["size"] = size

            tranforms.append(transform_class(**params))

        return W_Compose(tranforms)

    @staticmethod
    def __get_patch_transform(image_normalize):
        patch_trans = [transforms.ToTensor()]

        mean_and_std = get_image_normalize_mean_and_std(image_normalize)
        if mean_and_std:
            patch_trans.append(
                transforms.Normalize(
                    mean=torch.tensor(mean_and_std[0]),
                    std=torch.tensor(mean_and_std[1]),
                )
            )
        return transforms.Compose(patch_trans)


class TransformerDecoderTransformForPreTraining:
    """
    - BEiT:
    https://github.com/microsoft/unilm/blob/master/beit/datasets.py#L27

    - TrOCR:
    https://github.com/microsoft/unilm/blob/53995b4876464146365693396aaaa09e88a4494e/trocr/data_aug.py#L120
    """

    def __init__(
        self,
        size: Union[Tuple, List],
        second_size: Union[Tuple, List, None] = None,
        transforms_list=None,
        image_normalize="imagenet_default",
        has_multiview=False,
    ):
        self.common_transform = self.__get_common_transform(
            size, second_size, transforms_list
        )
        self.patch_transform = self.__get_patch_transform(image_normalize)
        self.second_size = second_size
        self.has_multiview = has_multiview

    def __call__(self, img, quads=None):
        if self.second_size is None and not self.has_multiview:
            for_patches, quads = self.common_transform(img, quads)
            for_patches = self.patch_transform(for_patches)
            return for_patches, quads
        elif self.has_multiview:
            for_patches, for_multiview_patches, quads = self.common_transform(
                img, quads
            )
            for_patches = self.patch_transform(for_patches)
            for_multiview_patches = self.patch_transform(for_multiview_patches)
            return for_patches, for_multiview_patches, quads
        else:
            raise ValueError("This transforms condition is not supported.")

    @staticmethod
    def __get_common_transform(size, second_size, transforms_list):
        tranforms = []
        for transform_obj in transforms_list:
            transform_class = TRANSFORM_NAME_TO_CLASS[transform_obj.name]
            if transform_obj.params is not None:
                params = dict(transform_obj.params)
            else:
                params = {}

            if transform_obj.name in [
                "Resize",
                "KeepAspectRatioBilinearResize",
                "ResizeMultiview",
            ]:
                params["size"] = size
            elif transform_obj.name == "ResizeTwoPic":
                params["size"] = size
                params["second_size"] = second_size

            tranforms.append(transform_class(**params))

        return W_Compose(tranforms)

    @staticmethod
    def __get_patch_transform(image_normalize):
        patch_trans = [transforms.ToTensor()]

        mean_and_std = get_image_normalize_mean_and_std(image_normalize)
        if mean_and_std:
            patch_trans.append(
                transforms.Normalize(
                    mean=torch.tensor(mean_and_std[0]),
                    std=torch.tensor(mean_and_std[1]),
                )
            )
        return transforms.Compose(patch_trans)
