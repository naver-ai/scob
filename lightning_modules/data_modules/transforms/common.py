"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import random
from typing import Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


def _interp(method):
    if method == "bicubic":
        return TF.InterpolationMode.BICUBIC
    elif method == "lanczos":
        return TF.InterpolationMode.LANCZOS
    elif method == "hamming":
        return TF.InterpolationMode.HAMMING
    else:
        return TF.InterpolationMode.BILINEAR


class ResizeTwoPic:
    """
    https://github.com/microsoft/unilm/blob/master/beit/datasets.py#L27

    resize function:
        TF.resize: size=(h, w)
        https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.resize.html#torchvision.transforms.functional.resize

        cv2.resize: size=(w, h)
        https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html#cv2.resize
    """

    def __init__(
        self,
        size: Union[Tuple, list],  # (height, width)
        second_size: Union[Tuple, list, None] = None,  # (height, width)
        interps: Tuple[str] = ("bilinear",),
        second_interps: Tuple[str] = ("lanczos",),
    ):
        self.size = size
        self.interps = [_interp(interp) for interp in interps]
        self.second_size = second_size
        self.second_interps = [_interp(interp) for interp in second_interps]

    def __call__(self, img, quads=None):
        interp = random.choice(self.interps)

        if isinstance(img, Image.Image):
            img_type_is_pillow = True
        else:
            img_type_is_pillow = False

        if img_type_is_pillow:
            out1 = TF.resize(img, self.size, interp)
        else:
            out1 = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            img_h, img_w = img.shape[:2]
            h_scale = self.size[0] / img_h
            w_scale = self.size[1] / img_w
            if len(quads) > 0:
                quads[:, :, 0] = quads[:, :, 0] * w_scale
                quads[:, :, 1] = quads[:, :, 1] * h_scale

        if self.second_size is None:
            return out1, quads
        else:
            interp2 = random.choice(self.second_interps)
            if img_type_is_pillow:
                out2 = TF.resize(img, self.second_size, interp2)
            else:
                out2 = cv2.resize(
                    img, self.second_size[::-1], interpolation=cv2.INTER_LINEAR
                )
            return out1, out2, quads


class W_Compose:
    """
    Modified pytorch compose
    pytorch.org/vision/0.10/transforms.html#torchvision.transforms.Compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, quads=None):
        second_img = None
        multiview_img = None
        for transform in self.transforms:
            if (
                isinstance(transform, ResizeTwoPic)
                and transform.second_size is not None
            ):
                img, second_img, quads = transform(img, quads)

            elif isinstance(transform, ResizeMultiview):
                img, multiview_img, quads = transform(img, quads)

            elif isinstance(transform, MoCo_PhotometricDistort):
                img, _ = transform(img, quads)
                multiview_img, quads = transform(multiview_img, quads)
            else:
                img, quads = transform(img, quads)

        if second_img is None and multiview_img is None:
            return img, quads
        elif multiview_img is not None:
            return img, multiview_img, quads
        else:
            return img, second_img, quads

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += f"\n    {transform}"
        format_string += "\n)"
        return format_string


class RandomRotate:
    """
    Random-Rotate. clockwise +
    """

    def __init__(self, prob, rot_range):
        self.probability = prob
        self.rot_range = rot_range

    def __call__(self, img, w_quads):
        if random.random() < self.probability:
            angle = random.uniform(self.rot_range[0], self.rot_range[1])

            img_h, img_w = img.shape[:2]
            center = (img_w / 2, img_h / 2)
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            img = cv2.warpAffine(img, rotate_matrix, (img_w, img_h))

            w_quads = self._rotate(w_quads, rotate_matrix)

        return img, w_quads

    @staticmethod
    def _rotate(_quad, rotate_matrix):
        if len(_quad) > 0:
            _additional = np.ones((len(_quad), 4, 1), dtype=_quad.dtype)
            _quad_target = np.concatenate([_quad, _additional], axis=2)
            _rotated_quad = np.matmul(_quad_target, rotate_matrix.T)
            _rotated_quad = _rotated_quad.astype(_quad.dtype)
        else:
            _rotated_quad = []
        return _rotated_quad


class CraftRandomCrop:
    def __init__(
        self, crop_aspect_ratio, attempts, num_box_thresholds, iou_threshold, crop_scale
    ):
        self.min_ar, self.max_ar = crop_aspect_ratio
        self.attempts = attempts
        self.num_box_thresholds = [None] + list(num_box_thresholds)
        self.threshold = iou_threshold
        self.min_scale, self.max_scale = crop_scale

    def __call__(self, img, w_quads):
        """Craft Random Crop
        Try random crop as max_trial, performs for imgs, c_quads.
        There is an aspect_ratio threshold and char box count threshold.
        None can be selected for the number of char boxes threshold.
        If None is selected, return data without cropping.
        ================================================================================
        When you crop, you can get out-of-boundary(OOB) coordinates.
        It works as follows to deal with them.
        [detector data processing]
        The OOB quads which located at the right, bottom, and right bottom are filtered.
        lightning_modules/data_modules/gt_maker.py ==> Refer to get_roi_bbox() function.
        ================================================================================
        Args:
            img (NDArray[uint8]): RGB image
            c_quads (NDArray[float32]): (N, 4, 2) quadrangle
            w_quads (NDArray[float32]): (N, 4, 2) quadrangle
            dc_polys (List[NDArray[float32]]): A list of 2d numpy polygons
        Returns:
            img (NDArray[uint8]): RGB image
            c_quads (NDArray[float32]): (N, 4, 2) cropped quadrangle
            w_quads (NDArray[float32]): (N, 4, 2) cropped quadrangle
            dc_polys (List[NDArray[float32]]): A list of 2d numpy polygons
        """
        img_h, img_w = img.shape[:2]
        num_box_thresh = random.choice(self.num_box_thresholds)
        w_bboxes = self._get_bbox(w_quads, img_h, img_w)

        if num_box_thresh is not None and w_bboxes is not None:
            for _ in range(self.attempts):
                rand_w = random.randrange(
                    round(self.min_scale * img_w), round(self.max_scale * img_w)
                )
                rand_h = random.randrange(
                    round(self.min_scale * img_h), round(self.max_scale * img_h)
                )

                aspect_ratio = rand_w / (rand_h + 1e-6)
                if aspect_ratio < self.min_ar or aspect_ratio > self.max_ar:
                    continue

                left = random.randrange(0, img_w - rand_w)
                top = random.randrange(0, img_h - rand_h)
                crop_bbox = np.array(
                    [left, top, left + rand_w, top + rand_h], dtype=np.float32
                )
                overlap = self._modified_jaccard_numpy(w_bboxes, crop_bbox)

                if np.sum(overlap > self.threshold) < num_box_thresh:
                    continue

                # crop success
                img = img[top : top + rand_h, left : left + rand_w]

                if len(w_quads) > 0:
                    w_quads -= crop_bbox[:2]

                break

        return img, w_quads

    @staticmethod
    def _get_bbox(quads, img_h, img_w):
        """Get bounding-box of quadrangle.
        Args:
            quads (ndrray[float32]): (N, 4, 2) or []
            img_h (int): image height
            img_w (int): image width
        Returns:
            bboxes (ndarray[float32]): (N, 4) x1y1x2y2 bbox
        """

        if len(quads) > 0:
            # bounding box 생성
            min_coords = np.min(quads, axis=1)
            max_coords = np.max(quads, axis=1)
            bboxes = np.concatenate([min_coords, max_coords], axis=1)

            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_h)

            # Remove if it's not a box. (check duplicated coordinates)
            problem_indices = np.where(
                (bboxes[:, 0] == bboxes[:, 2]) & (bboxes[:, 1] == bboxes[:, 3])
            )[0]
            bboxes = np.delete(bboxes, problem_indices, axis=0)
        else:
            bboxes = None
        return bboxes

    def _modified_jaccard_numpy(self, box_a, box_b):
        """
        Modified jaccard overlap (Modified IOU)  A ∩ B / A
        Args:
            box_a: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [4]
        Return:
            modified jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self._intersect(box_a, box_b)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        return inter / (area_a + 1e-6)

    @staticmethod
    def _intersect(box_a, box_b):
        """Calculate A ∩ B"""
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]


class Resize:
    """

    resize function:
        cv2.resize: size=(w, h)
        https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html#cv2.resize
    """

    def __init__(self, size, interpolation="random"):
        self.size = size  # (height, width)
        if interpolation == "random":
            self.resize_option = (
                cv2.INTER_LINEAR,
                cv2.INTER_NEAREST,
                cv2.INTER_AREA,
                cv2.INTER_CUBIC,
                cv2.INTER_LANCZOS4,
            )
        elif interpolation == "linear":
            self.resize_option = (cv2.INTER_LINEAR,)
        else:
            raise ValueError(f"Invalid interpolation: {interpolation}")

    def __call__(self, img, w_quads):
        interpolation = random.choice(self.resize_option)
        img_h, img_w = img.shape[:2]
        h_scale = self.size[0] / img_h
        w_scale = self.size[1] / img_w

        img = cv2.resize(img, self.size[::-1], interpolation=interpolation)

        if len(w_quads) > 0:
            w_quads[:, :, 0] = w_quads[:, :, 0] * w_scale
            w_quads[:, :, 1] = w_quads[:, :, 1] * h_scale

        return img, w_quads


class ResizeOD:
    """

    resize function:
        cv2.resize: size=(w, h)
        https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html#cv2.resize
    """

    def __init__(self, size, interpolation="random"):
        self.size = size  # (height, width)
        if interpolation == "random":
            self.resize_option = (
                cv2.INTER_LINEAR,
                cv2.INTER_NEAREST,
                cv2.INTER_AREA,
                cv2.INTER_CUBIC,
                cv2.INTER_LANCZOS4,
            )
        elif interpolation == "linear":
            self.resize_option = (cv2.INTER_LINEAR,)
        else:
            raise ValueError(f"Invalid interpolation: {interpolation}")

    def __call__(self, img, w_quads):
        interpolation = random.choice(self.resize_option)
        img_h, img_w = img.shape[:2]
        h_scale = self.size[0] / img_h
        w_scale = self.size[1] / img_w

        img = cv2.resize(img, self.size[::-1], interpolation=interpolation)

        return img, w_quads


class KeepAspectRatioBilinearResize:
    """
    """

    def __init__(self, size):
        # self.size: (h, w)
        self.size = size

    def __call__(self, img, w_quads):
        """Resize with keeping aspect_ratio"""
        img_h, img_w = img.shape[:2]
        major_axis = max(self.size)
        scale_ratio = major_axis / max((img_w, img_h))
        new_h = round(img_h * scale_ratio)
        new_w = round(img_w * scale_ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((major_axis, major_axis, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img

        if len(w_quads) > 0:
            w_quads *= scale_ratio

        return canvas, w_quads


class KeepAspectRatioBilinearResizeOD:

    def __init__(self, size):
        # self.size: (h, w)
        self.size = size

    def __call__(self, img, w_quads):
        """Resize with keeping aspect_ratio"""
        img_h, img_w = img.shape[:2]
        major_axis = max(self.size)
        scale_ratio = major_axis / max((img_w, img_h))
        new_h = round(img_h * scale_ratio)
        new_w = round(img_w * scale_ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((major_axis, major_axis, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img

        return canvas, w_quads


class RandomCrop:
    """
    For object detection
    """

    def __init__(self, prob, crop_size):
        # # self.size: (h, w)
        self.prob = prob
        self.crop_size = crop_size

    def __call__(self, img, w_quads):
        """Resize with keeping aspect_ratio"""
        if random.random() < self.prob:
            img_h, img_w = img.shape[:2]

            if img_h <= self.crop_size[0]:
                c_h = img_h
            else:
                c_h = random.randrange(
                    self.crop_size[0], min(img_h, self.crop_size[1]) + 1
                )
            if img_w <= self.crop_size[0]:
                c_w = img_w
            else:
                c_w = random.randrange(
                    self.crop_size[0], min(img_w, self.crop_size[1]) + 1
                )

            if img_h <= c_h:
                crop_tl_y = 0
            else:
                crop_tl_y = random.randrange(0, img_h - c_h)

            if img_w <= c_w:
                crop_tl_x = 0
            else:
                crop_tl_x = random.randrange(0, img_w - c_w)

            crop_br_y = crop_tl_y + c_h
            crop_br_x = crop_tl_x + c_w

            img = img[crop_tl_y:crop_br_y, crop_tl_x:crop_br_x, :]

            x_ratio = [crop_tl_x / img_w, crop_br_x / img_w]
            y_ratio = [crop_tl_y / img_h, crop_br_y / img_h]

            if len(w_quads) > 0:
                ex_areas = (w_quads[:, 2, 0] - w_quads[:, 0, 0]) * (
                    w_quads[:, 2, 1] - w_quads[:, 0, 1]
                )

                w_quads[:, :, 0] = (w_quads[:, :, 0] - x_ratio[0]) / (
                    x_ratio[1] - x_ratio[0]
                )
                w_quads[:, :, 1] = (w_quads[:, :, 1] - y_ratio[0]) / (
                    y_ratio[1] - y_ratio[0]
                )
                w_quads = np.clip(w_quads, 0, 1)
                areas = (w_quads[:, 2, 0] - w_quads[:, 0, 0]) * (
                    w_quads[:, 2, 1] - w_quads[:, 0, 1]
                )

                w_quads[areas == 0.0] = -10

        return img, w_quads


class MultiScaleResize:

    def __init__(self, size, shorter_sides, prob):
        # self.size: (h, w)
        self.canvas_size = size
        # shorter_sides = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
        self.shorter_side = random.choice(shorter_sides)
        self.prob = prob

    def __call__(self, img, w_quads):
        """Resize with keeping aspect_ratio"""
        if random.random() < self.prob:
            img_h, img_w = img.shape[:2]
            scale_ratio = self.shorter_side / min(img_h, img_w)
            fit_ratio = min(self.canvas_size[0] / img_h, self.canvas_size[1] / img_w)
            if fit_ratio < scale_ratio:
                scale_ratio = fit_ratio
            new_h = round(img_h * scale_ratio)
            new_w = round(img_w * scale_ratio)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros(
                (self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8
            )
            canvas[:new_h, :new_w] = img
        else:
            img_h, img_w = img.shape[:2]
            major_axis = max(self.canvas_size)
            scale_ratio = major_axis / max((img_w, img_h))
            new_h = round(img_h * scale_ratio)
            new_w = round(img_w * scale_ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((major_axis, major_axis, 3), dtype=np.uint8)
            canvas[:new_h, :new_w] = img

        return canvas, w_quads


class ResizeMultiview:
    """

    resize function:
        cv2.resize: size=(w, h)
        https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html#cv2.resize
    """

    def __init__(self, size, interpolation="random"):
        self.size = size  # (height, width)
        if interpolation == "random":
            self.resize_option = (
                cv2.INTER_LINEAR,
                cv2.INTER_NEAREST,
                cv2.INTER_AREA,
                cv2.INTER_CUBIC,
                cv2.INTER_LANCZOS4,
            )
        elif interpolation == "linear":
            self.resize_option = (cv2.INTER_LINEAR,)
        else:
            raise ValueError(f"Invalid interpolation: {interpolation}")

    def __call__(self, img, w_quads):
        interpolation = random.choice(self.resize_option)
        img_h, img_w = img.shape[:2]
        h_scale = self.size[0] / img_h
        w_scale = self.size[1] / img_w

        resized_img = cv2.resize(img, self.size[::-1], interpolation=interpolation)

        multiview_interpolation = random.choice(self.resize_option)
        multiview_resized_img = cv2.resize(
            img, self.size[::-1], interpolation=multiview_interpolation
        )

        if len(w_quads) > 0:
            w_quads[:, :, 0] = w_quads[:, :, 0] * w_scale
            w_quads[:, :, 1] = w_quads[:, :, 1] * h_scale

        return resized_img, multiview_resized_img, w_quads


class PhotometricDistort:
    def __init__(self, prob):
        self.augmentation = A.Compose(
            [
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(p=0.2),
                A.ChannelShuffle(),
                A.CLAHE(),
            ],
            p=prob,
        )

    def __call__(self, img, w_quads):
        img = self.augmentation(image=img)["image"]
        return img, w_quads


class MoCo_PhotometricDistort:
    # follow PhotometricDistort augmentationss of moco v2. Torchtransform is converted to albumentation
    # https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/main_moco.py#L225
    def __init__(self, prob):
        self.augmentation = A.Compose(
            [
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(sigma_limit=[0.1, 2.0], p=0.5),
            ],
        )

    def __call__(self, img, w_quads):
        img = self.augmentation(image=img)["image"]
        return img, w_quads


class Otor_OriginDistort:
    def __init__(self, prob):
        self.augmentation = A.Compose(
            [
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(p=0.2),
                A.ChannelShuffle(),
                A.CLAHE(),
                A.GaussianBlur(sigma_limit=[0.1, 2.0], p=0.5),
            ],
            p=prob,
        )

    def __call__(self, img, w_quads):
        img = self.augmentation(image=img)["image"]
        return img, w_quads


TRANSFORM_NAME_TO_CLASS = {
    "RandomRotate": RandomRotate,
    "CraftRandomCrop": CraftRandomCrop,
    "Resize": Resize,
    "ResizeOD": ResizeOD,
    "PhotometricDistort": PhotometricDistort,
    "MoCo_PhotometricDistort": MoCo_PhotometricDistort,
    "ResizeTwoPic": ResizeTwoPic,
    "ResizeMultiview": ResizeMultiview,
    "KeepAspectRatioBilinearResize": KeepAspectRatioBilinearResize,
    "RandomCrop": RandomCrop,
    "MultiScaleResize": MultiScaleResize,
    "KeepAspectRatioBilinearResizeOD": KeepAspectRatioBilinearResizeOD,
    "Otor_OriginDistort": Otor_OriginDistort,
}
