"""
Medical-safe augmentations and normalization for MRI images.
Resize to target size; handle grayscale (1 channel) and RGB (3 channels).
"""
from typing import Tuple, Optional

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image


def _pil_to_tensor(pil_img) -> torch.Tensor:
    """PIL to tensor; shape (C, H, W). If grayscale, C=1."""
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # (1, H, W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return torch.from_numpy(arr).float() / 255.0


def _get_grayscale_or_rgb(pil_img) -> torch.Tensor:
    """Convert to tensor and ensure consistent channels (1 or 3)."""
    t = _pil_to_tensor(pil_img)
    if t.shape[0] == 1:
        return t
    if t.shape[0] == 3:
        return t
    # e.g. RGBA -> take first 3
    return t[:3]


class ToTensorMRI:
    """Convert PIL to tensor (C, H, W) and optionally force grayscale (repeat to 3) or keep as-is."""

    def __init__(self, channels: int = 3):
        self.channels = channels

    def __call__(self, img: Image.Image) -> torch.Tensor:
        x = _get_grayscale_or_rgb(img)
        if x.shape[0] == 1 and self.channels == 3:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 3 and self.channels == 1:
            x = x[:1]
        return x


class NormalizeMRI:
    """Normalize with mean and std (per channel). Default ImageNet-like for 3ch; single value for 1ch."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        num_channels: int = 3,
    ):
        if num_channels == 1:
            mean, std = (0.5,), (0.5,)
        self.mean = torch.tensor(mean).view(num_channels, 1, 1)
        self.std = torch.tensor(std).view(num_channels, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


def get_train_transforms(
    image_size: Tuple[int, int],
    augmentation: dict,
    channels: int = 3,
) -> transforms.Compose:
    """Training transforms: resize, medical-safe augmentation, to tensor, normalize."""
    aug_list = [
        transforms.Resize(image_size),
        transforms.RandomRotation(augmentation.get("rotation_degrees", 15)),
        transforms.RandomAffine(
            degrees=0,
            translate=augmentation.get("affine_translate", (0.1, 0.1)),
            scale=augmentation.get("affine_scale", (0.9, 1.1)),
        ),
        transforms.RandomHorizontalFlip(p=augmentation.get("horizontal_flip_p", 0.5)),
    ]
    if augmentation.get("vertical_flip_p", 0.0) > 0:
        aug_list.append(
            transforms.RandomVerticalFlip(p=augmentation["vertical_flip_p"])
        )
    aug_list.extend([
        ToTensorMRI(channels=channels),
        NormalizeMRI(num_channels=channels),
    ])
    return transforms.Compose(aug_list)


def get_eval_transforms(
    image_size: Tuple[int, int],
    channels: int = 3,
) -> transforms.Compose:
    """Eval/val/test: resize, to tensor, normalize only."""
    return transforms.Compose([
        transforms.Resize(image_size),
        ToTensorMRI(channels=channels),
        NormalizeMRI(num_channels=channels),
    ])
