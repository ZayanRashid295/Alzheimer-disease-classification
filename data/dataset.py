"""
PyTorch Dataset for Alzheimer MRI: loads (path, label) pairs with transforms.
Handles corrupted images by skipping or raising based on config.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image


class AlzheimerMRIDataset(Dataset):
    """
    Dataset of MRI images with class labels.
    samples: list of (path, class_index).
    transform: callable that takes PIL Image and returns tensor.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
        skip_corrupted: bool = True,
    ):
        self.samples = samples
        self.transform = transform
        self.skip_corrupted = skip_corrupted
        self._valid_indices: Optional[List[int]] = None
        if skip_corrupted:
            self._build_valid_indices()

    def _build_valid_indices(self) -> None:
        """Optionally pre-scan and exclude corrupted files (lazy on first access)."""
        self._valid_indices = []
        for i in range(len(self.samples)):
            path, _ = self.samples[i]
            try:
                with Image.open(path) as img:
                    img.verify()
                self._valid_indices.append(i)
            except Exception:
                continue
        if not self._valid_indices:
            raise RuntimeError("No valid images found; all files may be corrupted.")

    def __len__(self) -> int:
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.samples)

    def _get_sample(self, index: int) -> Tuple[Path, int]:
        if self._valid_indices is not None:
            index = self._valid_indices[index]
        return self.samples[index]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self._get_sample(index)
        img = Image.open(path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB" and img.mode != "L":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
