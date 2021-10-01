#!/usr/bin/env python3

from pathlib import Path
from random import random
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor, einsum
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def make_dataset(root, subset) -> List[Tuple[Path, Path, Path]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'
    weak_path = root / subset / 'weak'

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))
    weak_labels = sorted(weak_path.glob("*.png"))

    return list(zip(images, full_labels, weak_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, transform=None,
                 mask_transform=None, augment=False, equalize=False):
        self.root_dir: str = root_dir
        self.transform: Callable = transform
        self.mask_transform: Callable = mask_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.files = make_dataset(root_dir, subset)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        img_path, gt_path, weak_path = self.files[index]

        img = Image.open(img_path)
        mask = Image.open(gt_path)
        weak_mask = Image.open(weak_path)

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            weak_mask = self.mask_transform(weak_mask)

        _, W, H = img.shape
        assert mask.shape == weak_mask.shape == (2, W, H)

        # Circle: 8011
        true_size = einsum("kwh->k", mask)
        bounds = einsum("k,b->kb", true_size, torch.tensor([0.9, 1.1], dtype=torch.float32))
        assert bounds.shape == (2, 2)  # binary, upper and lower bounds

        return {"img": img,
                "full_mask": mask,
                "weak_mask": weak_mask,
                "path": str(img_path),
                "true_size": true_size,
                "bounds": bounds}
