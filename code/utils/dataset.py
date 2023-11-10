#!/usr/bin/env python3

# MIT License

# Copyright (c) 2023 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
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
        K, _, _ = mask.shape
        assert mask.shape == weak_mask.shape == (K, W, H)

        # Circle: 8011
        true_size = einsum("kwh->k", mask).type(torch.float32)
        bounds = einsum("k,b->kb", true_size, torch.tensor([0.9, 1.1], dtype=torch.float32))
        assert bounds.shape == (K, 2)  # binary, upper and lower bounds

        return {"img": img,
                "full_mask": mask,
                "weak_mask": weak_mask,
                "path": str(img_path),
                "true_size": true_size,
                "bounds": bounds}
