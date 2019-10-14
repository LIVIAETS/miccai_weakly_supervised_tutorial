#!/usr/bin/env python3

import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from random import random

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []

    img_path = os.path.join(root, mode, 'Img')
    mask_path = os.path.join(root, mode, 'GT')

    images = os.listdir(img_path)
    labels = os.listdir(mask_path)

    images.sort()
    labels.sort()

    for it_im, it_gt in zip(images, labels):
        item = (os.path.join(img_path, it_im), os.path.join(mask_path, it_gt))
        items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        img = Image.open(img_path)  # .convert('RGB')
        mask = Image.open(mask_path)  # .convert('RGB')

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        return [img, mask, img_path]
