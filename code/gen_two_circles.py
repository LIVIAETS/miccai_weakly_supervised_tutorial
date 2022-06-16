#!/usr/bin/env python3.7

import random
import argparse
from pathlib import Path
from typing import Tuple
from functools import partial

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

# from utils import mmap_


def main(args) -> None:
    W, H = args.wh  # Tuple[int, int]
    r: int = args.r

    for folder, n_img in zip(["train", "val"], args.n):
        gt_folder: Path = Path(args.dest, folder, 'gt')
        img_folder: Path = Path(args.dest, folder, 'img')

        gt_folder.mkdir(parents=True, exist_ok=True)
        img_folder.mkdir(parents=True, exist_ok=True)

        gen_fn = partial(gen_img, W=W, H=W, r=r, gt_folder=gt_folder, img_folder=img_folder, ellipsis=args.ellipsis)

        # mmap_(gen_fn, range(n_img))
        for i in tqdm(range(n_img)):
            gen_fn(i)


def gen_img(i: int, W: int, H: int, r: int, gt_folder: Path, img_folder: Path, ellipsis: bool = False) -> None:
    img: Image = Image.new("L", (W, H), 0)
    gt: Image = Image.new("L", (W, H), 0)

    img_canvas = ImageDraw.Draw(img)
    gt_canvas = ImageDraw.Draw(gt)

    rx: int
    ry: int
    if not ellipsis:
        rx = r
        ry = r
    else:
        rx = np.random.randint(1, r)
        ry = np.random.randint(1, r)
    del r

    ax, ay = np.random.randint(rx, W - rx), np.random.randint(ry, H - ry)  # a for annoying
    img_canvas.ellipse([ax - rx, ay - ry, ax + rx, ay + ry], 255, 255)

    cx, cy = np.random.randint(rx, W - rx), np.random.randint(ry, H - ry)

    img_canvas.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], 125, 125)
    gt_canvas.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], 255, 255)

    img_arr: np.ndarray = np.asarray(img)
    with_noise: np.ndarray = noise(img_arr)

    filename: str = f"{i:05d}"
    gt.save(Path(gt_folder, filename).with_suffix(".png"))
    Image.fromarray(with_noise).save(Path(img_folder, filename).with_suffix(".png"))


def noise(arr: np.ndarray) -> np.ndarray:
    noise_level: int = np.random.randint(100)
    to_add = np.random.normal(0, noise_level, arr.shape).astype(np.int16).clip(-255, 255)

    return (arr + to_add).clip(0, 255).astype(np.uint8)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generation parameters')
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('-n', type=int, nargs=2, required=True)
    parser.add_argument('-wh', type=int, nargs=2, required=True, help="Size of image")
    parser.add_argument('-r', type=int, required=True, help="Radius of circle")
    parser.add_argument('--ellipsis', action='store_true')

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
