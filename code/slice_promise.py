#!/usr/bin/env python3.6

import re
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from numpy import unique as uniq
from skimage.transform import resize
from skimage.io import imread, imsave

from utils.utils import starmmap_, map_


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def get_p_id(path: Path, regex: str = "(Case\d+)(_segmentation)?") -> str:
    matched = re.match(regex, path.stem)

    if matched:
        return matched.group(1)
    raise ValueError(regex, path)


def save_slices(img_p: Path, gt_p: Path,
                dest_dir: Path, shape: Tuple[int],
                img_dir: str = "img", gt_dir: str = "gt") -> Tuple[int, int, int]:
    p_id: str = get_p_id(img_p)
    assert "Case" in p_id
    assert p_id == get_p_id(gt_p)

    # Load the data
    img = imread(str(img_p), plugin='simpleitk')
    gt = imread(str(gt_p), plugin='simpleitk')
    # print(img.shape, img.dtype, gt.shape, gt.dtype)
    # print(img.min(), img.max(), len(np.unique(img)))
    # print(np.unique(gt))

    assert img.shape == gt.shape
    assert img.dtype in [np.int16]
    assert gt.dtype in [np.int8]

    img_nib = sitk.ReadImage(str(img_p))
    dx, dy, dz = img_nib.GetSpacing()
    # print(dx, dy, dz)
    assert np.abs(dx - dy) <= 0.0000041, (dx, dy, dx - dy)
    assert 0.27 <= dx <= 0.75, dx
    assert 2.19994 <= dz <= 4.00001, dz

    x, y, z = img.shape
    assert (y, z) in [(320, 320), (512, 512), (256, 256), (384, 384)], (y, z)
    assert 15 <= x <= 54, x

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert norm_img.dtype == np.uint8

    save_dir_img: Path = dest_dir / img_dir
    save_dir_gt: Path = dest_dir / gt_dir
    save_dir_weak: Path = dest_dir / "weak"
    sizes_2d: np.ndarray = np.zeros(img.shape[-1])
    for j in range(len(img)):
        img_s = norm_img[j, :, :]
        gt_s = gt[j, :, :]
        assert img_s.shape == gt_s.shape

        # Resize and check the data are still what we expect
        # from time import time
        # tic = time()
        resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
        r_img: np.ndarray = resize_(img_s, shape).astype(np.uint8)
        r_gt: np.ndarray = resize_(gt_s, shape).astype(np.uint8)
        # print(time() - tic)
        assert r_img.dtype == r_gt.dtype == np.uint8
        assert 0 <= r_img.min() and r_img.max() <= 255  # The range might be smaller
        assert set(uniq(r_gt)).issubset(set(uniq(gt)))
        sizes_2d[j] = r_gt[r_gt == 1].sum()

        r_weak: np.ndarray = random_strat(r_gt, 1)

        r_gt *= 255
        r_weak *= 255

        for save_dir, data in zip([save_dir_img, save_dir_gt, save_dir_weak],
                                  [r_img, r_gt, r_weak]):
            filename = f"{p_id}_{j:02d}.png"
            save_dir.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(Path(save_dir, filename)), data)

    return sizes_2d.sum(), sizes_2d[sizes_2d > 0].min(), sizes_2d.max()


def random_strat(orig_mask: np.ndarray, filling: int) -> np.ndarray:
    res_arr: np.ndarray = np.zeros_like(orig_mask)

    size: int = orig_mask.sum()
    if size:  # Positive images
        res_img: Image.Image = Image.new("L", orig_mask.shape, 0)
        canvas = ImageDraw.Draw(res_img)
        xs, ys = np.where(orig_mask == 1)
        # print(len(xs), len(ys))
        assert len(xs) == len(ys)
        random_index: int = np.random.randint(len(xs))
        rx, ry = xs[random_index], ys[random_index]
        # Of course the coordinates are inverted
        # rx, ry = ry, rx
        # print(centroid, rx, ry)

        width: int = 5  # Hardcoded for now
        dw: int = int(width // 2)
        canvas.ellipse([rx - dw, ry - dw, rx + dw, ry + dw], fill=filling)

        # Remove overflow if needed
        masked_res: np.ndarray = np.einsum("hw,wh->wh", np.array(res_img), orig_mask).astype(np.uint8)
        res_arr = masked_res

    return res_arr


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    nii_paths: List[Path] = [p for p in src_path.rglob('*.mhd')]
    assert len(nii_paths) % 2 == 0, "Uneven number of .nii, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    img_nii_paths: List[Path] = sorted(p for p in nii_paths if "_segmentation" not in str(p))
    gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "_segmentation" in str(p))
    assert len(img_nii_paths) == len(gt_nii_paths)
    paths: List[Tuple[Path, Path]] = list(zip(img_nii_paths, gt_nii_paths))

    print(f"Found {len(img_nii_paths)} pairs in total")
    pprint(paths[:5])

    validation_paths: List[Tuple[Path, Path]] = random.sample(paths, args.retain)
    training_paths: List[Tuple[Path, Path]] = [p for p in paths if p not in validation_paths]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths))

    for mode, _paths in zip(["train", "val"], [training_paths, validation_paths]):
        img_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(img_paths)} pairs to {dest_dir}")
        assert len(img_paths) == len(gt_paths)

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape)
        sizes = starmmap_(pfun, zip(img_paths, gt_paths))
        # sizes = []
        # for paths in tqdm(list(zip(img_paths, gt_paths)), ncols=50):
        #     sizes.append(uc_(pfun)(paths))
        sizes_3d, sizes_2d_min, sizes_2d_max = map_(np.asarray, zip(*sizes))

        print("2d sizes: ", sizes_2d_min.min(), sizes_2d_max.max())
        print("3d sizes: ", sizes_3d.min(), sizes_3d.mean(), sizes_3d.max())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retain', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
